import uuid
import json
import traceback
from typing import Dict, Any, List
from datetime import datetime
import os

from app.core.config import get_settings
from app.services.openai_client import OpenAIClient
from app.stores.main_store import MainStore

settings = get_settings()

REFORM_PROMPT = """You reformulate finance user queries into a precise, self-contained search query.
Return JSON: {"reformulated": "string"}
If already precise, repeat it.
"""

DOC_SELECT_PROMPT = """You are given candidate document summaries with relevance scores (higher means more relevant).
Select the minimal subset that likely contains the answer; prefer higher scores when in doubt.
Return JSON: {"chosen_doc_ids": ["doc-..."], "reason": "string"}. If absolutely none relate, return an empty list and explain briefly.
"""

CHUNK_FILTER_PROMPT = """Given a user query and candidate chunks (text with IDs), select the IDs that are relevant. Also decide if question is answerable now.
Return JSON: {"relevant_chunk_ids": ["chunk-..."], "answerable": true/false, "missing_info_query": "string", "reason": "string"}
If not answerable, craft missing_info_query to retrieve new info.
"""

FINAL_ANSWER_PROMPT = """You are a finance QA assistant. Use only the provided chunks to answer. You may perform calculations. Return JSON: {"answer": "string", "reasoning": "string"}.
If numeric, include numeric form in answer.
"""

class QALoop:
    def __init__(self, store: MainStore):
        self.store = store
        self.emb = OpenAIClient()
        self._debug = settings.rag_debug

    def _chat(self, stage: str, system: str, prompt: str, schema: str):
        """Wrap chat_json adding debug print of (truncated) input and output."""
        if self._debug:
            print(f"[LLM-IN] stage={stage} sys={self._t(system,60)} prompt={self._t(prompt,220)} schema={schema}")
        try:
            resp = self.emb.chat_json(system, prompt, schema)
        except Exception as e:
            if self._debug:
                print(f"[LLM-ERR] stage={stage} error={e}")
            raise
        if self._debug:
            try:
                print(f"[LLM-OUT] stage={stage} json={self._t(json.dumps(resp),240)}")
            except Exception:
                print(f"[LLM-OUT] stage={stage} (non-serializable) resp={resp}")
        return resp

    def _t(self, text: str, limit: int = 180) -> str:
        if text is None:
            return ''
        return text if len(text) <= limit else text[:limit] + '…'

    def run(self, user_query: str) -> Dict[str, Any]:
        trace: Dict[str, Any] = {
            'id': str(uuid.uuid4()),
            'created_at': datetime.utcnow().isoformat(),
            'user_query': user_query,
            'steps': []
        }
        accumulated_chunks: Dict[str, Dict[str, Any]] = {}

        current_query = user_query
        for loop_idx in range(settings.iterative_max_loops):
            # Step 1: reformulate
            reform_json = self._chat('reformulate', settings.json_response_system_prompt, f"{REFORM_PROMPT}\nQuery: {current_query}", '{"reformulated":"string"}')
            reformulated = reform_json.get('reformulated', current_query)
            if self._debug:
                print(f"[RAG] loop={loop_idx} reformulated='{self._t(reformulated)}'")
            trace['steps'].append({'loop': loop_idx, 'type': 'reformulate', 'input': current_query, 'output': reformulated})

            # Step 2: doc retrieval (LangChain store directly uses text query)
            docs = self.store.retrieve_docs(reformulated, settings.top_k_docs)
            if self._debug:
                print(f"[RAG] loop={loop_idx} retrieved_docs={len(docs)} ids={[d['id'] for d in docs]}")
            trace['steps'].append({'loop': loop_idx, 'type': 'retrieve_docs', 'candidates': docs})
            # Step 3: doc selection via LLM
            doc_context = json.dumps([{ 'id': d['id'], 'score': round(d['score'],4), 'summary': d.get('summary_short', d.get('text')) } for d in docs])
            sel_json = self._chat('select_docs', settings.json_response_system_prompt, f"{DOC_SELECT_PROMPT}\nQuery: {reformulated}\nDocs: {doc_context}", '{"chosen_doc_ids":[],"reason":"string"}')
            chosen_ids = set(sel_json.get('chosen_doc_ids', []))
            if self._debug:
                print(f"[RAG] loop={loop_idx} selected_docs={list(chosen_ids)} reason={self._t(sel_json.get('reason',''))}")
            trace['steps'].append({'loop': loop_idx, 'type': 'select_docs', 'selection': list(chosen_ids), 'llm_raw': sel_json})

            # Step 4: chunk & table retrieval limited to chosen docs
            chunks = self.store.retrieve_chunks(reformulated, settings.top_k_chunks)
            tables = self.store.retrieve_tables(reformulated, settings.top_k_tables)
            # filter by chosen docs if any
            if chosen_ids:
                chunks = [c for c in chunks if 'doc-' + c['metadata'].get('source_file', '') in chosen_ids]
                tables = [t for t in tables if 'doc-' + t['metadata'].get('source_file', '') in chosen_ids]
            if self._debug:
                print(f"[RAG] loop={loop_idx} chunks={len(chunks)} tables={len(tables)} (after filter)")
            trace['steps'].append({'loop': loop_idx, 'type': 'retrieve_chunks', 'chunks': chunks, 'tables': tables})

            # Step 5: LLM chunk filtering & answerability
            limited_chunks = chunks[:8] + tables[:4]
            chunk_context = json.dumps([{ 'id': c['id'], 'text': c['text'][:500] } for c in limited_chunks])
            filter_json = self._chat('filter_chunks', settings.json_response_system_prompt, f"{CHUNK_FILTER_PROMPT}\nQuery: {reformulated}\nChunks: {chunk_context}", '{"relevant_chunk_ids":[],"answerable":false,"missing_info_query":"string","reason":"string"}')
            rel_ids = set(filter_json.get('relevant_chunk_ids', []))
            for c in limited_chunks:
                if c['id'] in rel_ids:
                    accumulated_chunks[c['id']] = c
            answerable = filter_json.get('answerable', False)
            if self._debug:
                print(f"[RAG] loop={loop_idx} selected_chunks={list(rel_ids)} answerable={answerable} missing={self._t(filter_json.get('missing_info_query',''))}")
            trace['steps'].append({'loop': loop_idx, 'type': 'filter_chunks', 'selected': list(rel_ids), 'answerable': answerable, 'llm_raw': filter_json})

            if answerable or loop_idx == settings.iterative_max_loops - 1:
                # final answer
                final_context = json.dumps([{ 'id': cid, 'text': c['text'][:1200] } for cid, c in accumulated_chunks.items()])
                final_json = self._chat('final_answer', settings.json_response_system_prompt, f"{FINAL_ANSWER_PROMPT}\nQuery: {user_query}\nChunks: {final_context}", '{"answer":"string","reasoning":"string"}')
                trace['steps'].append({'loop': loop_idx, 'type': 'final_answer', 'result': final_json})
                trace['final_answer'] = final_json
                if self._debug:
                    print(f"[RAG] loop={loop_idx} final_answer={self._t(final_json.get('answer',''))}")
                break
            else:
                current_query = filter_json.get('missing_info_query', current_query)

        return trace

    # Async style with event logger injection (to avoid circular import keep dynamic import)
    def run_with_events(self, user_query: str, logger) -> str:
        """Runs the loop emitting progress events. Returns trace_id."""
        settings = get_settings()
        trace_id = str(uuid.uuid4())
        trace: Dict[str, Any] = {
            'id': trace_id,
            'created_at': datetime.utcnow().isoformat(),
            'user_query': user_query,
            'steps': []
        }
        accumulated_chunks: Dict[str, Dict[str, Any]] = {}
        current_query = user_query
        logger.info('loop_start', trace_id=trace_id)
        for loop_idx in range(settings.iterative_max_loops):
            logger.progress('reformulate', loop_idx, settings.iterative_max_loops)
            reform_json = self._chat('reformulate', settings.json_response_system_prompt, f"{REFORM_PROMPT}\nQuery: {current_query}", '{"reformulated":"string"}')
            reformulated = reform_json.get('reformulated', current_query)
            if settings.rag_debug:
                print(f"[RAG] loop={loop_idx} reformulated='{self._t(reformulated)}'")
            trace['steps'].append({'loop': loop_idx, 'type': 'reformulate', 'input': current_query, 'output': reformulated})
            logger.progress('retrieve_docs', loop_idx, settings.iterative_max_loops)
            docs = self.store.retrieve_docs(reformulated, settings.top_k_docs)
            if settings.rag_debug:
                print(f"[RAG] loop={loop_idx} retrieved_docs={len(docs)} ids={[d['id'] for d in docs]} scores={[round(d.get('score',0.0),3) for d in docs]}")
            trace['steps'].append({'loop': loop_idx, 'type': 'retrieve_docs', 'candidates': docs})
            doc_context = json.dumps([
                {
                    'id': d['id'],
                    'score': round(d.get('score', 0.0), 4),
                    'summary': (
                        d.get('summary_short')
                        or d.get('summary')
                        or d.get('text', '')[:settings.doc_summary_max_chars]
                    )
                } for d in docs
            ])
            sel_json = self._chat('select_docs', settings.json_response_system_prompt, f"{DOC_SELECT_PROMPT}\nQuery: {reformulated}\nDocs: {doc_context}", '{"chosen_doc_ids":[],"reason":"string"}')
            chosen_ids = set(sel_json.get('chosen_doc_ids', []))
            if settings.rag_debug:
                print(f"[RAG] loop={loop_idx} selected_docs={list(chosen_ids)} reason={self._t(sel_json.get('reason',''))}")
            trace['steps'].append({'loop': loop_idx, 'type': 'select_docs', 'selection': list(chosen_ids), 'llm_raw': sel_json})
            logger.progress('retrieve_chunks', loop_idx, settings.iterative_max_loops)
            chunks = self.store.retrieve_chunks(reformulated, settings.top_k_chunks)
            tables = self.store.retrieve_tables(reformulated, settings.top_k_tables)
            if chosen_ids:
                chunks = [c for c in chunks if 'doc-' + c['metadata'].get('source_file', '') in chosen_ids]
                tables = [t for t in tables if 'doc-' + t['metadata'].get('source_file', '') in chosen_ids]
            if settings.rag_debug:
                print(f"[RAG] loop={loop_idx} chunks={len(chunks)} tables={len(tables)} (after filter)")
            trace['steps'].append({'loop': loop_idx, 'type': 'retrieve_chunks', 'chunks': chunks, 'tables': tables})
            limited_chunks = chunks[:8] + tables[:4]
            chunk_context = json.dumps([{ 'id': c['id'], 'text': c['text'][:500] } for c in limited_chunks])
            logger.progress('filter_chunks', loop_idx, settings.iterative_max_loops)
            filter_json = self._chat('filter_chunks', settings.json_response_system_prompt, f"{CHUNK_FILTER_PROMPT}\nQuery: {reformulated}\nChunks: {chunk_context}", '{"relevant_chunk_ids":[],"answerable":false,"missing_info_query":"string","reason":"string"}')
            rel_ids = set(filter_json.get('relevant_chunk_ids', []))
            for c in limited_chunks:
                if c['id'] in rel_ids:
                    accumulated_chunks[c['id']] = c
            answerable = filter_json.get('answerable', False)
            if settings.rag_debug:
                print(f"[RAG] loop={loop_idx} selected_chunks={list(rel_ids)} answerable={answerable} missing={self._t(filter_json.get('missing_info_query',''))}")
            trace['steps'].append({'loop': loop_idx, 'type': 'filter_chunks', 'selected': list(rel_ids), 'answerable': answerable, 'llm_raw': filter_json})
            if answerable or loop_idx == settings.iterative_max_loops - 1:
                logger.progress('final_answer', loop_idx, settings.iterative_max_loops)
                final_context = json.dumps([{ 'id': cid, 'text': c['text'][:1200] } for cid, c in accumulated_chunks.items()])
                final_json = self._chat('final_answer', settings.json_response_system_prompt, f"{FINAL_ANSWER_PROMPT}\nQuery: {user_query}\nChunks: {final_context}", '{"answer":"string","reasoning":"string"}')
                trace['steps'].append({'loop': loop_idx, 'type': 'final_answer', 'result': final_json})
                trace['final_answer'] = final_json
                if settings.rag_debug:
                    print(f"[RAG] loop={loop_idx} final_answer={self._t(final_json.get('answer',''))}")
                break
            else:
                current_query = filter_json.get('missing_info_query', current_query)
        # persist trace
        trace_path = os.path.join(settings.trace_dir, f"{trace_id}.json")
        with open(trace_path, 'w') as f:
            json.dump(trace, f)
        logger.info('trace_saved', trace_id=trace_id)
        logger.done(status='ok', trace_id=trace_id)
        return trace_id

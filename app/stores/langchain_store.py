import os
from typing import List, Dict, Any

from app.core.config import get_settings
from app.services.openai_client import OpenAIClient

settings = get_settings()

# Conditional imports to avoid heavy deps if not used
try:
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain.retrievers import BM25Retriever, EnsembleRetriever
except ImportError:  # pragma: no cover
    FAISS = None  # type: ignore
    RecursiveCharacterTextSplitter = None  # type: ignore
    OpenAIEmbeddings = None  # type: ignore
    BM25Retriever = None  # type: ignore
    EnsembleRetriever = None  # type: ignore

class LangChainStore:
    """LangChain-based retrieval implementation.

    Maintains three logical corpora: document summaries, chunks, tables.
    For each we build:
      - A FAISS dense vector index
      - A BM25 retriever
    Then we compose them via an EnsembleRetriever (weighted) per corpus.

    We expose a compatible API: retrieve_docs, retrieve_chunks, retrieve_tables
    returning list of dict(id,text,metadata,score,summary?,summary_short?).
    """

    def __init__(self):
        if FAISS is None:
            raise RuntimeError("LangChain not installed. Please add langchain-community and faiss-cpu to requirements.")
        self.emb_client = OpenAIClient()
        # We'll reuse embedding model via LangChain wrapper
        self.embedding = OpenAIEmbeddings(model=settings.embedding_model, openai_api_key=settings.openai_api_key)
        # corpora
        self._docs_texts: List[Dict[str, Any]] = []
        self._chunks_texts: List[Dict[str, Any]] = []
        self._tables_texts: List[Dict[str, Any]] = []
        # retrievers (built lazily)
        self._doc_retriever = None
        self._chunk_retriever = None
        self._table_retriever = None
        # persistence
        self.persist_dir = os.path.join(settings.persist_dir, 'lc')
        os.makedirs(self.persist_dir, exist_ok=True)
        self._load_persisted()

    def add_document(self, filename: str, summary: str, chunks: List[Dict[str, Any]], tables: List[Dict[str, Any]]):
        # store doc summary
        self._docs_texts.append({
            'id': f'doc-{filename}',
            'text': summary,
            'metadata': {'source_file': filename, 'type': 'summary'}
        })
        for c in chunks:
            self._chunks_texts.append({
                'id': c['id'],
                'text': c['text'],
                'metadata': {**c.get('metadata', {}), 'source_file': filename}
            })
        for t in tables:
            self._tables_texts.append({
                'id': t['id'],
                'text': t['text'],
                'metadata': {**t.get('metadata', {}), 'source_file': filename, 'type': 'table'}
            })
        # invalidate retrievers
        self._doc_retriever = None
        self._chunk_retriever = None
        self._table_retriever = None
        self._save_persisted()

    def _save_persisted(self):
        import json
        data = {
            'docs': self._docs_texts,
            'chunks': self._chunks_texts,
            'tables': self._tables_texts
        }
        with open(os.path.join(self.persist_dir, 'corpora.json'), 'w') as f:
            json.dump(data, f)

    def _load_persisted(self):
        import json
        path = os.path.join(self.persist_dir, 'corpora.json')
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                self._docs_texts = data.get('docs', [])
                self._chunks_texts = data.get('chunks', [])
                self._tables_texts = data.get('tables', [])
            except Exception as e:
                print(f"[WARN] Failed to load LangChain corpora: {e}")

    # building helpers
    def _build_retriever(self, corpus: List[Dict[str, Any]]):
        from langchain.schema import Document
        lc_docs = [Document(page_content=e['text'], metadata={'id': e['id'], **e['metadata']}) for e in corpus]
        if not lc_docs:
            return None
        # BM25
        bm25 = BM25Retriever.from_documents(lc_docs)
        # Dense
        faiss_vs = FAISS.from_documents(lc_docs, self.embedding)
        dense_retr = faiss_vs.as_retriever(search_kwargs={'k': 12})
        # Ensemble (weights approximate earlier alphas)
        ensemble = EnsembleRetriever(retrievers=[bm25, dense_retr], weights=[0.4, 0.6])
        return ensemble

    def _ensure(self):
        if self._doc_retriever is None:
            self._doc_retriever = self._build_retriever(self._docs_texts)
        if self._chunk_retriever is None:
            self._chunk_retriever = self._build_retriever(self._chunks_texts)
        if self._table_retriever is None:
            self._table_retriever = self._build_retriever(self._tables_texts)

    # retrieval API
    def retrieve_docs(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        self._ensure()
        if self._doc_retriever is None:
            return []
        docs = self._doc_retriever.get_relevant_documents(query)
        out = []
        for d in docs[:top_k]:
            doc_id = d.metadata.get('id')
            txt = d.page_content
            trunc = txt if len(txt) <= settings.doc_summary_max_chars else txt[:settings.doc_summary_max_chars] + '…'
            out.append({'id': doc_id, 'text': txt, 'metadata': d.metadata, 'score': d.metadata.get('score', 0.0), 'summary': txt, 'summary_short': trunc})
        return out

    def retrieve_chunks(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        self._ensure()
        if self._chunk_retriever is None:
            return []
        docs = self._chunk_retriever.get_relevant_documents(query)
        out = []
        for d in docs[:top_k]:
            out.append({'id': d.metadata.get('id'), 'text': d.page_content, 'metadata': d.metadata, 'score': d.metadata.get('score', 0.0)})
        return out

    def retrieve_tables(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        self._ensure()
        if self._table_retriever is None:
            return []
        docs = self._table_retriever.get_relevant_documents(query)
        out = []
        for d in docs[:top_k]:
            out.append({'id': d.metadata.get('id'), 'text': d.page_content, 'metadata': d.metadata, 'score': d.metadata.get('score', 0.0)})
        return out


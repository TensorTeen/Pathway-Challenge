import os
import uuid
from typing import List, Dict, Any
import numpy as np

from app.core.config import get_settings
from app.stores.vector_base import SimpleVectorStore, VectorRecord
from app.stores.hybrid_search import HybridIndex
from app.services.pdf_loader import PDFLoader
from app.services.openai_client import OpenAIClient

settings = get_settings()

class MainStore:
    def __init__(self):
        base = settings.persist_dir
        self.summary_store = SimpleVectorStore(os.path.join(base, 'summaries'))
        self.chunk_store = SimpleVectorStore(os.path.join(base, 'chunks'))
        self.table_store = SimpleVectorStore(os.path.join(base, 'tables'))
        self.emb = OpenAIClient()
        self.pdf_loader = PDFLoader(settings.chunk_size, settings.chunk_overlap)
        # hybrid indices (rebuilt on demand)
        self._hybrid_chunks = HybridIndex()
        self._hybrid_tables = HybridIndex()
        self._hybrid_docs = HybridIndex()
        self._rebuild_hybrids()
        # track ingested filenames to avoid duplicate ingestion on scans
        self._ingested = set(self.list_files())

    def _rebuild_hybrids(self):
        self._hybrid_chunks.build(self.chunk_store.vectors)
        self._hybrid_tables.build(self.table_store.vectors)
        self._hybrid_docs.build(self.summary_store.vectors)

    def load_pdf(self, file_path: str) -> Dict[str, Any]:
        parsed = self.pdf_loader.load(file_path)
        filename = os.path.basename(file_path)
        # summary
        coverage_text = parsed['full_text'][:settings.summary_chars]
        summary = self.emb.summarize(coverage_text)
        sum_vec = self.emb.embed_texts([summary])[0]
        sum_record = VectorRecord(id=f"doc-{filename}", vector=sum_vec, text=summary, metadata={'source_file': filename, 'type': 'summary'})
        self.summary_store.add([sum_record])
        # chunks
        chunk_texts = [c.text for c in parsed['chunks']]
        chunk_vecs = self.emb.embed_texts(chunk_texts)
        chunk_records = []
        for c, vec in zip(parsed['chunks'], chunk_vecs):
            meta = c.metadata
            chunk_records.append(VectorRecord(id=c.id + '-' + filename, vector=vec, text=c.text, metadata=meta))
        self.chunk_store.add(chunk_records)
        # tables
        table_texts = [t.text for t in parsed['tables']]
        table_vecs = self.emb.embed_texts(table_texts) if table_texts else []
        table_records = []
        for t, vec in zip(parsed['tables'], table_vecs):
            table_records.append(VectorRecord(id=t.id + '-' + filename, vector=vec, text=t.text, metadata=t.metadata))
        if table_records:
            self.table_store.add(table_records)
        self._rebuild_hybrids()
        return {
            'filename': filename,
            'summary': summary,
            'num_chunks': len(chunk_records),
            'num_tables': len(table_records)
        }

    def load_pdf_streaming(self, file_path: str, logger=None, batch_size: int = None) -> Dict[str, Any]:
        """Streaming / staged ingestion with progress events.

        Steps:
        1. Parse PDF (text + tables)
        2. Summarize coverage region
        3. Embed & add summary
        4. Embed chunks in batches (progress events)
        5. Embed tables in batches
        6. Rebuild hybrids at end
        """
        # default batch size from settings if not provided
        if batch_size is None:
            batch_size = settings.embedding_batch_size
        filename = os.path.basename(file_path)
        if logger: logger.info('parse_start', filename=filename)
        if settings.parse_debug:
            print(f"[INGEST] parse_start file={filename}")
        parsed = self.pdf_loader.load(file_path)
        if logger: logger.info('parse_complete', chunks=len(parsed['chunks']), tables=len(parsed['tables']))
        if settings.parse_debug:
            print(f"[INGEST] parse_complete chunks={len(parsed['chunks'])} tables={len(parsed['tables'])}")
        # summary
        coverage_text = parsed['full_text'][:settings.summary_chars]
        if logger: logger.info('summary_start')
        if settings.parse_debug:
            print(f"[INGEST] summary_start file={filename} chars={len(coverage_text)}")
        summary = self.emb.summarize(coverage_text)
        sum_vec = self.emb.embed_texts([summary])[0]
        self.summary_store.add([VectorRecord(id=f"doc-{filename}", vector=sum_vec, text=summary, metadata={'source_file': filename, 'type': 'summary'})], flush=False)
        if logger: logger.info('summary_done')
        if settings.parse_debug:
            print(f"[INGEST] summary_done")
        # chunks batched
        chunks = parsed['chunks']
        total_chunks = len(chunks)
        if logger: logger.info('chunk_embedding_start', total=total_chunks)
        if settings.parse_debug:
            print(f"[INGEST] chunk_embedding_start total={total_chunks} batch_size={batch_size}")
        chunk_records: List[VectorRecord] = []
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i+batch_size]
            texts = [c.text for c in batch]
            vecs = self.emb.embed_texts(texts)
            for c, v in zip(batch, vecs):
                chunk_records.append(VectorRecord(id=c.id + '-' + filename, vector=v, text=c.text, metadata=c.metadata))
            # periodic progress
            if logger:
                logger.progress('chunks', min(i+batch_size, total_chunks), total_chunks)
            if settings.parse_debug:
                print(f"[INGEST] chunk_batch_done upto={min(i+batch_size, total_chunks)}/{total_chunks}")
        self.chunk_store.add(chunk_records, flush=False)
        if logger: logger.info('chunk_embedding_done', count=total_chunks)
        if settings.parse_debug:
            print(f"[INGEST] chunk_embedding_done count={total_chunks}")
        # tables batched
        table_records: List[VectorRecord] = []
        tables = parsed['tables']
        total_tables = len(tables)
        if tables:
            if logger: logger.info('table_embedding_start', total=total_tables)
            if settings.parse_debug:
                print(f"[INGEST] table_embedding_start total={total_tables}")
            for i in range(0, total_tables, batch_size):
                batch = tables[i:i+batch_size]
                texts = [t.text for t in batch]
                vecs = self.emb.embed_texts(texts)
                for t, v in zip(batch, vecs):
                    table_records.append(VectorRecord(id=t.id + '-' + filename, vector=v, text=t.text, metadata=t.metadata))
                if logger:
                    logger.progress('tables', min(i+batch_size, total_tables), total_tables)
                if settings.parse_debug:
                    print(f"[INGEST] table_batch_done upto={min(i+batch_size, total_tables)}/{total_tables}")
            self.table_store.add(table_records, flush=False)
            if logger: logger.info('table_embedding_done', count=total_tables)
            if settings.parse_debug:
                print(f"[INGEST] table_embedding_done count={total_tables}")
        # flush all stores once
        self.summary_store.flush(); self.chunk_store.flush(); self.table_store.flush()
        if logger: logger.info('stores_flushed')
        if settings.parse_debug:
            print(f"[INGEST] stores_flushed")
        self._rebuild_hybrids()
        if logger: logger.info('hybrids_rebuilt')
        if settings.parse_debug:
            print(f"[INGEST] hybrids_rebuilt")
        meta = {
            'filename': filename,
            'summary': summary,
            'num_chunks': len(chunk_records),
            'num_tables': len(table_records)
        }
        if logger: logger.done(**meta)
        if settings.parse_debug:
            print(f"[INGEST] done file={filename} chunks={len(chunk_records)} tables={len(table_records)}")
        return meta

    def delete_file(self, filename: str):
        self.summary_store.delete_by_file(filename)
        self.chunk_store.delete_by_file(filename)
        self.table_store.delete_by_file(filename)
        self._rebuild_hybrids()

    def list_files(self) -> List[str]:
        files = set()
        for store in [self.summary_store, self.chunk_store, self.table_store]:
            for r in store.vectors:
                files.add(r.metadata.get('source_file'))
        return sorted(files)

    def scan_folder(self, logger=None, force: bool = False) -> Dict[str, Any]:
        """Scan the configured watch_dir and ingest any new PDFs.

        Args:
            logger: optional EventLogger-like object for progress events
            force: if True, re-ingest even if file previously seen
        Returns summary dict with counts.
        """
        watch = settings.watch_dir
        if not os.path.isdir(watch):
            return {"scanned": 0, "ingested": 0, "files": []}
        pdfs = [f for f in os.listdir(watch) if f.lower().endswith('.pdf')]
        ingested = []
        if logger: logger.info('scan_start', watch_dir=watch, total=len(pdfs))
        for name in pdfs:
            if not force and name in self._ingested:
                continue
            path = os.path.join(watch, name)
            try:
                meta = self.load_pdf_streaming(path, logger=None)  # internal logger optional
                self._ingested.add(name)
                ingested.append({"filename": name, **meta})
                if logger: logger.info('file_ingested', filename=name, chunks=meta['num_chunks'], tables=meta['num_tables'])
            except Exception as e:
                if logger: logger.error('file_failed', filename=name, error=str(e))
        if logger: logger.done(status='ok', ingested=len(ingested))
        return {"scanned": len(pdfs), "ingested": len(ingested), "files": ingested}

    # retrieval methods
    def retrieve_docs(self, query: str, query_vec: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        res = self._hybrid_docs.hybrid_search(query, query_vec, settings.alpha_doc, top_k)
        docs = []
        for r, score in res:
            packed = self._pack(r, score)
            # provide explicit summary field (original text is already summary for doc vectors)
            txt = packed['text']
            trunc = txt if len(txt) <= settings.doc_summary_max_chars else txt[:settings.doc_summary_max_chars] + '…'
            packed['summary'] = txt
            packed['summary_short'] = trunc
            docs.append(packed)
        return docs

    def retrieve_chunks(self, query: str, query_vec: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        res = self._hybrid_chunks.hybrid_search(query, query_vec, settings.alpha_chunk, top_k)
        return [self._pack(r, score) for r, score in res]

    def retrieve_tables(self, query: str, query_vec: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        res = self._hybrid_tables.hybrid_search(query, query_vec, settings.alpha_table, top_k)
        return [self._pack(r, score) for r, score in res]

    def _pack(self, rec: VectorRecord, score: float) -> Dict[str, Any]:
        return {
            'id': rec.id,
            'text': rec.text,
            'metadata': rec.metadata,
            'score': score
        }

import os
from typing import List, Dict, Any

from app.core.config import get_settings
from app.stores.langchain_store import LangChainStore
from app.services.pdf_loader import PDFLoader
from app.services.openai_client import OpenAIClient

settings = get_settings()

class MainStore:
    def __init__(self):
        self.emb = OpenAIClient()
        self.pdf_loader = PDFLoader(settings.chunk_size, settings.chunk_overlap)
        self.lc_store = LangChainStore()

    def load_pdf(self, file_path: str) -> Dict[str, Any]:
        parsed = self.pdf_loader.load(file_path)
        filename = os.path.basename(file_path)
        # summary
        coverage_text = parsed['full_text'][:settings.summary_chars]
        summary = self.emb.summarize(coverage_text)
        # LangChain path only
        # chunks
        chunk_records = parsed['chunks']
        # tables
        table_records = parsed['tables']
        # Commit to LangChain store directly
        self.lc_store.add_document(
            filename,
            summary,
            [ {'id': c.id + '-' + filename, 'text': c.text, 'metadata': c.metadata} for c in chunk_records ],
            [ {'id': t.id + '-' + filename, 'text': t.text, 'metadata': t.metadata} for t in table_records ]
        )
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
        # LangChain path only
        if logger: logger.info('summary_done')
        if settings.parse_debug:
            print(f"[INGEST] summary_done")
        # chunks batched
        chunks = parsed['chunks']
        total_chunks = len(chunks)
        if logger: logger.info('chunk_embedding_start', total=total_chunks)
        if settings.parse_debug:
            print(f"[INGEST] chunk_embedding_start total={total_chunks} batch_size={batch_size}")
        chunk_records = []
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i+batch_size]
            texts = [c.text for c in batch]
            # embeddings handled by LangChain retriever build later; store raw
            for c in batch:
                chunk_records.append(c)
            # periodic progress
            if logger:
                logger.progress('chunks', min(i+batch_size, total_chunks), total_chunks)
            if settings.parse_debug:
                print(f"[INGEST] chunk_batch_done upto={min(i+batch_size, total_chunks)}/{total_chunks}")
        if logger: logger.info('chunk_embedding_done', count=total_chunks)
        if settings.parse_debug:
            print(f"[INGEST] chunk_embedding_done count={total_chunks}")
        # tables batched
        table_records = []
        tables = parsed['tables']
        total_tables = len(tables)
        if tables:
            if logger: logger.info('table_embedding_start', total=total_tables)
            if settings.parse_debug:
                print(f"[INGEST] table_embedding_start total={total_tables}")
            for i in range(0, total_tables, batch_size):
                batch = tables[i:i+batch_size]
                texts = [t.text for t in batch]
                for t in batch:
                    table_records.append(t)
                if logger:
                    logger.progress('tables', min(i+batch_size, total_tables), total_tables)
                if settings.parse_debug:
                    print(f"[INGEST] table_batch_done upto={min(i+batch_size, total_tables)}/{total_tables}")
            if logger: logger.info('table_embedding_done', count=total_tables)
            if settings.parse_debug:
                print(f"[INGEST] table_embedding_done count={total_tables}")
        # commit to LangChain store once at end
        self.lc_store.add_document(
            filename,
            summary,
            [ {'id': c.id + '-' + filename, 'text': c.text, 'metadata': c.metadata} for c in chunk_records ],
            [ {'id': t.id + '-' + filename, 'text': t.text, 'metadata': t.metadata} for t in table_records ]
        )
        if logger: logger.info('stores_flushed')
        if settings.parse_debug:
            print(f"[INGEST] stores_flushed")
        if logger: logger.info('langchain_store_updated')
        if settings.parse_debug:
            print(f"[INGEST] langchain_store_updated")
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
        def _filter(corpus):
            return [e for e in corpus if e.get('metadata', {}).get('source_file') != filename]
        self.lc_store._docs_texts = _filter(self.lc_store._docs_texts)
        self.lc_store._chunks_texts = _filter(self.lc_store._chunks_texts)
        self.lc_store._tables_texts = _filter(self.lc_store._tables_texts)
        self.lc_store._doc_retriever = None
        self.lc_store._chunk_retriever = None
        self.lc_store._table_retriever = None

    def list_files(self) -> List[str]:
        files = {e['metadata'].get('source_file') for e in (self.lc_store._docs_texts + self.lc_store._chunks_texts + self.lc_store._tables_texts)}
        return sorted([f for f in files if f])

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
            path = os.path.join(watch, name)
            try:
                meta = self.load_pdf_streaming(path, logger=None)
                ingested.append({"filename": name, **meta})
                if logger: logger.info('file_ingested', filename=name, chunks=meta['num_chunks'], tables=meta['num_tables'])
            except Exception as e:
                if logger: logger.error('file_failed', filename=name, error=str(e))
        if logger: logger.done(status='ok', ingested=len(ingested))
        return {"scanned": len(pdfs), "ingested": len(ingested), "files": ingested}

    # retrieval methods
    def retrieve_docs(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        return self.lc_store.retrieve_docs(query, top_k)

    def retrieve_chunks(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        return self.lc_store.retrieve_chunks(query, top_k)

    def retrieve_tables(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        return self.lc_store.retrieve_tables(query, top_k)

    # _pack no longer needed (removed custom store logic)

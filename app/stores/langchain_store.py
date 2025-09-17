import os
import uuid
from typing import List, Dict, Any

from app.core.config import get_settings
from app.services.openai_client import OpenAIClient

settings = get_settings()

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant as LCQdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


class LangChainStore:
    """Simple dense vector retrieval using Qdrant.

    Maintains three Qdrant collections: docs, chunks, tables.
    Each uses OpenAI embeddings for dense semantic search.
    """

    def __init__(self):
        self.emb_client = OpenAIClient()
        self.embedding = OpenAIEmbeddings(model=settings.embedding_model, openai_api_key=settings.openai_api_key)
        # persistence directory & embedded Qdrant
        self.persist_dir = os.path.join(settings.persist_dir, 'qdrant')
        os.makedirs(self.persist_dir, exist_ok=True)
        self.qdrant = QdrantClient(path=self.persist_dir)
        self.col_docs = 'docs'
        self.col_chunks = 'chunks'
        self.col_tables = 'tables'
        # vectorstore instances
        self._docs_vs = None
        self._chunks_vs = None
        self._tables_vs = None
        self._ensure_collections()
        self._load_persisted()

    # ---------------- Persistence ----------------
    def _save_persisted(self):
        # No longer needed for pure Qdrant approach
        pass

    def _load_persisted(self):
        # Initialize vectorstores pointing to existing collections
        self._docs_vs = LCQdrant(client=self.qdrant, collection_name=self.col_docs, embeddings=self.embedding)
        self._chunks_vs = LCQdrant(client=self.qdrant, collection_name=self.col_chunks, embeddings=self.embedding)
        self._tables_vs = LCQdrant(client=self.qdrant, collection_name=self.col_tables, embeddings=self.embedding)

    def _ensure_collections(self):
        dim = 1536  # OpenAI text-embedding-3-small dimension
        existing = {c.name for c in self.qdrant.get_collections().collections}
        for name in [self.col_docs, self.col_chunks, self.col_tables]:
            if name not in existing:
                self.qdrant.create_collection(
                    collection_name=name,
                    vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE)
                )

    # ---------------- Adding Documents ----------------
    def add_document(self, filename: str, summary: str, chunks: List[Dict[str, Any]], tables: List[Dict[str, Any]]):
        if settings.rag_debug:
            print(f"[ADD_DOC] filename={filename} summary_len={len(summary) if summary else 0} chunks={len(chunks)} tables={len(tables)}")
        
        # Add document summary
        if summary:
            try:
                doc_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f'doc-{filename}'))
                self._docs_vs.add_texts(
                    texts=[summary],
                    metadatas=[{'source_file': filename, 'type': 'summary', 'original_id': f'doc-{filename}'}],
                    ids=[doc_uuid]
                )
                if settings.rag_debug:
                    print(f"[ADD_DOC] Added summary for {filename}")
            except Exception as e:
                print(f"[ADD_DOC] ERROR adding summary for {filename}: {e}")
        
        # Add chunks
        if chunks:
            try:
                texts = [c['text'] for c in chunks]
                metadatas = [{**c.get('metadata', {}), 'source_file': filename, 'original_id': c['id']} for c in chunks]
                # Generate UUIDs for chunk IDs
                ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, c['id'])) for c in chunks]
                self._chunks_vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                if settings.rag_debug:
                    print(f"[ADD_DOC] Added {len(chunks)} chunks for {filename}")
            except Exception as e:
                print(f"[ADD_DOC] ERROR adding chunks for {filename}: {e}")
        
        # Add tables
        if tables:
            try:
                texts = [t['text'] for t in tables]
                metadatas = [{**t.get('metadata', {}), 'source_file': filename, 'type': 'table', 'original_id': t['id']} for t in tables]
                # Generate UUIDs for table IDs
                ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, t['id'])) for t in tables]
                self._tables_vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                if settings.rag_debug:
                    print(f"[ADD_DOC] Added {len(tables)} tables for {filename}")
            except Exception as e:
                print(f"[ADD_DOC] ERROR adding tables for {filename}: {e}")

    def delete_file(self, filename: str):
        # Delete from each vectorstore by filtering metadata
        # Note: LangChain Qdrant doesn't have direct delete by metadata
        # This is a limitation - would need custom implementation or rebuild
        pass
    
    def list_files(self) -> List[str]:
        # Get unique source files from any collection
        try:
            points, _ = self.qdrant.scroll(collection_name=self.col_chunks, limit=1000)
            files = set()
            for p in points:
                payload = p.payload or {}
                source_file = payload.get('source_file')
                if source_file:
                    files.add(source_file)
                if settings.rag_debug and len(files) < 3:  # Debug first few
                    print(f"[LIST_FILES_DEBUG] payload keys: {list(payload.keys())}, source_file: {source_file}")
            print(f"[LIST_FILES] found {len(files)} files: {list(files)}")
            return sorted(list(files))
        except Exception as e:
            print(f"[LIST_FILES] ERROR: {e}")
            return []
    
    def get_collection_info(self):
        """Debug method to check collection status"""
        info = {}
        for col_name in [self.col_docs, self.col_chunks, self.col_tables]:
            try:
                collection_info = self.qdrant.get_collection(col_name)
                info[col_name] = {
                    'vectors_count': collection_info.vectors_count,
                    'points_count': collection_info.points_count
                }
            except Exception as e:
                info[col_name] = {'error': str(e)}
        if settings.rag_debug:
            print(f"[COLLECTION_INFO] {info}")
        return info

    def ensure_built(self):
        # No-op for dense-only approach - just check collections
        self.get_collection_info()


    # ---------------- Retrieval API ----------------
    def retrieve_docs(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        docs = self._docs_vs.similarity_search(query, k=top_k)
        if settings.rag_debug:
            print(f"[RETRIEVE] docs raw_count={len(docs)} requested_top_k={top_k}")
        
        out = []
        for d in docs:
            doc_id = d.metadata.get('original_id', d.metadata.get('id', 'unknown'))
            txt = d.page_content
            trunc = txt if len(txt) <= settings.doc_summary_max_chars else txt[:settings.doc_summary_max_chars] + '…'
            out.append({
                'id': doc_id, 
                'text': txt, 
                'metadata': d.metadata, 
                'score': 0.0,  # similarity_search doesn't return scores by default
                'summary': txt, 
                'summary_short': trunc
            })
        return out

    def retrieve_chunks(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        docs = self._chunks_vs.similarity_search(query, k=top_k)
        if settings.rag_debug:
            print(f"[RETRIEVE] chunks raw_count={len(docs)} requested_top_k={top_k}")
        
        out = []
        for d in docs:
            out.append({
                'id': d.metadata.get('original_id', d.metadata.get('id', 'unknown')), 
                'text': d.page_content, 
                'metadata': d.metadata, 
                'score': 0.0
            })
        return out

    def retrieve_tables(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        docs = self._tables_vs.similarity_search(query, k=top_k)
        if settings.rag_debug:
            print(f"[RETRIEVE] tables raw_count={len(docs)} requested_top_k={top_k}")
        
        out = []
        for d in docs:
            out.append({
                'id': d.metadata.get('original_id', d.metadata.get('id', 'unknown')), 
                'text': d.page_content, 
                'metadata': d.metadata, 
                'score': 0.0
            })
        return out


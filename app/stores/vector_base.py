import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class VectorRecord:
    id: str
    vector: np.ndarray
    text: str
    metadata: Dict[str, Any]

class SimpleVectorStore:
    """A lightweight persistent vector store with cosine similarity search.

    Supports optional batched persistence: pass flush=False to add() and then
    call flush() once after a bulk ingestion to cut down disk I/O.
    """
    def __init__(self, persist_path: str):
        self.persist_path = persist_path
        self.vectors: List[VectorRecord] = []
        self._id_index: Dict[str, int] = {}
        os.makedirs(persist_path, exist_ok=True)
        self.meta_file = os.path.join(persist_path, 'meta.json')
        self.vec_file = os.path.join(persist_path, 'vectors.npz')
        if os.path.exists(self.meta_file) and os.path.exists(self.vec_file):
            self._load()
        self._dirty = False

    def add(self, records: List[VectorRecord], flush: bool = True):
        for rec in records:
            if rec.id in self._id_index:
                # update existing
                idx = self._id_index[rec.id]
                self.vectors[idx] = rec
            else:
                self._id_index[rec.id] = len(self.vectors)
                self.vectors.append(rec)
        if flush:
            self._persist()
        else:
            self._dirty = True

    def delete_by_file(self, filename: str):
        keep: List[VectorRecord] = []
        for rec in self.vectors:
            if rec.metadata.get('source_file') != filename:
                keep.append(rec)
        self.vectors = keep
        self._id_index = {rec.id: i for i, rec in enumerate(self.vectors)}
        self._persist()

    def _persist(self):
        meta = [
            {
                'id': r.id,
                'text': r.text,
                'metadata': r.metadata
            } for r in self.vectors
        ]
        np.savez_compressed(self.vec_file, arr=np.vstack([r.vector for r in self.vectors]) if self.vectors else np.zeros((0, 1)))
        with open(self.meta_file, 'w') as f:
            json.dump(meta, f)
        self._dirty = False

    def flush(self):
        if self._dirty:
            self._persist()

    def _load(self):
        with open(self.meta_file, 'r') as f:
            meta = json.load(f)
        arr = np.load(self.vec_file)['arr']
        self.vectors = []
        self._id_index = {}
        for i, m in enumerate(meta):
            vec = arr[i]
            rec = VectorRecord(id=m['id'], vector=vec, text=m['text'], metadata=m['metadata'])
            self._id_index[rec.id] = i
            self.vectors.append(rec)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[VectorRecord, float]]:
        if not self.vectors:
            return []
        mat = np.vstack([r.vector for r in self.vectors])  # (n,d)
        # cosine similarity
        q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        mnorm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        sims = mnorm @ q
        idxs = np.argsort(-sims)[:top_k]
        return [(self.vectors[i], float(sims[i])) for i in idxs]

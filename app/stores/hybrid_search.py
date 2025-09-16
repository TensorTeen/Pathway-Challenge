from typing import List, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from .vector_base import VectorRecord
import re

class HybridIndex:
    def __init__(self):
        self.records: List[VectorRecord] = []
        self.bm25 = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t.lower() for t in re.findall(r"\w+", text)]

    def build(self, records: List[VectorRecord]):
        self.records = records
        # filter out records with no alphanumeric tokens for BM25 corpus
        corpus = []
        for r in records:
            toks = self._tokenize(r.text) if r.text else []
            if toks:
                corpus.append(toks)
        if corpus:
            try:
                self.bm25 = BM25Okapi(corpus)
            except ZeroDivisionError:
                # extremely small / pathological corpus (shouldn't happen after filtering) -> disable bm25
                self.bm25 = None
        else:
            self.bm25 = None

    def lexical_scores(self, query: str) -> np.ndarray:
        if not self.records or not self.bm25:
            return np.zeros((0,))
        toks = self._tokenize(query)
        scores = self.bm25.get_scores(toks)
        scores = np.array(scores)
        if scores.size:
            rng = np.ptp(scores)
            if rng > 0:
                scores = (scores - scores.min()) / (rng + 1e-9)
        return scores

    def dense_scores(self, query_vec: np.ndarray) -> np.ndarray:
        if not self.records:
            return np.zeros((0,))
        mat = np.vstack([r.vector for r in self.records])
        q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        mnorm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        sims = mnorm @ q
        rng = np.ptp(sims)
        if rng > 0:
            sims = (sims - sims.min()) / (rng + 1e-9)
        return sims

    def hybrid_search(self, query: str, query_vec: np.ndarray, alpha: float, top_k: int) -> List[Tuple[VectorRecord, float]]:
        if not self.records:
            return []
        ls = self.lexical_scores(query)
        ds = self.dense_scores(query_vec)
        # if lexical scores unavailable (size mismatch), fall back to pure dense
        if ls.shape != ds.shape or ls.size == 0:
            score = ds
        else:
            score = alpha * ds + (1 - alpha) * ls
        idxs = np.argsort(-score)[:top_k]
        return [(self.records[i], float(score[i])) for i in idxs]

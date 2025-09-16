import re
from typing import Iterable, List, Dict, Any
from dataclasses import dataclass

@dataclass
class TextChunk:
    text: str
    start: int
    end: int

class Chunker:
    """Configurable chunker supporting multiple strategies.

    Strategies:
    - fixed: sliding window with overlap
    - sentence: group sentences up to max size
    - recursive: fallback style (sentence grouping; if still too large, split)
    """
    def __init__(self, strategy: str, chunk_size: int, overlap: int, max_chunk_size: int, sentence_regex: str):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_chunk_size = max_chunk_size
        self.sentence_regex = re.compile(sentence_regex)

    def chunk(self, text: str) -> List[TextChunk]:
        if self.strategy == 'fixed':
            return list(self._fixed(text))
        elif self.strategy == 'sentence':
            return list(self._sentence_group(text))
        elif self.strategy == 'recursive':
            return list(self._recursive(text))
        else:
            # fallback to fixed
            return list(self._fixed(text))

    def _fixed(self, text: str) -> Iterable[TextChunk]:
        start = 0
        n = len(text)
        while start < n:
            end = min(start + self.chunk_size, n)
            yield TextChunk(text=text[start:end], start=start, end=end)
            if end == n:
                break
            start = end - self.overlap
            if start < 0:
                start = 0

    def _split_sentences(self, text: str) -> List[str]:
        # naive sentence splitting
        parts = self.sentence_regex.split(text)
        # reintroduce delimiters is skipped for simplicity; acceptable for grouping
        return [p.strip() for p in parts if p.strip()]

    def _sentence_group(self, text: str) -> Iterable[TextChunk]:
        sentences = self._split_sentences(text)
        buf: List[str] = []
        buf_len = 0
        cursor = 0
        for s in sentences:
            if not buf:
                chunk_start = text.find(s, cursor)
            tentative_len = buf_len + (1 if buf else 0) + len(s)
            if tentative_len > self.chunk_size and buf:
                chunk_text = ' '.join(buf)
                chunk_end = chunk_start + len(chunk_text)
                yield TextChunk(text=chunk_text, start=chunk_start, end=chunk_end)
                buf = [s]
                buf_len = len(s)
                cursor = chunk_end
                chunk_start = text.find(s, cursor)
            else:
                buf.append(s)
                buf_len += (1 if buf_len>0 else 0) + len(s)
        if buf:
            chunk_text = ' '.join(buf)
            chunk_end = chunk_start + len(chunk_text)
            yield TextChunk(text=chunk_text, start=chunk_start, end=chunk_end)

    def _recursive(self, text: str) -> Iterable[TextChunk]:
        # Start with sentence groups up to max_chunk_size; if a sentence itself exceeds, fallback to fixed split inside
        for group in self._sentence_group(text):
            if len(group.text) <= self.max_chunk_size:
                yield group
            else:
                # fall back to fixed splitting inside this group.text
                for sub in self._fixed(group.text):
                    # adjust offsets relative to original text
                    yield TextChunk(text=sub.text, start=group.start + sub.start, end=group.start + sub.end)

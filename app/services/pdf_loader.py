import fitz  # PyMuPDF
from typing import List, Dict, Any, Iterable
from dataclasses import dataclass
import re
import time
from tqdm import tqdm
from app.services.chunking import Chunker

@dataclass
class ParsedChunk:
    id: str
    text: str
    metadata: Dict[str, Any]

@dataclass
class ParsedTable:
    id: str
    text: str  # representation used for embedding
    raw: List[List[str]]
    metadata: Dict[str, Any]

class PDFLoader:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        # legacy parameters retained; actual behavior controlled by settings + Chunker
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        from app.core.config import get_settings
        s = get_settings()
        self._chunker = Chunker(
            strategy=s.chunk_strategy,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            max_chunk_size=s.max_chunk_size,
            sentence_regex=s.sentence_split_regex,
        )

    def load(self, file_path: str) -> Dict[str, Any]:
        from app.core.config import get_settings  # local import to avoid cycles
        settings = get_settings()
        doc = fitz.open(file_path)
        full_text_pages: List[str] = []
        chunks: List[ParsedChunk] = []
        tables: List[ParsedTable] = []
        filename = file_path.split('/')[-1]

        start_time = time.time()
        if settings.parse_debug:
            print(f"[PARSE] file={filename} pages={len(doc)} simple={settings.simple_pdf_parser}")

        # Fast path: just concatenate page text, skip block & table scans
        if settings.simple_pdf_parser:
            t_pages_start = time.time()
            for page_index in range(len(doc)):
                page = doc[page_index]
                text = page.get_text("text")  # simplest text extraction
                full_text_pages.append(text)
            if settings.parse_debug:
                print(f"[PARSE] collected_pages elapsed={time.time()-t_pages_start:.3f}s chars={sum(len(p) for p in full_text_pages)} hi")
            full_text = "\n".join(full_text_pages)
            # chunking
            if settings.parse_debug:
                print(f"[PARSE] collected_pages elapsed={time.time()-t_pages_start:.3f}s chars={sum(len(p) for p in full_text_pages)} hi")
            t_chunk_start = time.time()
            chunk_counter = 0
            chunk_objs = self._chunker.chunk(full_text)
            if settings.parse_debug:
                print(f"[PARSE] strategy={settings.chunk_strategy} planned_chunks={len(chunk_objs)}")
            for ch in tqdm(chunk_objs, desc="Chunking text", disable=not settings.parse_debug):
                chunks.append(ParsedChunk(
                    id=f"chunk-{chunk_counter}",
                    text=ch.text,
                    metadata={
                        'source_file': filename,
                        'type': 'chunk',
                        'char_start': ch.start,
                        'char_end': ch.end,
                    }
                ))
                chunk_counter += 1
            # optional table extraction disabled in fast mode unless explicitly enabled and flag on
            if settings.enable_table_extraction and not settings.simple_pdf_parser:
                pass  # unreachable branch, kept for clarity
            if settings.parse_debug:
                print(f"[PARSE] chunking elapsed={time.time()-t_chunk_start:.3f}s chunks={len(chunks)}")
                print(f"[PARSE] done total_elapsed={time.time()-start_time:.3f}s")
            return {'full_text': full_text, 'chunks': chunks, 'tables': tables}

        # Original (richer) path with naive table detection
        table_counter = 0
        chunk_counter = 0
        t_pages_start = time.time()
        for page_index in range(len(doc)):
            page = doc[page_index]
            text = page.get_text("text")
            full_text_pages.append(text)
            if settings.enable_table_extraction:
                blocks = page.get_text("blocks")
                for b in blocks:
                    btext = b[4]
                    if self._looks_like_table(btext):
                        rows = self._table_rows(btext)
                        if rows:
                            header = rows[0]
                            first_col = [r[0] for r in rows[1:]] if len(rows) > 1 and rows[0] else []
                            rep = " | ".join(header) + " || " + " ; ".join(first_col[:15])
                            tables.append(ParsedTable(
                                id=f"table-{table_counter}",
                                text=rep[:2000],
                                raw=rows,
                                metadata={'page': page_index, 'source_file': filename, 'type': 'table'}
                            ))
                            table_counter += 1
        if settings.parse_debug:
            print(f"[PARSE] pages+tables elapsed={time.time()-t_pages_start:.3f}s tables={len(tables)}")
        full_text = "\n".join(full_text_pages)
        t_chunk_start = time.time()
        chunk_objs = self._chunker.chunk(full_text)
        for ch in chunk_objs:
            chunks.append(ParsedChunk(
                id=f"chunk-{chunk_counter}",
                text=ch.text,
                metadata={'source_file': filename, 'type': 'chunk', 'char_start': ch.start, 'char_end': ch.end}
            ))
            chunk_counter += 1
        if settings.parse_debug:
            print(f"[PARSE] chunking elapsed={time.time()-t_chunk_start:.3f}s chunks={len(chunks)} total_elapsed={time.time()-start_time:.3f}s")
        return {'full_text': full_text, 'chunks': chunks, 'tables': tables}

    # _chunk_iter removed in favor of Chunker class

    def _looks_like_table(self, text: str) -> bool:
        lines = [l for l in text.splitlines() if l.strip()]
        if len(lines) < 2:
            return False
        pattern = re.compile(r"\S+\s{2,}\S+\s{2,}\S+")
        hits = sum(1 for ln in lines if pattern.search(ln))
        return hits >= 2

    def _table_rows(self, text: str) -> List[List[str]]:
        rows = []
        for line in text.splitlines():
            if not line.strip():
                continue
            cols = re.split(r"\s{2,}", line.strip())
            rows.append(cols)
        return rows

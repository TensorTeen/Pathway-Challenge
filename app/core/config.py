from pydantic import BaseModel
from functools import lru_cache
import os

class Settings(BaseModel):
    # IMPORTANT: Set OPENAI_API_KEY in environment; default left blank intentionally.
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "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")
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini-2024-07-18"
    summary_chars: int = 5000

    chunk_size: int = 1200
    chunk_overlap: int = 150
    embedding_batch_size: int = 32
    event_buffer_flush_events: int = 5  # how many events before disk flush

    table_max_rows: int = 50
    table_max_cols: int = 30

    # Hybrid search weights
    alpha_doc: float = 0.65  # weight for dense vs lexical in doc retrieval
    alpha_chunk: float = 0.55
    alpha_table: float = 0.50

    persist_dir: str = "data/persist"
    trace_dir: str = "data/traces"
    events_dir: str = "data/events"
    watch_dir: str = "data/inbox"  # directory where user drops PDFs
    auto_scan_on_start: bool = True  # automatically ingest new PDFs at startup
    simple_pdf_parser: bool = False  # fast path: page text only, no block/table scan
    enable_table_extraction: bool = True  # allow disabling table detection for speed
    parse_debug: bool = True  # verbose parsing / ingestion prints to stdout
    rag_debug: bool = True  # verbose RAG pipeline (query reformulation, retrieval steps)
    chunk_strategy: str = "recursive"  # one of: fixed, sentence, recursive
    max_chunk_size: int = 1200  # upper bound for adaptive strategies
    sentence_split_regex: str = r"(?<=[.!?])\s+"  # basic sentence boundary
    doc_summary_max_chars: int = 600  # truncate summary shown to selection LLM

    top_k_docs: int = 6
    top_k_chunks: int = 12
    top_k_tables: int = 6
    iterative_max_loops: int = 4

    json_response_system_prompt: str = (
        "You are a helpful finance domain assistant. You MUST ALWAYS respond with valid JSON matching the requested schema. "
        "Do not include any prose outside JSON."
    )

@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    os.makedirs(settings.persist_dir, exist_ok=True)
    os.makedirs(settings.trace_dir, exist_ok=True)
    os.makedirs(settings.events_dir, exist_ok=True)
    os.makedirs(settings.watch_dir, exist_ok=True)
    return settings

# Finance QA System (LangChain Retrieval Only)

This project implements a multi-stage iterative Retrieval-Augmented Generation (RAG) system for finance PDFs using a LangChain retrieval backend (FAISS + BM25 Ensemble) plus an iterative reasoning loop and a FastAPI + Streamlit interface.

## Retrieval Pipeline
For each uploaded PDF (parsed via **PyMuPDF**):
1. Summarize initial coverage text (LLM) for document-level representation.
2. Chunk the full text with a configurable strategy (fixed / sentence / recursive).
3. Optionally extract heuristic tables.
4. Persist raw corpora (summaries, chunks, tables) to `data/persist/lc/corpora.json`.
5. On first query, build BM25 + FAISS vector indexes and compose them with `EnsembleRetriever` (weights 0.4 lexical / 0.6 dense).

Indexes are rebuilt lazily (cold start cost) to keep persistence simple and version-agnostic.

## Iterative QA Loop
Per question:
1. LLM reformulates query (JSON).
2. Ensemble retrieval over summaries.
3. LLM selects minimal doc subset.
4. Retrieve chunks/tables from selected docs.
5. LLM filters chunks, decides answerability, may emit refined query.
6. Repeat until answerable or loop limit, then final answer (JSON) using accumulated chunks.

Every LLM interaction enforces JSON-only responses with simple schema prompts. A full trace (steps, selections, reasoning) is stored under `data/traces/{trace_id}.json`.

## Interfaces
### FastAPI Endpoints
Synchronous:
- `GET /files` – list indexed PDFs
- `POST /upload` – upload a PDF and index it
- `DELETE /files/{filename}` – remove a PDF from all stores
- `POST /question` – run QA loop; returns final answer + trace
- `POST /explain` – convert stored trace to human-readable explanation
- `GET /traces` – list stored traces (metadata)

Asynchronous (with progress events):
- `POST /upload_async` – start async ingestion, returns `{job_id}`
- `POST /question_async` – start async QA, returns `{job_id}`
- `GET /jobs/{job_id}` – poll JSONL events: progress, info, errors, completion

### Streamlit UI (`app/ui/app.py`)
Tabs:
- Documents: async upload with live progress + delete
- Ask: async question handling with progress + recent events + final explanation
- Explain: fetch explanation for a given trace ID

## Configuration
Centralized in `app/core/config.py` (chunk sizes, overlap, model names, loop limits, summary length, parsing flags). Set `OPENAI_API_KEY` for real embeddings & chat. Without a key, deterministic fallbacks allow offline dev (reduced quality).

## Running Locally
Install dependencies:
```bash
pip install -r requirements.txt
```
Export OpenAI key:
```bash
export OPENAI_API_KEY=sk-...
```
Start API:
```bash
uvicorn app.api.server:app --reload --port 8000
```
Start UI (new terminal):
```bash
streamlit run app/ui/app.py
```
Visit UI (default): http://localhost:8501

### Avoiding Excessive Reloads During Development
The `--reload` flag watches the entire working directory. Large or frequent writes under `data/` (embeddings, events, traces) can trigger rapid reload cycles. Prefer narrowing reload watch to `app/`.

The `--reload` flag watches the entire working directory. Large or frequent writes under `data/` (embeddings, events, traces) can trigger rapid reload cycles that look like the server is "shutting down" repeatedly. To mitigate:

1. Narrow the watch scope to application code only:
```bash
uvicorn app.api.server:app --reload --reload-dir app --port 8000
```
2. Or run without reload for heavy ingestion tests:
```bash
uvicorn app.api.server:app --port 8000
```
3. Optionally move persistence dirs outside the watched tree, e.g.:
```bash
export PERSIST_BASE=../qa_runtime_data
mkdir -p "$PERSIST_BASE"/{persist,traces,events}
ln -s "$PERSIST_BASE"/persist data/persist
ln -s "$PERSIST_BASE"/traces data/traces
ln -s "$PERSIST_BASE"/events data/events
```
4. Event logging & vector persistence are already batched to reduce churn; tune thresholds in `config.py` (`event_buffer_flush_events`, `embedding_batch_size`).

If you still see repeating "Shutting down" messages, confirm they correlate with file change reloads (expected) rather than crashes (check stack traces). No stack trace usually indicates just an auto-reload.

## Persistence
Embeddings & metadata stored under `data/persist/` (three subfolders). Traces under `data/traces/`. Event streams (progress logs) stored as JSONL under `data/events/` (one file per `job_id`).

## Folder-Based Ingestion (Drop PDFs)
Instead of calling the upload endpoint you can simply place PDF files into the watched directory defined by `watch_dir` in `config.py` (default: `data/inbox`). On API startup, if `auto_scan_on_start=True`, any new PDFs found there are ingested automatically.

Manual scan endpoint:
`POST /scan_folder` (optional query param `force=true` to re-ingest existing files)

Example curl:
```bash
curl -X POST 'http://localhost:8000/scan_folder'
```

Workflow:
1. Drop `report_Q1.pdf` into `data/inbox/`
2. (Optional) Trigger manual scan via `/scan_folder`
3. File is parsed & embedded; progress not event-streamed (synchronous) but batched for performance
4. The file now appears in `GET /files` and is available to the QA loop

Note: The legacy `/upload` and `/upload_async` endpoints still work; folder ingestion is an alternate path that simplifies deployment when documents arrive via external sync processes.

## Performance: Simplified PDF Parser
The default configuration now enables a fast parsing path (`simple_pdf_parser=True`) that:
- Extracts plain page text only (no block / table scans)
- Skips table detection unless you set `enable_table_extraction=True` AND `simple_pdf_parser=False`

Config flags in `config.py`:
- `simple_pdf_parser` (default True) — fastest ingestion; only text
- `enable_table_extraction` (default True) — honored only when `simple_pdf_parser` is False

To restore table extraction and richer parsing:
```python
simple_pdf_parser = False
enable_table_extraction = True
```
Then restart the API. Expect slower ingestion due to block analysis.

## Chunking Strategies
You can now choose how text is chunked before embedding via `chunk_strategy` in `config.py`:

Strategies:
- `fixed` (default): Sliding window of `chunk_size` with `chunk_overlap` (previous behavior, fastest, position-stable).
- `sentence`: Groups sentences together up to `chunk_size` (tries to avoid splitting sentences mid-way; chunk sizes vary).
- `recursive`: Sentence grouping first; if any grouped unit still exceeds `max_chunk_size`, it is split using the fixed window fallback.

Additional settings:
- `max_chunk_size`: Upper bound used by `recursive` strategy.
- `sentence_split_regex`: Basic regex to separate sentences; adjust for more/less aggressiveness.

Metadata for each chunk still includes `char_start` and `char_end` so you can map embeddings back to original document spans regardless of strategy.

Switch example (sentence grouping):
```python
chunk_strategy = "sentence"
```
Or recursive with safety limit:
```python
chunk_strategy = "recursive"
max_chunk_size = 1400
```
After changing, restart the API and re-ingest documents (existing persisted chunks will keep their old segmentation until reprocessed).

## Table Extraction Heuristic
Tables are detected via multi-space column alignment in block text. Representation = header row joined + sample of first column values. Improve by integrating structured table extraction (future work).

## Development Notes
- Fallback deterministic hash embeddings allow offline dev but produce meaningless semantic rankings.
- JSON parsing uses minimal repair logic; consider stricter schema validation.
- Hybrid weighting is tunable (alpha values in config) per store.

## Roadmap Ideas
- Add evaluation harness (precision/recall of retrieval & answer accuracy)
- Better table embeddings (cell co-occurrence, numeric feature vectors)
- Async processing & job queue for large PDF ingestion
- Caching reformulated queries & partial traces for multi-user sessions
- Guardrail validation for LLM JSON (pydantic schemas)

## Disclaimer
This system is a prototype for finance document Q&A. Always verify critical financial outputs; no warranty of correctness.

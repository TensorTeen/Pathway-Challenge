<div align="center">

# Finance Iterative RAG for Annual & 10-K Filings

**Dense-only hierarchical retrieval with iterative evidence curation for long, repetitive financial reports.**

</div>

## 1. Abstract
Financial annual reports (10-K, shareholder letters) exhibit high structural redundancy across years and issuers: repeated risk factor frames, boilerplate MD&A passages, tabular sections referencing identical KPIs with different period values. Naïve single‑pass RAG over raw chunks wastes retrieval budget on duplicated scaffolding and often surfaces semantically generic passages (e.g. “We face competition…”) instead of numerically grounded evidence. This project implements a lean, fully JSON‑disciplined, iterative Retrieval‑Augmented Generation loop inspired by hierarchical evidence collection (cf. recent multi‑stage RAG work such as HiREC‑style doc→chunk narrowing) but simplified for production pragmatics: (1) summarize each PDF for coarse document‑level routing, (2) LLM chooses candidate documents, (3) dense retrieval over fine‑grained chunks + optional tables limited to those docs, (4) LLM filters / assesses answerability, (5) loop continues with a targeted “missing info” reformulation until sufficient grounded context exists to answer. All retrieval is dense semantic search over Qdrant (no hybrid BM25, no custom scoring), minimizing operational complexity while preserving iterative focus. Full reasoning traces are persisted for auditability.

## 2. Key Design Goals & Rationale
| Goal | Rationale | Implementation Highlight |
|------|-----------|--------------------------|
| Hierarchical narrowing | Reduce wasted chunk lookups across large multi‑year corpora | summary → doc selection → chunk/table retrieval |
| Deterministic JSON I/O | Easy downstream consumption & guardrails | Every LLM call enforces minimal JSON schema |
| Minimal infra surface | Lower maintenance & cold‑start complexity | Single dense vector backend (Qdrant embedded) |
| Iterative evidence accrual | Avoid premature answering / hallucination | Missing info query drives additional loops |
| Auditability | FinReg / compliance style traceability | Structured per‑step trace in `data/traces/` |

## 3. Current Architecture (Simplified)
```
User Question
	│
	▼
Reformulate (LLM JSON)
	│ reformulated_query
	▼
Doc-Level Dense Retrieval (Qdrant 'docs' collection: summaries)
	│ candidates
	▼
Doc Selection (LLM JSON)
	│ chosen_doc_ids
	▼
Chunk/Table Retrieval (Qdrant 'chunks' & 'tables' collections, filtered by chosen docs)
	│ evidence candidates
	▼
Chunk Filtering + Answerability (LLM JSON)
	├─ if answerable → Final Answer (LLM JSON)
	└─ else → Missing Info Query → next loop (max N)
```

Collections:
* docs: 1 summary vector per PDF (LLM generated)
* chunks: sliding / sentence / recursive chunked text
* tables: lightweight textual table projections (heuristic extraction)

All vectors use OpenAI `text-embedding-3-small` (dimension 1536) stored in embedded Qdrant under `data/persist/qdrant`.

## 4. Data Ingestion Pipeline
1. Parse PDF → full text + (optional) tables (`PDFLoader`).
2. Generate document summary using chat model (first N chars window).
3. Chunk text (strategy configurable: fixed, sentence, recursive).
4. Deterministically generate UUIDv5 IDs for summary, each chunk & table; store original IDs in metadata for trace continuity.
5. Insert into three Qdrant collections via LangChain `LCQdrant` wrapper.
6. No hybrid ensemble, no BM25, no lazy rebuild step required.

Auto‑scan: On API start, PDFs placed in `data/inbox/` are ingested if `auto_scan_on_start=True`.

## 5. Iterative QA Loop (Detailed)
Loop Variables: `current_query`, `accumulated_chunks`.
Per iteration (max `iterative_max_loops`):
1. Reformulate → JSON {reformulated}
2. Retrieve doc summaries (`top_k_docs`).
3. Doc selection → JSON {chosen_doc_ids, reason}
4. Retrieve chunks (`top_k_chunks`) & tables (`top_k_tables`) globally then filter by chosen docs.
5. Filter / answerability → JSON {relevant_chunk_ids, answerable, missing_info_query}
6. If answerable or last loop → Final answer JSON {answer, reasoning}; else set `current_query = missing_info_query` and continue.

All steps appended into a persisted trace file `<trace_id>.json` with timestamps & loop index.

## 6. FastAPI Surface
| Method | Path | Purpose |
|--------|------|---------|
| GET | /files | List PDFs (currently requires collection scan; simplification WIP) |
| POST | /upload | Synchronous single PDF ingestion |
| POST | /upload_async | Async ingestion with event log |
| POST | /scan | Alias for folder scan ingestion |
| POST | /scan_folder | Synchronous watch directory scan |
| DELETE | /files/{filename} | Delete (stub – pending implementation) |
| POST | /question | Run QA loop & return full trace |
| POST | /question_async | Async QA with event stream |
| POST | /explain | Human-readable textual summary of a stored trace |
| GET | /traces | List stored traces metadata |
| GET | /trace/{id} | Fetch full trace JSON |
| GET | /jobs/{job_id} | Poll events for async jobs |
| GET | /health | Collection counts & heartbeat |

Event logs: JSONL per `job_id` in `data/events/`.

## 7. Streamlit UI (Lean)
The UI (`app/ui/app.py`) now exposes only:
* Ask: async question submit → shows progress events & final answer
* Explain: fetch explanation for a trace ID

Document upload / deletion tab was intentionally removed to reduce surface area.

## 8. Configuration (`app/core/config.py`)
Important fields:
* `openai_api_key` – provide via env var
* `chunk_strategy` – fixed | sentence | recursive
* `chunk_size`, `chunk_overlap`, `max_chunk_size`
* `top_k_docs`, `top_k_chunks`, `top_k_tables`, `iterative_max_loops`
* `simple_pdf_parser`, `enable_table_extraction`
* Debug toggles: `rag_debug`, `parse_debug`
* `doc_summary_max_chars`, `summary_chars`

Deprecated: `retrieval_alpha` (legacy hybrid) – slated for removal.

## 9. Installation & Quickstart
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
uvicorn app.api.server:app --reload --reload-dir app --port 8000
```
Optional UI:
```bash
streamlit run app/ui/app.py
```
Drop PDFs into `data/inbox/` or call `/upload`; then ask questions via UI or:
```bash
curl -X POST localhost:8000/question -H 'Content-Type: application/json' \
  -d '{"question": "What was Berkshire Hathaway net earnings in 2023?"}'
```

## 10. Financial Document Retrieval Challenges Addressed
| Challenge | Naïve Failure Mode | Mitigation Here |
|-----------|--------------------|-----------------|
| Boilerplate repetition | Top-k filled with generic risk text | Summaries + selection bias retrieval toward numerics |
| Cross-year drift of metrics | Mixed contexts from different years | Per-doc ID names + selection gating |
| Long tables converted to text | Embedding noise & truncation | Separate 'tables' collection, capped row/col sampling |
| Early hallucination | LLM answers after first partial retrieval | Iterative answerability gate |
| Traceability for compliance | Hard to audit answer provenance | Structured JSON trace with evidence IDs |

## 11. Persistence Layout
```
data/
  inbox/          # drop PDFs for auto-scan
  persist/
	 qdrant/       # embedded Qdrant collections (docs, chunks, tables)
  traces/         # per-answer trace JSON files
  events/         # async job event logs (.jsonl)
```

## 12. Extensibility Hooks
| Area | Extend By |
|------|-----------|
| Embeddings | Swap `OpenAIEmbeddings` for local model wrapper |
| Document parsing | Replace `PDFLoader` with structured XBRL/SEC parser |
| Table handling | Add richer semantic representation (cell type features) |
| Answer validation | Add post‑hoc numeric consistency checker |
| Evaluation | Insert retrieval hit-rate & answer accuracy harness |

## 13. Limitations / Known Gaps
* `/files` listing currently incomplete (scroll over chunks only; multi-collection aggregation WIP).
* `/delete` endpoint logic not yet implemented (requires point ID lookup & delete by metadata).
* No deduplication of near-identical boilerplate across issuers (could add MinHash cluster pruning).
* Single embedding model across summary/chunk/table may be suboptimal for numeric-heavy tables.
* No guardrail on numeric extraction accuracy (future: regex & reconciliation pass).

## 14. Roadmap (Short Horizon)
1. Implement collection-spanning file listing + deletion.
2. Remove `retrieval_alpha` and residual hybrid code paths.
3. Add batch question runner & evaluation notebook (precision / latency stats).
4. Introduce score-returning similarity interface (for debug ranking transparency).
5. Lightweight provenance rendering (highlight spans per chunk ID).

## 15. Disclaimer
Prototype system; outputs are not investment advice. Verify all financial figures against primary filings.

---
Questions or ideas? Open an issue or extend the loop logic in `app/services/qa_loop.py`.

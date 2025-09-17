from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import json
import uuid
import threading
import traceback

from app.stores.main_store import MainStore
from app.services.qa_loop import QALoop
from app.core.config import get_settings
from app.services.event_logger import EventLogger

settings = get_settings()

app = FastAPI(title="Finance QA System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = MainStore()
qa = QALoop(store)

# Optional startup scan
if settings.auto_scan_on_start:
    try:
        store.scan_folder()
    except Exception:
        pass

class QuestionRequest(BaseModel):
    question: str

class ExplainRequest(BaseModel):
    trace_id: str

class QuestionAsyncRequest(BaseModel):
    question: str

class JobStatusResponse(BaseModel):
    job_id: str
    events: list

@app.get("/files")
def list_files():
    return {"files": store.list_files()}

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, 'Only PDF files supported')
    dest_path = os.path.join('data', file.filename)
    with open(dest_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    meta = store.load_pdf(dest_path)
    return {"status": "ok", **meta}

@app.post("/upload_async")
def upload_pdf_async(background: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, 'Only PDF files supported')
    job_id = str(uuid.uuid4())
    dest_path = os.path.join('data', file.filename)
    with open(dest_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    logger = EventLogger(job_id)
    def task():
        try:
            # Use streaming ingestion; batch size derived from settings unless overridden
            meta = store.load_pdf_streaming(dest_path, logger=logger)
        except Exception as e:
            logger.error('ingest_failed', error=str(e), traceback=traceback.format_exc())
    background.add_task(task)
    return {"job_id": job_id, "status": "started"}

@app.post("/scan")
def scan_folder_alt():
    # Alternative endpoint name for scan
    result = store.scan_folder()
    return result

@app.post("/scan_folder")
def scan_folder(force: bool = False):
    # Simple synchronous scan (could be made async with events similar to uploads)
    result = store.scan_folder(force=force)
    return result

@app.delete("/files/{filename}")
def delete_file(filename: str):
    store.delete_file(filename)
    return {"status": "deleted", "filename": filename}

@app.post("/question")
def ask(req: QuestionRequest):
    trace = qa.run(req.question)
    trace_path = os.path.join(settings.trace_dir, f"{trace['id']}.json")
    with open(trace_path, 'w') as f:
        json.dump(trace, f)
    return trace

@app.post("/question_async")
def ask_async(req: QuestionAsyncRequest):
    job_id = str(uuid.uuid4())
    logger = EventLogger(job_id)
    def task():
        try:
            logger.info('qa_loop_start', question=req.question)
            trace = qa.run(req.question)
            trace_path = os.path.join(settings.trace_dir, f"{trace['id']}.json")
            with open(trace_path, 'w') as f:
                json.dump(trace, f)
            logger.info('qa_loop_complete', trace_id=trace['id'])
            logger.done(status='ok', trace_id=trace['id'])
        except Exception as e:
            logger.error('qa_failed', error=str(e), traceback=traceback.format_exc())
    threading.Thread(target=task, daemon=True).start()
    return {"job_id": job_id, "status": "started"}

@app.post("/explain")
def explain(req: ExplainRequest):
    trace_path = os.path.join(settings.trace_dir, f"{req.trace_id}.json")
    if not os.path.exists(trace_path):
        raise HTTPException(404, 'Trace not found')
    with open(trace_path, 'r') as f:
        trace = json.load(f)
    explanation = _explain_trace(trace)
    return {"trace_id": req.trace_id, "explanation": explanation}

@app.get("/jobs/{job_id}")
def job_events(job_id: str):
    events = EventLogger.read(job_id)
    return {"job_id": job_id, "events": events}

@app.get("/traces")
def list_traces():
    traces = []
    if os.path.exists(settings.trace_dir):
        for fname in os.listdir(settings.trace_dir):
            if fname.endswith('.json'):
                path = os.path.join(settings.trace_dir, fname)
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    traces.append({
                        'id': data.get('id', fname[:-5]),
                        'created_at': data.get('created_at'),
                        'user_query': data.get('user_query'),
                        'steps': len(data.get('steps', [])),
                        'has_final_answer': 'final_answer' in data
                    })
                except Exception:
                    continue
    traces = sorted(traces, key=lambda x: x.get('created_at') or '', reverse=True)
    return {'traces': traces}

@app.get("/health")
def health():
    try:
        # Get collection info from simplified store
        collection_info = store.lc_store.get_collection_info()
        counts = {
            'docs': collection_info.get('docs', {}).get('points_count', 0),
            'chunks': collection_info.get('chunks', {}).get('points_count', 0),
            'tables': collection_info.get('tables', {}).get('points_count', 0)
        }
        return {"status": "ok", "backend": "langchain", **counts}
    except Exception as e:
        return {"status": "error", "backend": "langchain", "error": str(e)}

@app.get("/trace/{trace_id}")
def get_trace(trace_id: str):
    """Return full stored trace JSON including final answer and steps."""
    trace_path = os.path.join(settings.trace_dir, f"{trace_id}.json")
    if not os.path.exists(trace_path):
        raise HTTPException(404, 'Trace not found')
    with open(trace_path, 'r') as f:
        trace = json.load(f)
    return trace

def _explain_trace(trace):
    parts = [f"Answering question: {trace['user_query']}"]
    for step in trace.get('steps', []):
        t = step['type']
        if t == 'reformulate':
            parts.append(f"Loop {step['loop']}: Reformulated query -> {step['output']}")
        elif t == 'retrieve_docs':
            parts.append(f"Retrieved {len(step['candidates'])} candidate summaries")
        elif t == 'select_docs':
            parts.append(f"Selected docs: {step['selection']}")
        elif t == 'retrieve_chunks':
            parts.append(f"Retrieved {len(step['chunks'])} chunks and {len(step['tables'])} tables")
        elif t == 'filter_chunks':
            parts.append(f"Filter selected {len(step['selected'])} chunks, answerable={step['answerable']}")
        elif t == 'final_answer':
            parts.append(f"Final answer produced")
    return "\n".join(parts)

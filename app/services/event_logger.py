import json
import os
import threading
import time
from typing import Dict, Any, List, Optional
from app.core.config import get_settings

settings = get_settings()
_lock = threading.Lock()

class EventLogger:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.path = os.path.join(settings.events_dir, f"{job_id}.jsonl")
        self._buffer: List[str] = []
        self._flush_every = settings.event_buffer_flush_events
        self._write_event('job_started', {})

    def _write_event(self, event_type: str, payload: Dict[str, Any]):
        evt = {
            'ts': time.time(),
            'event': event_type,
            'data': payload
        }
        line = json.dumps(evt, ensure_ascii=False)
        self._buffer.append(line)
        if len(self._buffer) >= self._flush_every or event_type in ('error', 'job_finished'):
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        with _lock:
            with open(self.path, 'a') as f:
                f.write("\n".join(self._buffer) + "\n")
        self._buffer.clear()

    def info(self, message: str, **kwargs):
        self._write_event('info', {'message': message, **kwargs})

    def progress(self, stage: str, current: int, total: int, **kwargs):
        pct = float(current) / float(total) if total else 0.0
        self._write_event('progress', {'stage': stage, 'current': current, 'total': total, 'pct': pct, **kwargs})

    def error(self, message: str, **kwargs):
        self._write_event('error', {'message': message, **kwargs})

    def done(self, **kwargs):
        self._write_event('job_finished', kwargs)
        # Ensure all buffered events are on disk
        self._flush()

    @staticmethod
    def read(job_id: str) -> List[Dict[str, Any]]:
        path = os.path.join(settings.events_dir, f"{job_id}.jsonl")
        if not os.path.exists(path):
            return []
        out = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out

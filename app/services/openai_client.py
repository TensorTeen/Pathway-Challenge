import os
import json
from typing import List, Dict, Any
import numpy as np
import hashlib
import time

from app.core.config import get_settings

try:
    from openai import OpenAI
except ImportError:  # placeholder if openai not installed yet
    OpenAI = None

settings = get_settings()

class OpenAIClient:
    def __init__(self):
        self.api_key = settings.openai_api_key
        if OpenAI and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []
        if self.client is None:
            # deterministic pseudo embedding fallback
            return [self._fake_embed(t) for t in texts]
        model = settings.embedding_model
        # openai new python client embedding usage (adapt if needed)
        resp = self.client.embeddings.create(model=model, input=texts)
        out = []
        for d in resp.data:
            out.append(np.array(d.embedding, dtype=np.float32))
        return out

    def chat_json(self, system: str, user: str, schema_desc: str) -> Dict[str, Any]:
        prompt = f"You MUST respond ONLY with valid JSON. Schema: {schema_desc}. If unsure, output an empty JSON object matching schema keys.\nUser Query: {user}" 
        if self.client is None:
            # Improved deterministic fallback: attempt to create JSON matching schema keys
            try:
                schema_obj = json.loads(schema_desc)
            except Exception:
                # if schema string not valid JSON just wrap
                return {"response": user[:160]}
            out: Dict[str, Any] = {}
            lower_user = user.lower()
            for k, v in schema_obj.items():
                # heuristics based on key name
                if k in ('reformulated', 'missing_info_query'):
                    # echo a trimmed question or keep same if already question-like
                    out[k] = user.strip()[:140]
                elif k in ('reason', 'reasoning'):
                    out[k] = "fallback reasoning"
                elif k.startswith('chosen_doc'):
                    out[k] = []
                elif k.startswith('relevant_chunk'):
                    out[k] = []
                elif k == 'answer':
                    out[k] = "fallback answer based on provided context"
                elif k == 'answerable':
                    out[k] = False
                elif k == 'summary':
                    out[k] = user[:200]
                else:
                    # generic placeholder by type inference
                    if isinstance(v, list):
                        out[k] = []
                    elif isinstance(v, bool):
                        out[k] = False
                    elif isinstance(v, (int, float)):
                        out[k] = 0
                    else:
                        out[k] = ""
            return out
        resp = self.client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        content = resp.choices[0].message.content
        # attempt parse
        for _ in range(2):
            try:
                return json.loads(content)
            except Exception:
                # minimal repair: extract between first and last braces
                if '{' in content and '}' in content:
                    content = content[content.find('{'):content.rfind('}')+1]
                else:
                    return {"raw": content}
        return {"raw": content}

    def summarize(self, text: str, max_chars: int = 1200) -> str:
        snippet = text[:max_chars]
        schema = '{"summary": "string"}'
        data = self.chat_json(settings.json_response_system_prompt, f"Summarize the following finance document snippet:\n{snippet}", schema)
        return data.get('summary', '')

    def _fake_embed(self, text: str) -> np.ndarray:
        # simple hashing to stable vector
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], 'little'))
        return rng.standard_normal(256).astype(np.float32)

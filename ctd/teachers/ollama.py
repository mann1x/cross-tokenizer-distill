"""Ollama teacher backend (local or LAN).

Ollama has no echo/forced-decode mode, so we run one HTTP call per
teacher-token position with a growing prompt prefix. Server-side KV
caching softens the GPU cost but each call still pays an HTTP roundtrip.
Practical only for small corpora or smoke tests.

For production-scale precompute against a massive teacher, prefer
`OpenAICompletionsTeacher` against Together / Fireworks / DeepInfra,
which support echo+logprobs in a single call per example.
"""

from __future__ import annotations

import time
from typing import List

import requests

from ctd.teachers.base import PositionLogprobs, TeacherBackend


class OllamaTeacher(TeacherBackend):
    def __init__(
        self,
        base_url: str,
        model_id: str,
        tokenizer,
        top_k: int = 20,
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        super().__init__(model_id=model_id, tokenizer=tokenizer, top_k=top_k)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def _post(self, prompt: str) -> dict:
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_id,
                        "prompt": prompt,
                        "stream": False,
                        "raw": True,  # skip Ollama's chat template wrapper
                        "options": {
                            "num_predict": 1,
                            "temperature": 0,
                        },
                        "logprobs": True,
                        "top_logprobs": min(self.top_k, 20),
                    },
                    timeout=self.timeout,
                )
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError):
                if attempt == self.max_retries:
                    raise
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError("unreachable")

    def get_token_topk(
        self, text: str
    ) -> tuple[List[int], List[PositionLogprobs]]:
        teacher_ids = self.tokenizer.encode(text, add_special_tokens=False)
        results: List[PositionLogprobs] = []

        for k in range(len(teacher_ids)):
            prefix_str = (
                self.tokenizer.decode(teacher_ids[:k]) if k > 0 else ""
            )
            data = self._post(prefix_str)
            lp = data.get("logprobs") or []
            if not lp:
                results.append(
                    PositionLogprobs(topk_ids=[teacher_ids[k]], topk_logprobs=[0.0])
                )
                continue
            top = lp[0].get("top_logprobs") or []
            ids: List[int] = []
            logprobs: List[float] = []
            for entry in top:
                tid = self._str_to_id(entry["token"])
                if tid is None:
                    continue
                ids.append(tid)
                logprobs.append(float(entry["logprob"]))
            if not ids:
                ids = [teacher_ids[k]]
                logprobs = [0.0]
            results.append(PositionLogprobs(topk_ids=ids, topk_logprobs=logprobs))

        return teacher_ids, results

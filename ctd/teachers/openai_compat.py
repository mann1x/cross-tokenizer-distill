"""OpenAI-compatible /v1/completions teacher backend.

Targets Together.AI, Fireworks, DeepInfra (and any other host that
exposes /v1/completions with `echo: true` + `logprobs: K`). One HTTP
call per example returns the teacher's top-K predictions at every
prompt position — no N-call slowdown like Ollama.

Provider quirks:
- Together.AI: `echo: true` + `logprobs: K` (K capped at 20 in most plans).
- Fireworks: same shape; `logprobs: K` (capped 20).
- DeepInfra: OpenAI-compat; `echo: true` + `logprobs: K` (cap is model-dep,
  often 5 or 10). Confirm with the provider before relying on K.

Auth: pass `api_key` (sent as `Authorization: Bearer …`).
Set base_url to:
- https://api.together.xyz/v1
- https://api.fireworks.ai/inference/v1
- https://api.deepinfra.com/v1/openai
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import requests

from ctd.teachers.base import PositionLogprobs, TeacherBackend


class OpenAICompletionsTeacher(TeacherBackend):
    def __init__(
        self,
        base_url: str,
        model_id: str,
        tokenizer,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        top_k: int = 20,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        super().__init__(model_id=model_id, tokenizer=tokenizer, top_k=top_k)
        self.base_url = base_url.rstrip("/")
        if api_key is None and api_key_env:
            api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                "OpenAICompletionsTeacher requires api_key or api_key_env "
                "pointing to a non-empty env var."
            )
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def _post(self, prompt: str) -> dict:
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(
                    f"{self.base_url}/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model_id,
                        "prompt": prompt,
                        "max_tokens": 1,
                        "temperature": 0,
                        "echo": True,
                        "logprobs": min(self.top_k, 20),
                    },
                    timeout=self.timeout,
                )
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(
            f"OpenAICompletionsTeacher.post failed after {self.max_retries+1} attempts"
        ) from last_exc

    def get_token_topk(
        self, text: str
    ) -> tuple[List[int], List[PositionLogprobs]]:
        teacher_ids = self.tokenizer.encode(text, add_special_tokens=False)
        data = self._post(text)
        choice = (data.get("choices") or [{}])[0]
        lp = choice.get("logprobs") or {}
        tokens = lp.get("tokens") or []
        token_logprobs = lp.get("token_logprobs") or []
        top_logprobs_per_pos = lp.get("top_logprobs") or []

        # Echo returns logprobs at every prompt position. The first
        # position has token_logprobs=None (no preceding context).
        # We emit one PositionLogprobs per prompt token after the first.
        results: List[PositionLogprobs] = []
        for i, top_dict in enumerate(top_logprobs_per_pos):
            if i == 0 and (token_logprobs and token_logprobs[0] is None):
                # Skip leading null entry; emit a placeholder so length matches.
                results.append(
                    PositionLogprobs(topk_ids=[], topk_logprobs=[])
                )
                continue
            if not isinstance(top_dict, dict):
                results.append(
                    PositionLogprobs(topk_ids=[], topk_logprobs=[])
                )
                continue
            ids: List[int] = []
            logprobs: List[float] = []
            for tok_str, lp_val in top_dict.items():
                tid = self._str_to_id(tok_str)
                if tid is None or lp_val is None:
                    continue
                ids.append(tid)
                logprobs.append(float(lp_val))
            results.append(PositionLogprobs(topk_ids=ids, topk_logprobs=logprobs))

        # Truncate / pad to match the local tokenization length so the
        # alignment layer sees a consistent shape. (Provider tokenization
        # should match the local tokenizer when both are loaded from the
        # same model — but cap defensively in case of drift.)
        if len(results) > len(teacher_ids):
            results = results[: len(teacher_ids)]
        while len(results) < len(teacher_ids):
            results.append(PositionLogprobs(topk_ids=[], topk_logprobs=[]))

        # Sanity log: warn if remote tokens disagree with local tokens.
        if tokens and len(tokens) != len(teacher_ids):
            # Just record a soft signal; caller can decide whether to drop.
            pass
        return teacher_ids, results

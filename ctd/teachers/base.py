"""Base class for remote teacher backends used in CTD precompute.

A teacher backend turns a text into a per-token list of top-K
(teacher_vocab_id, logprob) predictions. Higher layers (precompute_remote.py)
take care of alignment to student tokenization and optional projection
into student vocab via VocabMapper.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PositionLogprobs:
    """Top-K teacher predictions at one teacher-token position."""

    topk_ids: List[int]
    topk_logprobs: List[float]


class TeacherBackend(ABC):
    """Provides per-position top-K logprobs over teacher vocab for a given text.

    The returned list is aligned to the teacher's tokenization of the input
    text: result[k] holds the top-K predictions for teacher_token[k] given
    teacher_token[0..k-1] (i.e. forward-decode logits at position k).
    """

    def __init__(self, model_id: str, tokenizer, top_k: int = 20):
        self.model_id = model_id
        self.tokenizer = tokenizer  # local HF tokenizer for the same model
        self.top_k = top_k
        self._vocab_str_to_id = tokenizer.get_vocab()

    @abstractmethod
    def get_token_topk(
        self, text: str
    ) -> tuple[List[int], List[PositionLogprobs]]:
        """Return (teacher_token_ids, per_position_topk).

        teacher_token_ids has length L_t (tokenization of `text`).
        per_position_topk has length L_t (or L_t-1 if the backend can
        only score positions 1..L_t-1; the precompute layer handles
        either).
        """

    def _str_to_id(self, token_str: str) -> Optional[int]:
        """Resolve an API-returned token string back to a teacher vocab ID.

        Tries the vocab dict first (constant time, hits for every byte-level
        BPE token). Falls back to encoding as-is and accepting if it produces
        exactly one token.
        """
        tid = self._vocab_str_to_id.get(token_str)
        if tid is not None:
            return tid
        try:
            ids = self.tokenizer.encode(token_str, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        except Exception:
            pass
        return None

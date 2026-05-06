"""CTD utilities — shared helpers used across trainers and generation paths."""
from __future__ import annotations
from typing import Iterable


def make_teacher_token_blacklist(
    tokenizer,
    names_csv: str = "",
    ids_csv: str = "",
) -> list[int]:
    """Resolve a set of teacher-vocab token IDs to mask.

    Used by:
      - On-policy KL trainers (06_train_grpo_kl_distill.py, etc.) — set
        the IDs to -inf in teacher logits before softmax so they're
        excluded from KL.
      - Teacher generation paths (gen_teacher_completions.py,
        eval_teacher_chat.py) — pass as ``bad_words_ids=[[id], ...]``
        to ``model.generate()`` so the teacher never emits them.

    Args:
        tokenizer: HF tokenizer of the TEACHER (the vocab we mask in).
        names_csv: Comma-separated token strings to mask. Each is
            (a) encoded via ``tokenizer.encode(s, add_special_tokens=False)``,
            and (b) looked up in ``tokenizer.get_vocab()`` for
            single-piece special tokens like ``<think>`` that some
            tokenizers store directly. Both paths' IDs are unioned.
        ids_csv: Comma-separated raw token IDs (additive to ``names_csv``).

    Returns:
        Sorted list of unique teacher token IDs.

    Examples:
        >>> # Reasoning teachers (R1-distill family, DeepCoder, etc.)
        >>> make_teacher_token_blacklist(tok, names_csv="<think>,</think>")
        [151648, 151649]

        >>> # Direct ID list (when you already know them)
        >>> make_teacher_token_blacklist(tok, ids_csv="151648,151649")
        [151648, 151649]
    """
    ids: set[int] = set()
    if names_csv:
        vocab = tokenizer.get_vocab()
        for tok_str in [s for s in names_csv.split(",") if s]:
            # Path 1: direct vocab lookup — only path that's safe for arbitrary
            # multi-char strings. Special tokens like <think> in DeepSeek-R1 family
            # live as single dedicated IDs; this picks them up.
            if tok_str in vocab:
                ids.add(vocab[tok_str])
                continue
            # Path 2: encode and accept ONLY if it tokenizes to a single piece.
            # Multi-piece encodings would over-ban (e.g. encoding "<think>" on
            # Qwen-Coder splits into [<, think, >] — banning '>' kills all code).
            enc = tokenizer.encode(tok_str, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
            else:
                import warnings
                warnings.warn(
                    f"make_teacher_token_blacklist: '{tok_str}' is not a single "
                    f"token in this tokenizer (encodes to {len(enc)} pieces: {enc}). "
                    f"Skipping — banning sub-pieces would over-restrict generation. "
                    f"If you really want it, pass --mask-teacher-token-ids with the "
                    f"resolved IDs you want to ban.",
                    stacklevel=2,
                )
    if ids_csv:
        ids.update(int(x) for x in ids_csv.split(",") if x)
    return sorted(ids)


def bad_words_ids_for_generate(blacklist: Iterable[int]) -> list[list[int]] | None:
    """Convert a flat ID list to the nested format ``model.generate(bad_words_ids=...)`` expects."""
    bl = list(blacklist)
    if not bl:
        return None
    return [[i] for i in bl]

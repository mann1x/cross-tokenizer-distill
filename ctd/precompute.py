"""Standalone teacher logit cache builder.

Produces a top-K teacher cache aligned to student token positions,
optionally projected to student vocab at write time so that the
training pipeline doesn't need any CTD-specific code (drop-in
replacement for same-vocab caches).

Output cache schema (torch.save):

    {
        "values":  Tensor [N_tokens, top_K]   # log-probabilities
        "indices": Tensor [N_tokens, top_K]   # vocab indices (student or teacher, depending on project_at_write_time)
        "mask":    Tensor [N_tokens]          # bool — True where the alignment is valid
        "block_offsets": Tensor [N_blocks+1]  # block start positions (for sequence-level loading)
        "meta": {
            "teacher_model": str,
            "teacher_tokenizer": str,
            "student_tokenizer": str,
            "alignment": str,
            "suffix_reencode": bool,
            "projection": str | None,
            "project_at_write_time": bool,
            "top_k": int,
            "n_total_tokens": int,
            "n_aligned_tokens": int,
            "n_suffix_reencode": int,
            "ctd_version": str,
            "seed": int,
        },
    }

This shape is identical to the existing same-vocab top-K caches used
by Mythic-RDT (`teacher_cache/dscoder_*.pt`), so the trainer needs no
changes when switching teachers.
"""

from __future__ import annotations

from typing import Iterable, Literal, Optional

Tokenizer = "transformers.PreTrainedTokenizerBase"  # noqa: F821
PreTrainedModel = "transformers.PreTrainedModel"  # noqa: F821


def precompute_aligned_cache(
    teacher_model: "PreTrainedModel",
    teacher_tokenizer: "Tokenizer",
    student_tokenizer: "Tokenizer",
    text_corpus: Iterable[str],
    output_path: str,
    top_k: int = 32,
    alignment: Literal["byte_anchor", "student_offset"] = "student_offset",
    suffix_reencode: bool = True,
    projection: Optional["VocabMapper"] = None,  # noqa: F821 (fwd ref)
    project_at_write_time: bool = True,
    block_size: int = 2048,
    device: str = "cuda",
    seed: int = 0,
    log_every: int = 100,
) -> dict:
    """Build a top-K teacher cache aligned to the student tokenization.

    Args:
        teacher_model: HF model (already on device, in eval mode).
        teacher_tokenizer: teacher's tokenizer.
        student_tokenizer: student's tokenizer.
        text_corpus: iterable of text strings (e.g. lines from a JSONL).
        output_path: path to write the .pt cache.
        top_k: number of top logits to retain per position.
        alignment: 'byte_anchor' or 'student_offset'.
        suffix_reencode: enable smart KV-cache reuse for non-aligned
            positions (only effective with 'student_offset').
        projection: VocabMapper instance. Required when
            project_at_write_time=True.
        project_at_write_time: if True, apply mapper.project_topk()
            during precompute so the cache file uses student vocab
            indices. Recommended.
        block_size: number of student tokens per cache block. The
            cache is laid out as concatenated blocks for efficient
            random access during training.
        device: 'cuda' typically.
        seed: corpus shuffling seed (recorded in meta).
        log_every: emit progress every N examples.

    Returns:
        Dict with summary stats (n_examples, n_tokens, suffix_reencode_rate, etc.).
    """
    raise NotImplementedError(
        "precompute_aligned_cache not yet implemented — depends on "
        "alignment.build_alignment + mapper.project_topk."
    )

"""Position alignment — match student token positions to teacher token positions.

Given the same input text tokenized by two different tokenizers, this
module finds, for every student position j, the teacher state that
predicts the byte immediately after the student's prefix s[0..j].

Two strategies:

  - byte_anchor: only return alignments where teacher and student
    tokenization end at the same byte offset. Skip non-aligned
    positions. Cheap, lossy.

  - student_offset (default): for every student position, materialise
    teacher's prediction at that exact byte offset. For aligned
    positions reuse teacher's natural logit; for non-aligned, run
    teacher on a small suffix continuation with KV cache reuse.

See docs/DESIGN.md for the math and edge-case handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

Tokenizer = "transformers.PreTrainedTokenizerBase"  # noqa: F821

AlignmentMode = Literal["byte_anchor", "student_offset"]


@dataclass
class AlignmentTable:
    """Per-example alignment from student positions to teacher prediction sites.

    Attributes:
        student_pos: tensor [N_student] of student token positions
            (0..N_student-1) that have a teacher target.
        teacher_pos: tensor [N_student] — the teacher position whose
            logit to use. Set to -1 for positions requiring suffix
            re-encode.
        suffix_token_ids: list[Optional[list[int]]] — for each
            student position needing suffix re-encode, the teacher
            tokens to append. None for clean alignments.
        kv_anchor_pos: tensor [N_student] — teacher position to load
            KV cache from (used only for suffix re-encode entries).
        mask: bool tensor [N_student] — True where the alignment is
            valid (False for byte_anchor mode positions that were
            dropped).
    """

    student_pos: torch.Tensor
    teacher_pos: torch.Tensor
    suffix_token_ids: list[Optional[list[int]]]
    kv_anchor_pos: torch.Tensor
    mask: torch.Tensor


def compute_byte_offsets(token_ids: list[int], tokenizer: "Tokenizer") -> list[int]:
    """Return cumulative byte offsets for each token in the sequence.

    offsets[i] = total bytes spanned by tokens[0..i] (inclusive of i).
    offsets[0] starts AFTER the first token (so offsets[0] = len(decode([tokens[0]]))).

    Implementation note: tokenizer.decode([id]) is correct only when
    the tokenizer doesn't apply leading-space normalisation. For
    tokenizers that prepend whitespace (Llama, Qwen, DeepSeek), we
    decode incremental prefixes instead and diff lengths. Slower but
    correct.
    """
    offsets = []
    # Decode incremental prefixes — handles whitespace normalization correctly.
    for i in range(1, len(token_ids) + 1):
        s = tokenizer.decode(token_ids[:i], skip_special_tokens=False)
        offsets.append(len(s.encode("utf-8")))
    return offsets


def build_alignment(
    text: str,
    teacher_token_ids: list[int],
    student_token_ids: list[int],
    teacher_tokenizer: "Tokenizer",
    student_tokenizer: "Tokenizer",
    mode: AlignmentMode = "student_offset",
    suffix_reencode: bool = True,
) -> AlignmentTable:
    """Build the per-position alignment table for one example.

    Args:
        text: original text (pre-tokenization).
        teacher_token_ids: result of teacher_tokenizer.encode(text).
        student_token_ids: result of student_tokenizer.encode(text).
        teacher_tokenizer, student_tokenizer: the tokenizer objects.
        mode: alignment strategy. See module docstring.
        suffix_reencode: when mode='student_offset', whether to
            generate suffix continuations for non-aligned positions
            (True) or skip them (False, equivalent to byte_anchor).

    Returns:
        AlignmentTable with one entry per student position.
    """
    raise NotImplementedError(
        "build_alignment not yet implemented — see docs/DESIGN.md section "
        "'Position alignment' for the algorithm."
    )

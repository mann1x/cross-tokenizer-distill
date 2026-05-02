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

from dataclasses import dataclass, field
from typing import Literal, Optional

Tokenizer = "transformers.PreTrainedTokenizerBase"  # noqa: F821

AlignmentMode = Literal["byte_anchor", "student_offset"]


@dataclass
class AlignmentEntry:
    """Per-student-position alignment record.

    For each student position j (predicting the token AFTER the j-th
    student token), this tells the precompute loop which teacher
    logit to use as the distillation target.

    If suffix_token_ids is None and teacher_pos >= 0:
        Use teacher's natural logit at teacher_pos.

    If suffix_token_ids is not None:
        Reuse teacher's KV cache at kv_anchor_pos, append
        suffix_token_ids, take the FINAL logit. (kv_anchor_pos may be
        -1 meaning "from scratch / start of sequence".)

    If valid is False:
        Skip this position entirely.
    """

    student_pos: int
    valid: bool = True
    teacher_pos: int = -1
    suffix_token_ids: Optional[list[int]] = None
    kv_anchor_pos: int = -1


@dataclass
class AlignmentTable:
    """Per-example alignment table."""

    entries: list[AlignmentEntry]
    student_offsets: list[int] = field(default_factory=list)
    teacher_offsets: list[int] = field(default_factory=list)
    n_aligned: int = 0
    n_suffix: int = 0
    n_dropped: int = 0

    def __len__(self) -> int:
        return len(self.entries)


def compute_byte_offsets(token_ids: list[int], tokenizer: "Tokenizer") -> list[int]:
    """Return cumulative byte offsets for each token in the sequence.

    offsets[i] = total bytes spanned by tokens[0..i] (inclusive of i).

    We decode incremental prefixes to handle whitespace normalization
    correctly (Llama/Qwen/DeepSeek tokenizers normalise leading spaces
    in ways that break a naive sum-of-decoded-token-bytes).
    """
    offsets = []
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
        teacher_tokenizer, student_tokenizer: tokenizer objects.
        mode: alignment strategy.
        suffix_reencode: when mode='student_offset', whether to plan
            suffix continuations for non-aligned positions (True) or
            skip them (False, equivalent to byte_anchor).

    Returns:
        AlignmentTable.
    """
    text_bytes = text.encode("utf-8")
    teacher_offsets = compute_byte_offsets(teacher_token_ids, teacher_tokenizer)
    student_offsets = compute_byte_offsets(student_token_ids, student_tokenizer)

    teacher_offset_to_pos: dict[int, int] = {b: i for i, b in enumerate(teacher_offsets)}

    entries: list[AlignmentEntry] = []
    n_aligned = 0
    n_suffix = 0
    n_dropped = 0

    for j, s_off in enumerate(student_offsets):
        if s_off in teacher_offset_to_pos:
            entries.append(
                AlignmentEntry(
                    student_pos=j,
                    valid=True,
                    teacher_pos=teacher_offset_to_pos[s_off],
                )
            )
            n_aligned += 1
            continue

        if mode == "byte_anchor" or (mode == "student_offset" and not suffix_reencode):
            entries.append(AlignmentEntry(student_pos=j, valid=False))
            n_dropped += 1
            continue

        # student_offset + suffix_reencode: plan a suffix continuation.
        # Find the largest teacher position k* with teacher_offsets[k*] <= s_off.
        k_star = -1
        for k, t_off in enumerate(teacher_offsets):
            if t_off <= s_off:
                k_star = k
            else:
                break

        anchor_byte = teacher_offsets[k_star] if k_star >= 0 else 0
        suffix_bytes = text_bytes[anchor_byte:s_off]
        if not suffix_bytes:
            entries.append(AlignmentEntry(student_pos=j, valid=False))
            n_dropped += 1
            continue

        try:
            suffix_str = suffix_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            entries.append(AlignmentEntry(student_pos=j, valid=False))
            n_dropped += 1
            continue

        suffix_ids = teacher_tokenizer.encode(suffix_str, add_special_tokens=False)
        if not suffix_ids:
            entries.append(AlignmentEntry(student_pos=j, valid=False))
            n_dropped += 1
            continue

        entries.append(
            AlignmentEntry(
                student_pos=j,
                valid=True,
                teacher_pos=-1,
                suffix_token_ids=suffix_ids,
                kv_anchor_pos=k_star,
            )
        )
        n_suffix += 1

    return AlignmentTable(
        entries=entries,
        student_offsets=student_offsets,
        teacher_offsets=teacher_offsets,
        n_aligned=n_aligned,
        n_suffix=n_suffix,
        n_dropped=n_dropped,
    )

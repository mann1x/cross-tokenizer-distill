"""VocabMapper — build a sparse projection from teacher vocab to student vocab.

The mapper is the foundational primitive of CTD. Given two tokenizers,
it constructs a sparse matrix M of shape (V_student, V_teacher) such
that for any teacher distribution p_T over its vocab,

    p_S_projected = M @ p_T

is a distribution-like quantity over the student vocab.

The mapper also reports coverage statistics (what fraction of teacher's
average mass survives projection) so callers can decide go / no-go
before paying for a full precompute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

# Type alias — accept either a HF AutoTokenizer or a tokenizers.Tokenizer.
# We only need .decode() / .encode() on individual ids and strings.
Tokenizer = "transformers.PreTrainedTokenizerBase"  # noqa: F821 (forward ref)


MultiTokenStrategy = Literal["strict", "distribute", "first_token"]


@dataclass
class CoverageReport:
    """Diagnostic report from VocabMapper construction.

    Returned by VocabMapper.coverage_report().
    """

    teacher_vocab_size: int
    student_vocab_size: int
    single_token_count: int
    multi_token_count: int
    dropped_count: int
    avg_multi_token_fragments: float
    bytewise_roundtrip_ok: bool
    strategy: MultiTokenStrategy

    @property
    def single_token_rate(self) -> float:
        return self.single_token_count / max(self.teacher_vocab_size, 1)

    @property
    def multi_token_rate(self) -> float:
        return self.multi_token_count / max(self.teacher_vocab_size, 1)

    @property
    def dropped_rate(self) -> float:
        return self.dropped_count / max(self.teacher_vocab_size, 1)

    def __str__(self) -> str:
        return (
            f"=== Coverage report ===\n"
            f"  teacher vocab: {self.teacher_vocab_size}, "
            f"student vocab: {self.student_vocab_size}\n"
            f"  byte-roundtrip ok: {self.bytewise_roundtrip_ok}\n"
            f"  strategy: {self.strategy}\n"
            f"  single-token map rate: {self.single_token_rate:.1%}\n"
            f"  multi-token rate:      {self.multi_token_rate:.1%} "
            f"(avg fragments: {self.avg_multi_token_fragments:.2f})\n"
            f"  dropped rate:          {self.dropped_rate:.1%}\n"
        )


class VocabMapper:
    """Sparse projection from teacher vocab to student vocab.

    Construct via VocabMapper.from_tokenizers(...). The resulting
    object exposes:

      - matrix: torch.Tensor (sparse COO) of shape (V_student, V_teacher)
      - coverage_report(): CoverageReport
      - project_topk(values, indices): apply projection to top-K
        teacher logits, return top-K' student-vocab values + indices.
    """

    def __init__(
        self,
        matrix: torch.Tensor,
        teacher_vocab_size: int,
        student_vocab_size: int,
        report: CoverageReport,
        strategy: MultiTokenStrategy,
    ):
        self.matrix = matrix
        self.teacher_vocab_size = teacher_vocab_size
        self.student_vocab_size = student_vocab_size
        self._report = report
        self.strategy = strategy

    @classmethod
    def from_tokenizers(
        cls,
        teacher_tokenizer: "Tokenizer",
        student_tokenizer: "Tokenizer",
        multi_token: MultiTokenStrategy = "distribute",
        verify_roundtrip_samples: int = 200,
    ) -> "VocabMapper":
        """Build a VocabMapper from a pair of HF tokenizers.

        Args:
            teacher_tokenizer: source tokenizer
            student_tokenizer: target tokenizer
            multi_token: how to handle teacher tokens that decode to
                multiple student tokens. "strict" drops them,
                "distribute" splits 1/n across fragments, "first_token"
                puts all mass on the first fragment.
            verify_roundtrip_samples: number of teacher token IDs to
                sample-test for byte-exact decode/encode round-trip in
                the OPPOSITE direction (student → student via teacher
                vocab is a different question; we skip that).

        Returns:
            VocabMapper instance.

        Raises:
            ValueError: if tokenizer pair is incompatible (cannot
                normalise to a common byte representation).
        """
        # Implementation in next iteration. This skeleton documents the API.
        raise NotImplementedError(
            "VocabMapper.from_tokenizers not yet implemented — see docs/DESIGN.md "
            "section 'Vocabulary projection' for the algorithm."
        )

    def coverage_report(self) -> CoverageReport:
        """Return diagnostic stats on this mapping."""
        return self._report

    def project_topk(
        self,
        teacher_topk_values: torch.Tensor,  # [..., K]
        teacher_topk_indices: torch.Tensor,  # [..., K]
        out_topk: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project teacher's top-K logits onto student vocab.

        Args:
            teacher_topk_values: log-probabilities or raw logits at
                teacher's top-K positions. Shape [..., K].
            teacher_topk_indices: corresponding teacher vocab indices.
                Shape [..., K].
            out_topk: number of student-vocab top entries to retain.

        Returns:
            (student_topk_values, student_topk_indices) — both shape
            [..., out_topk]. Indices are in student vocab.
        """
        raise NotImplementedError(
            "project_topk not yet implemented — depends on mapper.matrix."
        )

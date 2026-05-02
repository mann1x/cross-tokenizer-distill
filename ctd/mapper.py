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
from tqdm import tqdm

Tokenizer = "transformers.PreTrainedTokenizerBase"  # noqa: F821 (forward ref)


MultiTokenStrategy = Literal["strict", "distribute", "first_token"]


@dataclass
class CoverageReport:
    """Diagnostic report from VocabMapper construction."""

    teacher_vocab_size: int
    student_vocab_size: int
    single_token_count: int
    multi_token_count: int
    dropped_count: int
    avg_multi_token_fragments: float
    bytewise_roundtrip_ok: bool
    roundtrip_failures: int
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

    @property
    def coverage(self) -> float:
        """Fraction of teacher tokens that retain ANY mass after projection."""
        if self.strategy == "strict":
            return self.single_token_rate
        return self.single_token_rate + self.multi_token_rate

    def __str__(self) -> str:
        return (
            f"=== Coverage report ===\n"
            f"  teacher vocab: {self.teacher_vocab_size}, "
            f"student vocab: {self.student_vocab_size}\n"
            f"  byte-roundtrip ok: {self.bytewise_roundtrip_ok} "
            f"({self.roundtrip_failures} failures)\n"
            f"  strategy: {self.strategy}\n"
            f"  single-token map rate: {self.single_token_rate:.1%}\n"
            f"  multi-token rate:      {self.multi_token_rate:.1%} "
            f"(avg fragments: {self.avg_multi_token_fragments:.2f})\n"
            f"  dropped rate:          {self.dropped_rate:.1%}\n"
            f"  total coverage:        {self.coverage:.1%}\n"
        )


class VocabMapper:
    """Sparse projection from teacher vocab to student vocab.

    Construct via VocabMapper.from_tokenizers(...). The resulting object
    exposes:

      - matrix: torch.sparse_coo_tensor of shape (V_student, V_teacher)
      - coverage_report(): CoverageReport
      - project_topk(values, indices): apply projection to top-K teacher
        logits, return projected student-vocab top-K' tensor.
    """

    def __init__(
        self,
        matrix: torch.Tensor,
        teacher_vocab_size: int,
        student_vocab_size: int,
        report: CoverageReport,
        strategy: MultiTokenStrategy,
    ):
        self.matrix = matrix  # sparse COO [V_student, V_teacher]
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
        progress: bool = True,
    ) -> "VocabMapper":
        """Build a VocabMapper from a pair of HF tokenizers."""
        v_t = teacher_tokenizer.vocab_size
        v_s = student_tokenizer.vocab_size

        # Sample-verify byte round-trip.
        rt_failures = 0
        sample_step = max(1, v_t // verify_roundtrip_samples)
        for t_id in range(0, v_t, sample_step):
            try:
                s = teacher_tokenizer.decode([t_id], skip_special_tokens=False)
                if not s:
                    continue
                ids = student_tokenizer.encode(s, add_special_tokens=False)
                if not ids:
                    rt_failures += 1
                    continue
                back = student_tokenizer.decode(ids, skip_special_tokens=False)
                if back.encode("utf-8") != s.encode("utf-8"):
                    rt_failures += 1
            except Exception:
                rt_failures += 1
        roundtrip_ok = rt_failures < (verify_roundtrip_samples * 0.05)

        # Build sparse triplets.
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        single_token_count = 0
        multi_token_count = 0
        dropped_count = 0
        total_fragments = 0

        iterator = range(v_t)
        if progress:
            iterator = tqdm(iterator, desc=f"VocabMapper({multi_token})", total=v_t)

        for t_id in iterator:
            try:
                s = teacher_tokenizer.decode([t_id], skip_special_tokens=False)
            except Exception:
                dropped_count += 1
                continue
            if not s:
                dropped_count += 1
                continue
            try:
                s_ids = student_tokenizer.encode(s, add_special_tokens=False)
            except Exception:
                dropped_count += 1
                continue
            if not s_ids:
                dropped_count += 1
                continue

            if len(s_ids) == 1:
                rows.append(s_ids[0])
                cols.append(t_id)
                vals.append(1.0)
                single_token_count += 1
            else:
                multi_token_count += 1
                total_fragments += len(s_ids)
                if multi_token == "strict":
                    pass
                elif multi_token == "first_token":
                    rows.append(s_ids[0])
                    cols.append(t_id)
                    vals.append(1.0)
                elif multi_token == "distribute":
                    weight = 1.0 / len(s_ids)
                    for s_id in s_ids:
                        rows.append(s_id)
                        cols.append(t_id)
                        vals.append(weight)
                else:
                    raise ValueError(f"Unknown multi_token strategy: {multi_token}")

        if rows:
            indices = torch.tensor([rows, cols], dtype=torch.long)
            values = torch.tensor(vals, dtype=torch.float32)
            matrix = torch.sparse_coo_tensor(indices, values, size=(v_s, v_t)).coalesce()
        else:
            matrix = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
                size=(v_s, v_t),
            ).coalesce()

        avg_frags = (total_fragments / multi_token_count) if multi_token_count else 0.0

        report = CoverageReport(
            teacher_vocab_size=v_t,
            student_vocab_size=v_s,
            single_token_count=single_token_count,
            multi_token_count=multi_token_count,
            dropped_count=dropped_count,
            avg_multi_token_fragments=avg_frags,
            bytewise_roundtrip_ok=roundtrip_ok,
            roundtrip_failures=rt_failures,
            strategy=multi_token,
        )

        return cls(
            matrix=matrix,
            teacher_vocab_size=v_t,
            student_vocab_size=v_s,
            report=report,
            strategy=multi_token,
        )

    def coverage_report(self) -> CoverageReport:
        return self._report

    def project_distribution(self, teacher_dist: torch.Tensor) -> torch.Tensor:
        """Project a full teacher distribution onto student vocab.

        Args:
            teacher_dist: [..., V_teacher] tensor.

        Returns:
            [..., V_student] tensor (dense). Mass may sum to <1 if the
            projection drops some teacher tokens.
        """
        leading = teacher_dist.shape[:-1]
        flat = teacher_dist.reshape(-1, self.teacher_vocab_size)
        M = self.matrix.to(flat.device).to(flat.dtype)
        out = torch.sparse.mm(M, flat.T).T  # [B, V_s]
        return out.reshape(*leading, self.student_vocab_size)

    def project_topk(
        self,
        teacher_topk_values: torch.Tensor,
        teacher_topk_indices: torch.Tensor,
        out_topk: int = 32,
        already_softmaxed: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project teacher's top-K logits onto student vocab.

        Returns log-probabilities + indices in student vocab. Output
        log-probs are renormalised over the retained student top-K' so
        the result is a well-defined log-distribution.
        """
        leading = teacher_topk_values.shape[:-1]
        K = teacher_topk_values.shape[-1]
        device = teacher_topk_values.device

        if already_softmaxed:
            probs_K = teacher_topk_values
        else:
            probs_K = teacher_topk_values.softmax(dim=-1)

        flat_probs = probs_K.reshape(-1, K)
        flat_idx = teacher_topk_indices.reshape(-1, K)
        B = flat_probs.shape[0]

        # Build a dense teacher distribution from top-K (zero elsewhere).
        dense_T = torch.zeros(B, self.teacher_vocab_size, device=device, dtype=probs_K.dtype)
        dense_T.scatter_add_(1, flat_idx, flat_probs)

        # Project to student vocab.
        M = self.matrix.to(device).to(dense_T.dtype)
        dense_S = torch.sparse.mm(M, dense_T.T).T  # [B, V_s]

        # Take top-K' on the student side.
        topk_vals, topk_ids = dense_S.topk(out_topk, dim=-1)

        # Renormalise + log.
        topk_vals = topk_vals.clamp(min=1e-12)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
        log_topk = topk_vals.log()

        return (
            log_topk.reshape(*leading, out_topk),
            topk_ids.reshape(*leading, out_topk),
        )

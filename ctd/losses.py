"""Distillation losses operating on projected teacher distributions.

Each loss takes a student log-probability tensor (or raw logits) and a
teacher distribution that's already been projected to student vocab
(or top-K thereof). The loss handles the masking for non-aligned
positions and renormalisation for partial-mass distributions.

The cache shape we operate on (matching ctd.precompute output and the
existing same-vocab caches in Mythic-RDT):

    teacher_topk_log_values: [B, L, K]   log-probabilities (renormalised)
    teacher_topk_indices:    [B, L, K]   indices in student vocab
    alignment_mask:          [B, L]      bool — True where the position is valid

The loss is summed over valid positions (mask=True) and divided by the
number of valid positions. Teacher mass at non-top-K student tokens is
treated as zero (sparse top-K KL).
"""

from __future__ import annotations

from typing import Literal, Optional

import torch

LossKind = Literal["kl", "jsd", "mse", "uld_sorted_kl"]


class CTDLoss:
    """Cross-tokenizer distillation loss.

    Args:
        kind: 'kl' (default) | 'jsd' | 'mse' | 'uld_sorted_kl'.
        temperature: applied to STUDENT logits before computing log-probs.
            Teacher's distribution is assumed to already have been
            temperature-scaled at precompute time (or to be temperature 1).
        mapper: optional VocabMapper for project-at-train-time mode. If
            given, the cache is assumed to be teacher-vocab indices and
            projection happens here instead.
        eps: numerical floor.
    """

    def __init__(
        self,
        kind: LossKind = "kl",
        temperature: float = 1.0,
        mapper: Optional["VocabMapper"] = None,  # noqa: F821 (fwd ref)
        eps: float = 1e-12,
    ):
        if kind not in ("kl", "jsd", "mse", "uld_sorted_kl"):
            raise ValueError(f"Unknown loss kind: {kind}")
        self.kind = kind
        self.temperature = temperature
        self.mapper = mapper
        self.eps = eps

    def __call__(
        self,
        student_logits: torch.Tensor,            # [B, L, V_student]
        teacher_topk_log_values: torch.Tensor,    # [B, L, K] log-probs
        teacher_topk_indices: torch.Tensor,       # [B, L, K] (student-vocab if mapper is None)
        alignment_mask: Optional[torch.Tensor] = None,  # [B, L] bool
    ) -> torch.Tensor:
        """Compute distillation loss summed over valid positions.

        Returns scalar loss tensor.
        """
        # Optional project-at-train-time path.
        if self.mapper is not None:
            teacher_topk_log_values, teacher_topk_indices = self.mapper.project_topk(
                teacher_topk_log_values,
                teacher_topk_indices,
                out_topk=teacher_topk_log_values.shape[-1],
                already_softmaxed=False,  # treat as logits
            )

        if self.kind == "uld_sorted_kl":
            return self._uld_sorted_kl(
                student_logits, teacher_topk_log_values, alignment_mask
            )

        # Standard top-K projected losses (kl / jsd / mse).
        # Hinton-style soft targets: BOTH student and teacher distributions
        # are softened at temperature T, and the resulting KL is multiplied
        # by T² to keep gradient magnitude comparable to the unsoftened CE
        # term (so changing T shifts the loss surface, not its scale).
        T = self.temperature
        student_log_probs = (student_logits / T).log_softmax(dim=-1)
        # student_log_probs: [B, L, V_S]; gather at indices [B, L, K] → [B, L, K]
        student_topk_log = student_log_probs.gather(-1, teacher_topk_indices)

        # Teacher cache stores log-softmax over top-K at T=1. Treating those
        # log-probs as proxy logits, re-soften over the top-K support at T:
        #   softmax(z_t / T) ∝ softmax(log P_t / T)   (within top-K)
        teacher_topk_log_T = (teacher_topk_log_values / T).log_softmax(dim=-1)
        teacher_topk_probs_T = teacher_topk_log_T.exp()  # [B, L, K]

        if self.kind == "kl":
            # KL(P_T || P_S) on the top-K support.
            per_pos = (
                teacher_topk_probs_T * (teacher_topk_log_T - student_topk_log)
            ).sum(dim=-1)  # [B, L]

        elif self.kind == "jsd":
            # JSD on the top-K support: 0.5 * (KL(P || M) + KL(Q || M))
            student_topk_probs = student_topk_log.exp()
            mid = 0.5 * (teacher_topk_probs_T + student_topk_probs).clamp(min=self.eps)
            mid_log = mid.log()
            kl_pm = (teacher_topk_probs_T * (teacher_topk_log_T - mid_log)).sum(-1)
            kl_qm = (student_topk_probs * (student_topk_log - mid_log)).sum(-1)
            per_pos = 0.5 * (kl_pm + kl_qm)

        elif self.kind == "mse":
            # MSE on the (re-softened) top-K probabilities.
            student_topk_probs = student_topk_log.exp()
            per_pos = ((teacher_topk_probs_T - student_topk_probs) ** 2).sum(-1)

        else:
            raise AssertionError(f"unreachable: {self.kind}")

        # T² rescale to keep gradient magnitude scale-invariant in T.
        per_pos = per_pos * (T ** 2)

        # Mask invalid positions and average.
        if alignment_mask is None:
            return per_pos.mean()

        mask = alignment_mask.to(per_pos.dtype)
        n_valid = mask.sum().clamp(min=1.0)
        return (per_pos * mask).sum() / n_valid

    def _uld_sorted_kl(
        self,
        student_logits: torch.Tensor,
        teacher_topk_log_values: torch.Tensor,
        alignment_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Universal Logit Distillation (vocab-agnostic fallback).

        Sort BOTH distributions descending, take top-K from each, KL on
        the matching ranks. No vocab projection needed — operates on
        rank-aligned probability mass.
        """
        T = self.temperature
        K = teacher_topk_log_values.shape[-1]
        student_log_probs = (student_logits / T).log_softmax(dim=-1)
        student_top_log, _ = student_log_probs.topk(K, dim=-1)
        # Renormalise both top-K supports; soften teacher at T (proxy logits).
        student_top_log = student_top_log - student_top_log.logsumexp(dim=-1, keepdim=True)
        teacher_top_log = (teacher_topk_log_values / T).log_softmax(dim=-1)
        teacher_top_probs = teacher_top_log.exp()
        per_pos = (teacher_top_probs * (teacher_top_log - student_top_log)).sum(dim=-1)
        # T² rescale (Hinton).
        per_pos = per_pos * (T ** 2)

        if alignment_mask is None:
            return per_pos.mean()
        mask = alignment_mask.to(per_pos.dtype)
        n_valid = mask.sum().clamp(min=1.0)
        return (per_pos * mask).sum() / n_valid

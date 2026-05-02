"""Distillation losses operating on projected teacher distributions.

Each loss takes a student log-probability tensor (or raw logits) and a
teacher distribution that's already been projected to student vocab
(or top-K thereof). The loss handles the masking for non-aligned
positions and renormalisation for partial-mass distributions.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch

LossKind = Literal["kl", "jsd", "mse", "uld_sorted_kl"]


class CTDLoss:
    """Cross-tokenizer distillation loss.

    Wraps a chosen divergence kind and applies the standard
    temperature scaling, mask handling, and (optionally) projection
    via a VocabMapper at training time.

    Args:
        kind: 'kl' | 'jsd' | 'mse' | 'uld_sorted_kl'.
        temperature: applied to teacher logits BEFORE projection.
        mapper: optional VocabMapper. If None, assumes teacher
            top-K cache is already in student vocab (i.e. precompute
            used project_at_write_time=True).
        renormalize_partial: if projected teacher mass is < 1
            (some teacher mass dropped), renormalize before computing
            divergence.
    """

    def __init__(
        self,
        kind: LossKind = "kl",
        temperature: float = 1.0,
        mapper: Optional["VocabMapper"] = None,  # noqa: F821 (fwd ref)
        renormalize_partial: bool = True,
    ):
        self.kind = kind
        self.temperature = temperature
        self.mapper = mapper
        self.renormalize_partial = renormalize_partial

    def __call__(
        self,
        student_logits: torch.Tensor,        # [B, L, V_student]
        teacher_topk_values: torch.Tensor,    # [B, L, K]
        teacher_topk_indices: torch.Tensor,   # [B, L, K]
        alignment_mask: Optional[torch.Tensor] = None,  # [B, L] bool
    ) -> torch.Tensor:
        """Compute distillation loss summed over valid positions.

        Returns scalar loss tensor.
        """
        raise NotImplementedError(
            "CTDLoss.__call__ not yet implemented — see docs/DESIGN.md "
            "section 'Loss formulation'."
        )

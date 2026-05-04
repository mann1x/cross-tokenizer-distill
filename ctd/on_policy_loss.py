"""On-policy distillation losses for full-vocab live-teacher setup.

Used by 06_train_onpolicy.py: at each step the student generates a
continuation, both student and teacher are forwarded on the
(prompt+continuation) sequence, and one of these losses is applied at
the continuation positions only (prompt positions are masked out).

All losses operate on FULL student/teacher logits (same vocab — no
projection needed). For cross-vocab on-policy, project teacher to
student vocab before calling.

Conventions:
- student_logits, teacher_logits: [B, L, V] both for the full sequence.
- mask: [B, L] bool — True where loss should be computed (continuation
  positions, not prompt positions or padding).
- temperature T: applied to BOTH distributions, with T^2 rescale (Hinton).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _masked_mean(per_pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum loss over masked positions, divide by valid count."""
    m = mask.to(per_pos.dtype)
    n = m.sum().clamp(min=1.0)
    return (per_pos * m).sum() / n


def fkl_loss(student_logits, teacher_logits, mask, T=1.0):
    """Forward KL: KL(P_T || P_S). Standard distillation direction.

    Drives student to cover teacher's modes (mode-seeking from student's
    perspective; mass-covering from teacher's). Same as Hinton-KD applied
    on-policy at student-sampled positions.
    """
    s_log = F.log_softmax(student_logits / T, dim=-1)
    t_log = F.log_softmax(teacher_logits / T, dim=-1)
    t_p = t_log.exp()
    per_pos = (t_p * (t_log - s_log)).sum(dim=-1)
    per_pos = per_pos * (T ** 2)
    return _masked_mean(per_pos, mask)


def rkl_loss(student_logits, teacher_logits, mask, T=1.0):
    """Reverse KL: KL(P_S || P_T). MiniLLM (Gu et al. NeurIPS'24) loss.

    Drives student to focus mass on the SUBSET of teacher modes it can
    represent — better suited for student << teacher capacity. Combined
    with on-policy student sampling, this is the canonical MiniLLM setup.
    """
    s_log = F.log_softmax(student_logits / T, dim=-1)
    t_log = F.log_softmax(teacher_logits / T, dim=-1)
    s_p = s_log.exp()
    per_pos = (s_p * (s_log - t_log)).sum(dim=-1)
    per_pos = per_pos * (T ** 2)
    return _masked_mean(per_pos, mask)


def jsd_loss(student_logits, teacher_logits, mask, T=1.0, beta=0.5, eps=1e-12):
    """Generalized JSD: 0.5 * (KL(P_T || M) + KL(P_S || M)) where M = mix.

    GKD (Agarwal et al. ICLR'24) uses this with beta-controlled mixing
    distribution. beta=0.5 = symmetric JSD. beta=0 = forward KL,
    beta=1 = reverse KL. Combined with on-policy sampling = full GKD.
    """
    s_log = F.log_softmax(student_logits / T, dim=-1)
    t_log = F.log_softmax(teacher_logits / T, dim=-1)
    s_p = s_log.exp()
    t_p = t_log.exp()
    mid = (beta * t_p + (1.0 - beta) * s_p).clamp(min=eps)
    mid_log = mid.log()
    kl_tm = (t_p * (t_log - mid_log)).sum(-1)
    kl_sm = (s_p * (s_log - mid_log)).sum(-1)
    per_pos = beta * kl_tm + (1.0 - beta) * kl_sm
    per_pos = per_pos * (T ** 2)
    return _masked_mean(per_pos, mask)


def hybrid_loss(student_logits, teacher_logits, mask, T=1.0, alpha=0.5):
    """Hybrid: alpha * forward KL + (1-alpha) * reverse KL.

    Symmetric blend of mode-seeking + mass-covering pressures.
    """
    return alpha * fkl_loss(student_logits, teacher_logits, mask, T) + \
           (1.0 - alpha) * rkl_loss(student_logits, teacher_logits, mask, T)


LOSSES = {
    "fkl": fkl_loss,
    "rkl": rkl_loss,
    "jsd": jsd_loss,
    "hybrid": hybrid_loss,
}

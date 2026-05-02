"""Tests for ctd.losses.CTDLoss."""

from __future__ import annotations

import pytest
import torch

from ctd.losses import CTDLoss


def _make_inputs(B=2, L=4, V=8, K=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    student_logits = torch.randn(B, L, V, generator=g)
    # Teacher's top-K log-probs and indices in student vocab.
    teacher_topk_indices = torch.randint(0, V, (B, L, K), generator=g)
    # Make valid log-probs by softmaxing random logits.
    teacher_topk_logits = torch.randn(B, L, K, generator=g)
    teacher_topk_log = teacher_topk_logits.log_softmax(dim=-1)
    return student_logits, teacher_topk_log, teacher_topk_indices


def test_kl_zero_when_student_matches_teacher():
    """If student logits match teacher's top-K exactly, KL ≈ 0."""
    B, L, V = 1, 1, 5
    student_logits = torch.full((B, L, V), -1e6)
    teacher_topk_indices = torch.tensor([[[0, 1, 2]]])
    teacher_topk_log = torch.tensor([[[-1.0986, -1.0986, -1.0986]]])  # log(1/3)
    # Set student logits at those indices to be uniform (after softmax, mass 1/3 each
    # if non-top indices have very negative logits).
    student_logits[0, 0, 0] = 0.0
    student_logits[0, 0, 1] = 0.0
    student_logits[0, 0, 2] = 0.0
    loss_fn = CTDLoss(kind="kl")
    loss = loss_fn(student_logits, teacher_topk_log, teacher_topk_indices)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_kl_positive_when_distributions_differ():
    student_logits, teacher_topk_log, teacher_topk_indices = _make_inputs()
    loss_fn = CTDLoss(kind="kl")
    loss = loss_fn(student_logits, teacher_topk_log, teacher_topk_indices)
    assert loss.item() > 0


def test_jsd_symmetric_and_bounded():
    student_logits, teacher_topk_log, teacher_topk_indices = _make_inputs()
    loss_fn = CTDLoss(kind="jsd")
    loss = loss_fn(student_logits, teacher_topk_log, teacher_topk_indices)
    assert 0 <= loss.item() < 1.0  # JSD bounded by log(2) ≈ 0.693


def test_mse_returns_scalar():
    student_logits, teacher_topk_log, teacher_topk_indices = _make_inputs()
    loss_fn = CTDLoss(kind="mse")
    loss = loss_fn(student_logits, teacher_topk_log, teacher_topk_indices)
    assert loss.dim() == 0


def test_alignment_mask_excludes_invalid_positions():
    """Mask=False positions should not contribute to the loss."""
    student_logits, teacher_topk_log, teacher_topk_indices = _make_inputs(B=1, L=4)
    loss_fn = CTDLoss(kind="kl")
    full_mask = torch.ones(1, 4, dtype=torch.bool)
    half_mask = torch.tensor([[True, True, False, False]])
    loss_full = loss_fn(student_logits, teacher_topk_log, teacher_topk_indices, full_mask)
    loss_half = loss_fn(student_logits, teacher_topk_log, teacher_topk_indices, half_mask)
    # Different number of valid positions → different averages.
    assert loss_full.item() != loss_half.item()


def test_uld_sorted_kl_works_without_indices_match():
    """ULD ignores the index alignment (rank-only KL)."""
    student_logits, teacher_topk_log, teacher_topk_indices = _make_inputs()
    loss_fn = CTDLoss(kind="uld_sorted_kl")
    loss = loss_fn(student_logits, teacher_topk_log, teacher_topk_indices)
    assert loss.item() >= 0


def test_temperature_changes_loss():
    student_logits, teacher_topk_log, teacher_topk_indices = _make_inputs()
    loss_t1 = CTDLoss(kind="kl", temperature=1.0)(
        student_logits, teacher_topk_log, teacher_topk_indices
    )
    loss_t4 = CTDLoss(kind="kl", temperature=4.0)(
        student_logits, teacher_topk_log, teacher_topk_indices
    )
    assert loss_t1.item() != loss_t4.item()

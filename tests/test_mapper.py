"""Tests for ctd.mapper.VocabMapper."""

from __future__ import annotations

import pytest
import torch

from ctd.mapper import VocabMapper


class FakeTokenizer:
    """Minimal tokenizer interface — vocab_size + encode + decode.

    Vocabulary represents tokens as fixed strings. Encode does
    longest-match greedy splitting; decode joins.
    """

    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.id_to_str = {i: t for i, t in enumerate(tokens)}
        self.str_to_id = {t: i for i, t in enumerate(tokens)}
        self.vocab_size = len(tokens)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(self.id_to_str.get(i, "") for i in ids)

    def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
        # Greedy longest-match.
        out = []
        i = 0
        while i < len(s):
            best = None
            best_len = 0
            for tok, tid in self.str_to_id.items():
                if not tok:
                    continue
                if s.startswith(tok, i) and len(tok) > best_len:
                    best = tid
                    best_len = len(tok)
            if best is None:
                # No match — drop char (toy behavior).
                i += 1
            else:
                out.append(best)
                i += best_len
        return out


def test_identity_mapping():
    """Same tokenizer on both sides → identity mapping."""
    tok = FakeTokenizer(["a", "b", "ab"])
    mapper = VocabMapper.from_tokenizers(tok, tok, multi_token="distribute", progress=False)
    rep = mapper.coverage_report()
    # Every teacher token decodes/encodes back to a single student token.
    assert rep.single_token_count == 3
    assert rep.multi_token_count == 0
    # Matrix should be identity on dense.
    M = mapper.matrix.to_dense()
    assert torch.allclose(M, torch.eye(3))


def test_split_mapping_distribute():
    """Teacher 'ab' splits into student ['a', 'b'] under distribute strategy."""
    teacher = FakeTokenizer(["a", "b", "ab"])
    student = FakeTokenizer(["a", "b"])
    mapper = VocabMapper.from_tokenizers(
        teacher, student, multi_token="distribute", progress=False
    )
    rep = mapper.coverage_report()
    # 'a' (id=0) and 'b' (id=1) are single-token matches.
    # 'ab' (id=2) is multi-token (splits to a+b → student [0, 1]).
    assert rep.single_token_count == 2
    assert rep.multi_token_count == 1
    M = mapper.matrix.to_dense()
    # M[a_student=0, ab_teacher=2] = 0.5 ; M[b_student=1, ab_teacher=2] = 0.5
    assert M[0, 2].item() == pytest.approx(0.5)
    assert M[1, 2].item() == pytest.approx(0.5)


def test_split_mapping_strict():
    """Strict drops the multi-token mass."""
    teacher = FakeTokenizer(["a", "b", "ab"])
    student = FakeTokenizer(["a", "b"])
    mapper = VocabMapper.from_tokenizers(
        teacher, student, multi_token="strict", progress=False
    )
    M = mapper.matrix.to_dense()
    # The 'ab' column should be all zeros.
    assert M[:, 2].sum().item() == 0.0


def test_split_mapping_first_token():
    """first_token sends all mass to the first sub-token."""
    teacher = FakeTokenizer(["a", "b", "ab"])
    student = FakeTokenizer(["a", "b"])
    mapper = VocabMapper.from_tokenizers(
        teacher, student, multi_token="first_token", progress=False
    )
    M = mapper.matrix.to_dense()
    assert M[0, 2].item() == pytest.approx(1.0)
    assert M[1, 2].item() == 0.0


def test_project_distribution_preserves_mass_when_full_coverage():
    teacher = FakeTokenizer(["a", "b"])
    student = FakeTokenizer(["a", "b"])
    mapper = VocabMapper.from_tokenizers(teacher, student, progress=False)
    teacher_dist = torch.tensor([[0.7, 0.3]])
    student_dist = mapper.project_distribution(teacher_dist)
    assert torch.allclose(student_dist.sum(dim=-1), torch.tensor([1.0]))
    assert torch.allclose(student_dist, torch.tensor([[0.7, 0.3]]))


def test_project_topk_returns_log_distribution():
    teacher = FakeTokenizer(["a", "b", "c"])
    student = FakeTokenizer(["a", "b", "c"])
    mapper = VocabMapper.from_tokenizers(teacher, student, progress=False)
    # Top-3 logits for one position.
    vals = torch.tensor([[1.0, 0.5, 0.0]])
    idx = torch.tensor([[0, 1, 2]])
    log_topk, topk_ids = mapper.project_topk(vals, idx, out_topk=3)
    # exp(log_topk) should sum to 1.
    assert torch.allclose(log_topk.exp().sum(dim=-1), torch.tensor([1.0]), atol=1e-5)
    # Highest logit (value 1.0 at id 0) should have highest projected prob.
    assert topk_ids[0, 0].item() == 0

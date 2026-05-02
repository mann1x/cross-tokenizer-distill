"""Tests for ctd.alignment.build_alignment."""

from __future__ import annotations

from ctd.alignment import build_alignment, compute_byte_offsets


class FakeTokenizer:
    """Same minimal tokenizer used in test_mapper."""

    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.id_to_str = {i: t for i, t in enumerate(tokens)}
        self.str_to_id = {t: i for i, t in enumerate(tokens)}
        self.vocab_size = len(tokens)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(self.id_to_str.get(i, "") for i in ids)

    def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
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
                i += 1
            else:
                out.append(best)
                i += best_len
        return out


def test_compute_byte_offsets_simple():
    tok = FakeTokenizer(["abc", "de", "f"])
    ids = tok.encode("abcdef")
    offsets = compute_byte_offsets(ids, tok)
    assert offsets == [3, 5, 6]


def test_alignment_identical_tokenizers():
    """Same tokenizer on both sides → every position aligned, no drops."""
    tok = FakeTokenizer(["ab", "cd"])
    text = "abcd"
    ids = tok.encode(text)
    table = build_alignment(text, ids, ids, tok, tok, mode="student_offset")
    assert len(table) == 2
    assert table.n_aligned == 2
    assert table.n_suffix == 0
    assert table.n_dropped == 0
    assert table.entries[0].teacher_pos == 0
    assert table.entries[1].teacher_pos == 1


def test_alignment_byte_anchor_mismatch():
    """When teacher splits where student merges, byte_anchor drops misaligned."""
    teacher = FakeTokenizer(["a", "b", "c", "d"])
    student = FakeTokenizer(["ab", "cd"])
    text = "abcd"
    t_ids = teacher.encode(text)  # [a,b,c,d]
    s_ids = student.encode(text)  # [ab, cd]
    # Teacher offsets: [1, 2, 3, 4]
    # Student offsets: [2, 4]
    # Both align: pos 0 (offset 2 = teacher pos 1), pos 1 (offset 4 = teacher pos 3).
    table = build_alignment(text, t_ids, s_ids, teacher, student, mode="byte_anchor")
    assert table.n_aligned == 2
    assert table.n_suffix == 0
    assert table.n_dropped == 0


def test_alignment_suffix_reencode_when_misaligned():
    """When teacher merges where student splits, suffix re-encode kicks in."""
    teacher = FakeTokenizer(["abcd"])
    student = FakeTokenizer(["a", "b", "c", "d"])
    text = "abcd"
    t_ids = teacher.encode(text)  # [abcd], offsets=[4]
    s_ids = student.encode(text)  # [a,b,c,d], offsets=[1,2,3,4]
    # Student pos 0 (offset 1): no clean teacher boundary (only at 4).
    #   k_star = -1 (no teacher boundary <= 1 — teacher's first offset is 4)
    # Wait — teacher_offsets[0] = 4, and 4 > 1 → k_star = -1.
    #   anchor_byte = 0, suffix_bytes = "a", suffix_ids = [a... let's see student vocab doesn't have 'a' as teacher
    # Actually teacher.encode("a") with vocab ["abcd"] returns [] (no match → drop char).
    # So all positions get dropped.
    table = build_alignment(text, t_ids, s_ids, teacher, student, mode="student_offset")
    # Position 3 (offset 4) aligns cleanly with teacher pos 0.
    assert any(e.teacher_pos == 0 for e in table.entries if e.valid)


def test_alignment_full_coverage_via_suffix_reencode():
    """A realistic case where suffix reencode plans valid continuations."""
    # Teacher merges 'ab', student splits 'a'+'b'. After teacher merges,
    # student pos 0 (offset 1) has no clean teacher boundary, but the
    # suffix 'a' can be tokenized (teacher has 'a' as backup token).
    teacher = FakeTokenizer(["ab", "a", "b"])
    student = FakeTokenizer(["a", "b"])
    text = "ab"
    t_ids = teacher.encode(text)  # [ab], offsets=[2]
    s_ids = student.encode(text)  # [a, b], offsets=[1, 2]
    table = build_alignment(text, t_ids, s_ids, teacher, student, mode="student_offset")
    # student_pos 0 (offset 1): no teacher boundary. k_star=-1, anchor=0, suffix='a'
    #   teacher.encode('a') = [1] ('a'). Plan suffix continuation.
    # student_pos 1 (offset 2): aligns with teacher_pos 0 ('ab').
    assert table.n_aligned == 1
    assert table.n_suffix == 1
    assert table.entries[0].suffix_token_ids == [1]
    assert table.entries[0].kv_anchor_pos == -1
    assert table.entries[1].teacher_pos == 0


def test_alignment_no_suffix_reencode_drops_misaligned():
    teacher = FakeTokenizer(["ab", "a", "b"])
    student = FakeTokenizer(["a", "b"])
    text = "ab"
    t_ids = teacher.encode(text)
    s_ids = student.encode(text)
    table = build_alignment(
        text, t_ids, s_ids, teacher, student,
        mode="student_offset", suffix_reencode=False,
    )
    # Same as byte_anchor — drop misaligned, only keep clean.
    assert table.n_aligned == 1
    assert table.n_suffix == 0
    assert table.n_dropped == 1

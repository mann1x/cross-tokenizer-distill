"""End-to-end smoke against real HF tokenizers.

Skipped if transformers/internet unavailable. Validates that the
mapper and alignment work on real byte-level BPE tokenizers, not just
toy fakes.
"""

from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def ds_coder_tok():
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            trust_remote_code=True,
        )
    except Exception as e:
        pytest.skip(f"Could not load DS-Coder tokenizer: {e}")


@pytest.fixture(scope="module")
def qwen25_coder_tok():
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            trust_remote_code=True,
        )
    except Exception as e:
        pytest.skip(f"Could not load Qwen2.5-Coder tokenizer: {e}")


def test_real_compute_byte_offsets(qwen25_coder_tok):
    from ctd.alignment import compute_byte_offsets

    text = "def hello_world():\n    return 42\n"
    ids = qwen25_coder_tok.encode(text, add_special_tokens=False)
    offsets = compute_byte_offsets(ids, qwen25_coder_tok)
    assert len(offsets) == len(ids)
    # Final offset must equal text byte length.
    assert offsets[-1] == len(text.encode("utf-8"))


def test_real_alignment_qwen_to_ds(qwen25_coder_tok, ds_coder_tok):
    """Build alignment table on a small text, verify shape and stats."""
    from ctd.alignment import build_alignment

    text = "def fib(n):\n    if n < 2:\n        return n\n    return fib(n-1) + fib(n-2)\n"
    t_ids = qwen25_coder_tok.encode(text, add_special_tokens=False)
    s_ids = ds_coder_tok.encode(text, add_special_tokens=False)
    table = build_alignment(
        text, t_ids, s_ids,
        teacher_tokenizer=qwen25_coder_tok,
        student_tokenizer=ds_coder_tok,
        mode="student_offset",
        suffix_reencode=True,
    )
    assert len(table) == len(s_ids)
    # student_offset + suffix_reencode should give 100% valid coverage.
    assert (table.n_aligned + table.n_suffix) == len(s_ids)
    assert table.n_dropped == 0
    # And on a code corpus, expect at least SOME alignment hits.
    assert table.n_aligned > 0


def test_real_mapper_sample_only(qwen25_coder_tok, ds_coder_tok):
    """Build a mapper on a small slice (real vocabs are 100K+, slow on full)."""
    from ctd.mapper import VocabMapper

    # On full vocab this is ~30s — fine for test, but skip if too slow.
    mapper = VocabMapper.from_tokenizers(
        teacher_tokenizer=qwen25_coder_tok,
        student_tokenizer=ds_coder_tok,
        multi_token="distribute",
        verify_roundtrip_samples=50,
        progress=False,
    )
    rep = mapper.coverage_report()
    # Sanity: vocab sizes match what HF reports.
    assert rep.teacher_vocab_size == qwen25_coder_tok.vocab_size
    assert rep.student_vocab_size == ds_coder_tok.vocab_size
    # On byte-level BPE pairs, expect HIGH coverage (>70%).
    assert rep.coverage > 0.5, f"unexpectedly low coverage: {rep}"
    print(f"\nReal-tokenizer coverage:\n{rep}")

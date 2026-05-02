"""End-to-end test for ctd.precompute.

Uses a tiny HF model so the test runs in seconds without GPU.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny_model_and_tok():
    """A genuinely tiny model (~10M params) for end-to-end smoke."""
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "sshleifer/tiny-gpt2", torch_dtype=torch.float32
        )
        tok = transformers.AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model.eval()
        return model, tok
    except Exception as e:
        pytest.skip(f"Could not load tiny-gpt2: {e}")


def test_precompute_same_tokenizer_passthrough(tiny_model_and_tok):
    """When teacher and student share a tokenizer, no projection happens —
    output cache should have meaningful aligned entries."""
    model, tok = tiny_model_and_tok
    from ctd.precompute import precompute_aligned_cache

    texts = ["Hello world.", "The quick brown fox.", "def foo():\n    return 42"]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "cache.pt"
        meta = precompute_aligned_cache(
            teacher_model=model,
            teacher_tokenizer=tok,
            student_tokenizer=tok,
            text_corpus=texts,
            output_path=str(out_path),
            top_k=8,
            alignment="student_offset",
            suffix_reencode=True,
            projection=None,
            project_at_write_time=False,
            device="cpu",
            progress=False,
        )

        assert out_path.exists()
        assert meta["n_total_tokens"] > 0
        assert meta["n_aligned_tokens"] > 0
        # Same tokenizer → 100% clean alignment.
        assert meta["n_suffix_reencode"] == 0
        assert meta["n_dropped_tokens"] == 0

        cache = torch.load(out_path, weights_only=False)
        assert cache["values"].shape[1] == 8
        assert cache["values"].shape[0] == cache["indices"].shape[0]
        assert cache["mask"].all()
        # Values are log-probs from log_softmax → all <= 0.
        assert (cache["values"] <= 0).all()


def test_precompute_with_projection(tiny_model_and_tok):
    """Build a mapper (identity, since same tokenizer) and route through projection."""
    model, tok = tiny_model_and_tok
    from ctd.mapper import VocabMapper
    from ctd.precompute import precompute_aligned_cache

    mapper = VocabMapper.from_tokenizers(tok, tok, multi_token="distribute", progress=False)

    texts = ["Hello world."]
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "cache_proj.pt"
        meta = precompute_aligned_cache(
            teacher_model=model,
            teacher_tokenizer=tok,
            student_tokenizer=tok,
            text_corpus=texts,
            output_path=str(out_path),
            top_k=8,
            projection=mapper,
            project_at_write_time=True,
            device="cpu",
            progress=False,
        )
        assert meta["projection_strategy"] == "distribute"
        assert meta["project_at_write_time"] is True
        cache = torch.load(out_path, weights_only=False)
        # Each position's projected log-probs should sum to ≈ 1 (log of renormalised).
        probs_sum = cache["values"].exp().sum(dim=-1)
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-3)

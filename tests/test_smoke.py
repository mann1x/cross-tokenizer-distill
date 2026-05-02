"""Smoke tests — does the package import cleanly?"""

import ctd


def test_package_imports():
    assert ctd.__version__ == "0.0.1"
    assert hasattr(ctd, "VocabMapper")
    assert hasattr(ctd, "CTDLoss")
    assert hasattr(ctd, "precompute_aligned_cache")

"""cross-tokenizer-distill — knowledge distillation across vocabularies.

Public API:

    from ctd import VocabMapper, CTDLoss, precompute_aligned_cache

See README.md for usage and docs/DESIGN.md for the math.
"""

from ctd.mapper import VocabMapper
from ctd.losses import CTDLoss
from ctd.precompute import precompute_aligned_cache
from ctd.util import make_teacher_token_blacklist, bad_words_ids_for_generate

__all__ = [
    "VocabMapper", "CTDLoss", "precompute_aligned_cache",
    "make_teacher_token_blacklist", "bad_words_ids_for_generate",
]
__version__ = "0.0.1"

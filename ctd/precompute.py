"""Standalone teacher logit cache builder.

Produces a top-K teacher cache aligned to student token positions,
optionally projected to student vocab at write time so that the
training pipeline doesn't need any CTD-specific code (drop-in
replacement for same-vocab caches).

Output cache schema (torch.save):

    {
        "values":       Tensor [N_tokens, top_K]  log-probabilities
        "indices":      Tensor [N_tokens, top_K]  vocab indices
                        (student or teacher, depending on
                        project_at_write_time)
        "mask":         Tensor [N_tokens] bool — True where the
                        alignment is valid
        "block_offsets":Tensor [N_examples+1] — block start positions
                        (for sequence-level loading)
        "meta": {
            "teacher_model": str,
            "teacher_tokenizer": str,
            "student_tokenizer": str,
            "alignment": str,
            "suffix_reencode": bool,
            "projection_strategy": str | None,
            "project_at_write_time": bool,
            "top_k": int,
            "n_total_tokens": int,
            "n_aligned_tokens": int,
            "n_suffix_reencode": int,
            "n_dropped_tokens": int,
            "ctd_version": str,
            "seed": int,
        },
    }

This shape matches existing same-vocab top-K caches (e.g. Mythic-RDT
`teacher_cache/dscoder_*.pt`), so trainers need no changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Literal, Optional

import torch
from tqdm import tqdm

from ctd.alignment import build_alignment

# Avoid circular import — read version from package __init__ at call time.
def _get_version() -> str:
    import ctd
    return ctd.__version__

Tokenizer = "transformers.PreTrainedTokenizerBase"  # noqa: F821
PreTrainedModel = "transformers.PreTrainedModel"  # noqa: F821
VocabMapper = "ctd.mapper.VocabMapper"  # noqa: F821


@torch.no_grad()
def _full_teacher_forward(
    teacher_model,
    teacher_input_ids: torch.Tensor,  # [1, L_t]
) -> tuple[torch.Tensor, object]:
    """Run teacher forward, return (logits, past_key_values).

    Logits shape: [1, L_t, V_teacher]. KV cache is the model's native
    Cache object (DynamicCache for modern transformers, tuple for
    legacy).
    """
    out = teacher_model(
        input_ids=teacher_input_ids,
        use_cache=True,
        return_dict=True,
    )
    return out.logits, out.past_key_values


def _slice_kv_cache(past_key_values, end_pos: int):
    """Truncate a KV cache to keep only positions [0, end_pos).

    Always returns a Cache-compatible object (DynamicCache when modern
    transformers is available) or None for empty. Modern HF models
    (Qwen2, Llama-3, etc.) only accept Cache instances on the input
    side, not legacy tuples — so we normalise everything here.

    end_pos is exclusive — passing 0 returns None.
    """
    if past_key_values is None or end_pos == 0:
        return None

    # Convert any input (legacy tuple OR Cache instance) to legacy form,
    # slice, then convert back to DynamicCache.
    if isinstance(past_key_values, tuple):
        legacy = past_key_values
    elif hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
    else:
        raise TypeError(
            f"Unsupported KV cache type: {type(past_key_values)}. "
            f"Need legacy tuple or HF Cache subclass with to_legacy_cache()."
        )

    sliced_legacy = tuple(
        (k[..., :end_pos, :].clone(), v[..., :end_pos, :].clone())
        for (k, v) in legacy
    )

    # Convert back to a Cache-compatible object if available.
    try:
        from transformers import DynamicCache
        return DynamicCache.from_legacy_cache(sliced_legacy)
    except ImportError:
        return sliced_legacy


@torch.no_grad()
def _suffix_continuation_logit(
    teacher_model,
    suffix_ids: list[int],
    kv_cache,
    device: str,
) -> torch.Tensor:
    """Run teacher forward on suffix_ids continuing from kv_cache.

    Returns the FINAL logit (after the last suffix token) — that's
    the next-token prediction at the student's byte offset.

    Shape: [V_teacher].
    """
    suffix_tensor = torch.tensor([suffix_ids], dtype=torch.long, device=device)
    out = teacher_model(
        input_ids=suffix_tensor,
        past_key_values=kv_cache,
        use_cache=False,  # don't grow the cache, we won't reuse
        return_dict=True,
    )
    return out.logits[0, -1, :]  # [V_teacher]


def _project_or_passthrough(
    logit_or_topk: torch.Tensor,  # [V_teacher] dense OR [K] top-K values
    topk_indices: Optional[torch.Tensor],  # [K] or None
    top_k: int,
    projection: Optional["VocabMapper"],
    project_at_write_time: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take a teacher distribution (dense or top-K) and produce final
    cache (values, indices) for one position.

    Returns:
        (values, indices) — both shape [top_k]. Values are
        log-probabilities. Indices are in student vocab if projected,
        else teacher vocab.
    """
    if projection is None or not project_at_write_time:
        # Just take top-K of teacher's distribution.
        if topk_indices is not None:
            # Already top-K from caller.
            top_vals = logit_or_topk
            top_ids = topk_indices
        else:
            # Dense logits, take top-K.
            top_vals, top_ids = logit_or_topk.topk(top_k)
        log_probs = top_vals.log_softmax(dim=-1)
        return log_probs, top_ids

    # Project to student vocab.
    if topk_indices is None:
        # Dense → take large enough top-K first to feed projection.
        top_vals, top_ids = logit_or_topk.topk(min(top_k * 4, logit_or_topk.shape[-1]))
    else:
        top_vals = logit_or_topk
        top_ids = topk_indices

    # project_topk expects [..., K] — add a leading batch dim.
    log_proj, ids_proj = projection.project_topk(
        top_vals.unsqueeze(0),
        top_ids.unsqueeze(0),
        out_topk=top_k,
        already_softmaxed=False,
    )
    return log_proj[0], ids_proj[0]


def precompute_aligned_cache(
    teacher_model,
    teacher_tokenizer,
    student_tokenizer,
    text_corpus: Iterable[str],
    output_path: str,
    top_k: int = 32,
    alignment: Literal["byte_anchor", "student_offset"] = "student_offset",
    suffix_reencode: bool = True,
    projection: Optional["VocabMapper"] = None,
    project_at_write_time: bool = True,
    max_seq_len: int = 2048,
    device: str = "cuda",
    seed: int = 0,
    progress: bool = True,
) -> dict:
    """Build a top-K teacher cache aligned to the student tokenization.

    Args:
        teacher_model: HF model (already on device, in eval mode).
        teacher_tokenizer / student_tokenizer: HF tokenizers.
        text_corpus: iterable of text strings.
        output_path: path to write the .pt cache.
        top_k: number of top logits to retain per position.
        alignment: 'byte_anchor' or 'student_offset'.
        suffix_reencode: enable smart KV-cache reuse for non-aligned
            positions (only effective with 'student_offset').
        projection: VocabMapper instance. Required when
            project_at_write_time=True.
        project_at_write_time: if True, apply mapper.project_topk()
            during precompute so the cache file uses student vocab
            indices. Recommended.
        max_seq_len: truncate examples longer than this (student-side
            tokens).
        device: 'cuda' typically.
        seed: recorded in meta.
        progress: tqdm progress bar over corpus.

    Returns:
        Dict with summary stats.
    """
    if project_at_write_time and projection is None:
        raise ValueError("project_at_write_time=True requires projection=VocabMapper(...)")

    teacher_model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_values: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []
    all_mask: list[torch.Tensor] = []
    block_offsets: list[int] = [0]

    n_total = 0
    n_aligned = 0
    n_suffix = 0
    n_dropped = 0

    iterator = text_corpus
    if progress:
        # If text_corpus is a list/iterable with len, tqdm shows total.
        try:
            total = len(text_corpus)  # type: ignore
        except Exception:
            total = None
        iterator = tqdm(text_corpus, total=total, desc="precompute")

    for text in iterator:
        if not text:
            continue

        # Tokenize both sides.
        teacher_ids_list = teacher_tokenizer.encode(text, add_special_tokens=False)
        student_ids_list = student_tokenizer.encode(text, add_special_tokens=False)

        if not teacher_ids_list or not student_ids_list:
            continue

        # Truncate student-side; teacher truncates to whatever covers
        # the student's max_seq_len in bytes (we'll let alignment skip
        # over-long teacher positions naturally).
        student_ids_list = student_ids_list[:max_seq_len]

        # Build alignment FIRST so we know which teacher positions we need.
        table = build_alignment(
            text=text,
            teacher_token_ids=teacher_ids_list,
            student_token_ids=student_ids_list,
            teacher_tokenizer=teacher_tokenizer,
            student_tokenizer=student_tokenizer,
            mode=alignment,
            suffix_reencode=suffix_reencode,
        )

        # Run teacher forward once on the full teacher sequence.
        # Convert to tensor on device.
        t_ids = torch.tensor([teacher_ids_list], dtype=torch.long, device=device)
        try:
            t_logits, t_kv = _full_teacher_forward(teacher_model, t_ids)
        except torch.cuda.OutOfMemoryError:
            # Skip examples that are too big for the device.
            continue

        # Per-position: pull teacher logit (or run suffix continuation).
        ex_values = []
        ex_indices = []
        ex_mask = []

        for entry in table.entries:
            if not entry.valid:
                # Placeholder zeros so the cache shape matches student_ids_list.
                ex_values.append(torch.zeros(top_k, dtype=torch.float32))
                if project_at_write_time:
                    ex_indices.append(torch.zeros(top_k, dtype=torch.long))
                else:
                    ex_indices.append(torch.zeros(top_k, dtype=torch.long))
                ex_mask.append(False)
                n_dropped += 1
                continue

            if entry.suffix_token_ids is None:
                # Clean alignment — pull teacher logit at teacher_pos.
                logit = t_logits[0, entry.teacher_pos, :]  # [V_teacher]
                n_aligned += 1
            else:
                # Suffix re-encode.
                anchor_kv = _slice_kv_cache(t_kv, entry.kv_anchor_pos + 1)
                logit = _suffix_continuation_logit(
                    teacher_model,
                    entry.suffix_token_ids,
                    anchor_kv,
                    device=device,
                )
                n_suffix += 1

            # Take top-K (raw logits → projection layer expects logits).
            top_vals, top_ids = logit.topk(top_k)
            values, indices = _project_or_passthrough(
                top_vals, top_ids, top_k, projection, project_at_write_time
            )
            ex_values.append(values.cpu().float())
            ex_indices.append(indices.cpu().long())
            ex_mask.append(True)

        # Stack and append to global cache.
        ex_values_t = torch.stack(ex_values)        # [L_s, top_k]
        ex_indices_t = torch.stack(ex_indices)       # [L_s, top_k]
        ex_mask_t = torch.tensor(ex_mask, dtype=torch.bool)  # [L_s]

        all_values.append(ex_values_t)
        all_indices.append(ex_indices_t)
        all_mask.append(ex_mask_t)
        n_total += ex_values_t.shape[0]
        block_offsets.append(n_total)

        # Free the per-example KV cache.
        del t_logits, t_kv

    if not all_values:
        raise RuntimeError("No examples produced any cache entries.")

    cache = {
        "values": torch.cat(all_values, dim=0),
        "indices": torch.cat(all_indices, dim=0),
        "mask": torch.cat(all_mask, dim=0),
        "block_offsets": torch.tensor(block_offsets, dtype=torch.long),
        "meta": {
            "teacher_tokenizer": getattr(teacher_tokenizer, "name_or_path", "<unknown>"),
            "student_tokenizer": getattr(student_tokenizer, "name_or_path", "<unknown>"),
            "alignment": alignment,
            "suffix_reencode": suffix_reencode,
            "projection_strategy": projection.strategy if projection else None,
            "project_at_write_time": project_at_write_time,
            "top_k": top_k,
            "n_total_tokens": n_total,
            "n_aligned_tokens": n_aligned,
            "n_suffix_reencode": n_suffix,
            "n_dropped_tokens": n_dropped,
            "ctd_version": _get_version(),
            "seed": seed,
            "max_seq_len": max_seq_len,
        },
    }

    torch.save(cache, output_path)
    # Also dump meta as a sidecar JSON for easy inspection.
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(cache["meta"], f, indent=2)

    return cache["meta"]

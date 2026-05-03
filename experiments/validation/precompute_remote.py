"""Build a CTD top-K teacher cache against a REMOTE teacher.

Supports:
- Local Ollama (`--provider ollama`) — useful for free experimentation
  and small caches; one HTTP call per token, slow at scale.
- OpenAI-compat hosted (`--provider {together,fireworks,deepinfra}`) —
  one call per example via `echo: true` + `logprobs: K`. Good for
  production-scale precompute against massive teachers without
  renting a GPU pod.

Output cache shape is identical to ctd.precompute, so trainers
(05_train.py / Mythic-RDT CTDTrainer) need no changes.

Usage examples:

    # Local Ollama, gemma4:e2b, no projection (keep teacher vocab).
    python precompute_remote.py \\
        --provider ollama \\
        --base-url http://solidpc:11433 \\
        --model-id gemma4:e2b \\
        --teacher-tokenizer google/gemma-2-2b \\
        --student-tokenizer deepseek-ai/deepseek-coder-1.3b-instruct \\
        --corpus data/mini_corpus.jsonl \\
        --output cache/ollama_gemma4_e2b.pt \\
        --top-k 20

    # Together.AI, Qwen3-Coder-480B, project to student vocab.
    TOGETHER_API_KEY=... python precompute_remote.py \\
        --provider together \\
        --model-id Qwen/Qwen3-Coder-480B \\
        --teacher-tokenizer Qwen/Qwen3-Coder \\
        --student-tokenizer deepseek-ai/deepseek-coder-1.3b-instruct \\
        --corpus data/corpus_5k.jsonl \\
        --output cache/together_qwen3_coder_480b.pt \\
        --top-k 20 \\
        --project-at-write-time
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from ctd.alignment import build_alignment
from ctd.mapper import VocabMapper
from ctd.teachers import (
    OllamaTeacher,
    OpenAICompletionsTeacher,
    PositionLogprobs,
    TeacherBackend,
)


PROVIDER_PRESETS = {
    "ollama": {
        "default_base_url": "http://localhost:11434",
        "needs_api_key": False,
    },
    "together": {
        "default_base_url": "https://api.together.xyz/v1",
        "needs_api_key": True,
        "api_key_env": "TOGETHER_API_KEY",
    },
    "fireworks": {
        "default_base_url": "https://api.fireworks.ai/inference/v1",
        "needs_api_key": True,
        "api_key_env": "FIREWORKS_API_KEY",
    },
    "deepinfra": {
        "default_base_url": "https://api.deepinfra.com/v1/openai",
        "needs_api_key": True,
        "api_key_env": "DEEPINFRA_API_KEY",
    },
}


def make_backend(args, teacher_tokenizer) -> TeacherBackend:
    preset = PROVIDER_PRESETS[args.provider]
    base_url = args.base_url or preset["default_base_url"]
    if args.provider == "ollama":
        return OllamaTeacher(
            base_url=base_url,
            model_id=args.model_id,
            tokenizer=teacher_tokenizer,
            top_k=args.top_k,
            timeout=args.timeout,
        )
    api_key = args.api_key or os.environ.get(preset["api_key_env"])
    if not api_key:
        raise SystemExit(
            f"Provider '{args.provider}' needs an API key; set "
            f"--api-key or env {preset['api_key_env']}."
        )
    return OpenAICompletionsTeacher(
        base_url=base_url,
        model_id=args.model_id,
        tokenizer=teacher_tokenizer,
        api_key=api_key,
        top_k=args.top_k,
        timeout=args.timeout,
    )


def topk_to_tensors(
    pos: PositionLogprobs, top_k: int
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Pad / truncate one position's top-K to fixed shape, return (vals, ids, valid)."""
    if not pos.topk_ids:
        return (
            torch.zeros(top_k, dtype=torch.float32),
            torch.zeros(top_k, dtype=torch.long),
            False,
        )
    ids = list(pos.topk_ids)[:top_k]
    vals = list(pos.topk_logprobs)[:top_k]
    pad = top_k - len(ids)
    if pad > 0:
        ids = ids + [0] * pad
        vals = vals + [-1e30] * pad
    return (
        torch.tensor(vals, dtype=torch.float32),
        torch.tensor(ids, dtype=torch.long),
        True,
    )


def project_position(
    vals: torch.Tensor,
    ids: torch.Tensor,
    top_k: int,
    projection: Optional[VocabMapper],
) -> tuple[torch.Tensor, torch.Tensor]:
    if projection is None:
        # Keep teacher vocab; turn logprobs into the same domain.
        # Renormalise within the top-K so the cache stays well-defined.
        return vals.log_softmax(dim=-1), ids
    log_proj, ids_proj = projection.project_topk(
        vals.unsqueeze(0),
        ids.unsqueeze(0),
        out_topk=top_k,
        already_softmaxed=False,
    )
    return log_proj[0], ids_proj[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=list(PROVIDER_PRESETS))
    parser.add_argument("--model-id", required=True,
                        help="Provider-side model id (e.g. 'gemma4:e2b' for Ollama, 'Qwen/Qwen3-Coder-480B' for Together).")
    parser.add_argument("--base-url", default=None,
                        help="Override provider default base URL.")
    parser.add_argument("--api-key", default=None,
                        help="API key (or use the provider's env var).")
    parser.add_argument("--teacher-tokenizer", required=True,
                        help="HF tokenizer id matching the remote teacher (loaded locally).")
    parser.add_argument("--student-tokenizer", required=True,
                        help="HF tokenizer id of the student.")
    parser.add_argument("--corpus", required=True,
                        help="JSONL file with {'text': ...} per line.")
    parser.add_argument("--output", required=True, help="Cache .pt path.")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top-K to retain per position. Most providers cap at 20.")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-examples", type=int, default=0,
                        help="0 = all.")
    parser.add_argument("--alignment", default="student_offset",
                        choices=["byte_anchor", "student_offset"])
    parser.add_argument("--project-at-write-time", action="store_true",
                        help="Project to student vocab via VocabMapper. "
                             "Otherwise the cache stays in teacher vocab.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"[precompute-remote] provider={args.provider} model={args.model_id}")
    print(f"[precompute-remote] teacher_tok={args.teacher_tokenizer}  student_tok={args.student_tokenizer}")
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_tokenizer)
    student_tok = AutoTokenizer.from_pretrained(args.student_tokenizer)

    backend = make_backend(args, teacher_tok)

    projection: Optional[VocabMapper] = None
    if args.project_at_write_time:
        print("[precompute-remote] Building VocabMapper (teacher → student)...")
        projection = VocabMapper(
            source_tokenizer=teacher_tok,
            target_tokenizer=student_tok,
        )

    texts: List[str] = []
    with open(args.corpus) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("text") or ""
            if t:
                texts.append(t)
    if args.max_examples:
        texts = texts[: args.max_examples]
    print(f"[precompute-remote] Corpus: {len(texts)} examples")

    all_values: List[torch.Tensor] = []
    all_indices: List[torch.Tensor] = []
    all_mask: List[torch.Tensor] = []
    block_offsets: List[int] = [0]
    n_total = 0
    n_aligned = 0
    n_dropped = 0

    for text in tqdm(texts, desc="precompute-remote"):
        try:
            teacher_ids, per_pos = backend.get_token_topk(text)
        except Exception as exc:  # one bad example shouldn't kill the run
            print(f"  skip example (backend error): {type(exc).__name__}: {str(exc)[:120]}")
            continue

        student_ids = student_tok.encode(text, add_special_tokens=False)[: args.max_seq_len]
        if not student_ids or not teacher_ids:
            continue

        table = build_alignment(
            text=text,
            teacher_token_ids=teacher_ids,
            student_token_ids=student_ids,
            teacher_tokenizer=teacher_tok,
            student_tokenizer=student_tok,
            mode=args.alignment,
            suffix_reencode=False,  # remote teachers can't do KV-suffix tricks
        )

        ex_values: List[torch.Tensor] = []
        ex_indices: List[torch.Tensor] = []
        ex_mask: List[bool] = []
        for entry in table.entries:
            if (
                not entry.valid
                or entry.suffix_token_ids is not None
                or entry.teacher_pos is None
                or entry.teacher_pos >= len(per_pos)
            ):
                ex_values.append(torch.zeros(args.top_k, dtype=torch.float32))
                ex_indices.append(torch.zeros(args.top_k, dtype=torch.long))
                ex_mask.append(False)
                n_dropped += 1
                continue
            vals, ids, valid = topk_to_tensors(per_pos[entry.teacher_pos], args.top_k)
            if not valid:
                ex_values.append(vals)
                ex_indices.append(ids)
                ex_mask.append(False)
                n_dropped += 1
                continue
            proj_vals, proj_ids = project_position(vals, ids, args.top_k, projection)
            ex_values.append(proj_vals)
            ex_indices.append(proj_ids)
            ex_mask.append(True)
            n_aligned += 1

        if not ex_values:
            continue
        all_values.append(torch.stack(ex_values))
        all_indices.append(torch.stack(ex_indices))
        all_mask.append(torch.tensor(ex_mask, dtype=torch.bool))
        n_total += len(ex_values)
        block_offsets.append(n_total)

    if not all_values:
        raise SystemExit("No examples produced any cache entries.")

    cache = {
        "values": torch.cat(all_values, dim=0),
        "indices": torch.cat(all_indices, dim=0),
        "mask": torch.cat(all_mask, dim=0),
        "block_offsets": torch.tensor(block_offsets, dtype=torch.long),
        "meta": {
            "teacher_provider": args.provider,
            "teacher_model_id": args.model_id,
            "teacher_tokenizer": args.teacher_tokenizer,
            "student_tokenizer": args.student_tokenizer,
            "alignment": args.alignment,
            "suffix_reencode": False,
            "projection_strategy": projection.strategy if projection else None,
            "project_at_write_time": bool(projection),
            "top_k": args.top_k,
            "n_total_tokens": n_total,
            "n_aligned_tokens": n_aligned,
            "n_dropped_tokens": n_dropped,
            "n_suffix_reencode": 0,
            "max_seq_len": args.max_seq_len,
            "seed": args.seed,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.output)
    meta_path = Path(args.output).with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(cache["meta"], f, indent=2)
    print(f"[precompute-remote] Wrote {args.output} "
          f"(n_total={n_total} n_aligned={n_aligned} n_dropped={n_dropped})")
    print(f"[precompute-remote] Meta → {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/bin/bash
# Phase 7 (post-orchestrator): full HE-164 on BASE (no LoRA) so we can compare
# the SFT winner against base on the full eval — the smoke set is biased toward
# A_OLD-passes (sweep gate may be misleading).
set -uo pipefail
cd /workspace/cross-tokenizer-distill/experiments/validation
source /workspace/venv-tf4/bin/activate
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/cross-tokenizer-distill

LOGDIR=logs/overnight
mkdir -p $LOGDIR results/full_sft

echo "[p7] waiting for decision_overnight.md..."
until [ -f results/decision_overnight.md ]; do sleep 60; done
echo "[p7] decision_overnight.md found at $(date -u +%FT%TZ)"

# Base HE-164 — adapter-less. 06_eval.py expects --adapter, so use a tiny shim:
# load PEFT with no adapter id by patching the call; easier path is to just use
# 06_eval directly with a "null" adapter that resolves to the base.
# But peft requires a real adapter dir. Workaround: use 06_eval source to know
# how it loads, then write base_full_he.py inline.
cat > /tmp/base_full_he.py << "PYEOF"
import argparse, json, sqlite3, sys, signal, time
from pathlib import Path
from datasets import load_dataset
sys.path.insert(0, "/workspace/cross-tokenizer-distill/experiments/validation")
from importlib import import_module
ev = import_module("06_eval")

p = argparse.ArgumentParser()
p.add_argument("--base-model", default="deepseek-ai/deepseek-coder-1.3b-instruct")
p.add_argument("--output", required=True)
p.add_argument("--cache-db", required=True)
p.add_argument("--he-limit", type=int, default=164)
p.add_argument("--exec-timeout", type=int, default=30)
args = p.parse_args()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
print("[base-he] loading base model", flush=True)
tok = AutoTokenizer.from_pretrained(args.base_model)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map={"": "cuda"})
model.eval()

ds = load_dataset("openai/openai_humaneval", split="test")
problems = list(ds)[:args.he_limit]
con = ev.open_cache(args.cache_db)
done = ev.cache_done_ids(con, "humaneval")
n_pass = 0; n_done = 0; failures = []
for i, prob in enumerate(problems, 1):
    tid = prob["task_id"]
    if tid in done:
        row = con.execute("SELECT passed, info FROM results WHERE task=? AND task_id=?", ("humaneval",tid)).fetchone()
        passed = bool(row[0]); info = row[1] or ""
    else:
        t0 = time.time()
        gen = ev.generate_one(model, tok, prob["prompt"])
        gt = time.time() - t0
        t1 = time.time()
        passed, info = ev.score_humaneval(prob, gen, timeout_s=args.exec_timeout)
        et = time.time() - t1
        ev.cache_save(con, "humaneval", tid, passed, info, gen)
        running = (n_pass + (1 if passed else 0)) / i * 100.0
        verdict = "PASS" if passed else "FAIL"
        print(f"  HE {i}/{len(problems)} {tid:25s} {verdict} gen={gt:5.1f}s exec={et:5.2f}s running={n_pass+(1 if passed else 0)}/{i}={running:.1f}% info={info[:50]}", flush=True)
    if passed: n_pass += 1
    else: failures.append({"task_id":tid, "info":info[:100]})
    n_done = i

out = {"adapter":"BASE_NO_FINETUNE","humaneval":{"n_total":n_done,"n_pass":n_pass,"pass@1":n_pass/n_done,"failures":failures}}
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
open(args.output,"w").write(json.dumps(out, indent=2))
print(f"[base-he] DONE {n_pass}/{n_done} = {n_pass/n_done:.1%} → {args.output}", flush=True)
PYEOF

if [ ! -f results/full_sft/BASE_HE164.json ]; then
    echo "[p7] running base HE-164..."
    python -u /tmp/base_full_he.py \
        --output results/full_sft/BASE_HE164.json \
        --cache-db results/full_sft/BASE_HE164.db \
        --he-limit 164 --exec-timeout 30 \
        2>&1 | tee $LOGDIR/base_he164.log | tail -3
else
    echo "[p7] SKIP base HE-164 (exists)"
fi

# Append to decision_overnight.md
python3 << "PYEOF" >> results/decision_overnight.md
import json
print()
print("## Phase 7: BASE full HE-164")
b = json.load(open("results/full_sft/BASE_HE164.json"))["humaneval"]
print(f"- BASE: HE pass@1 = {b[\"n_pass\"]}/{b[\"n_total\"]} = {b[\"pass@1\"]:.1%}")
print()
print("### Verdict vs BASE (full HE-164):")
import os
from pathlib import Path
for f in sorted(Path("results/full_sft").glob("S*_HE164.json")):
    r = json.load(open(f))["humaneval"]
    delta = r["n_pass"] - b["n_pass"]
    print(f"- **{f.stem}**: {r[\"n_pass\"]}/{r[\"n_total\"]} = {r[\"pass@1\"]:.1%} (delta_vs_base = {delta:+d})")
# A_OLD (results/A.json) and B_R04 / C_R04 too
for label, path in [("A (old SFT)","results/A.json"),("B_R04","results/B_R04.json"),("C_R04","results/C_R04.json")]:
    try:
        r = json.load(open(path))["humaneval"]
        delta = r["n_pass"] - b["n_pass"]
        print(f"- **{label}**: {r[\"n_pass\"]}/{r[\"n_total\"]} = {r[\"pass@1\"]:.1%} (delta_vs_base = {delta:+d})")
    except Exception as e:
        print(f"- {label}: missing ({e})")
PYEOF

echo "[p7] DONE $(date -u +%FT%TZ)"

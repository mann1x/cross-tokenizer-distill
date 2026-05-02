# Validation experiment — small-models 3-way A/B/C

See `../../docs/VALIDATION.md` for the full design doc. This README is
the operational quick-reference.

## Run order

```bash
# 0. Setup
conda create -n ctd python=3.11
conda activate ctd
pip install -e ../..
pip install datasets accelerate bitsandbytes

# 1. Diagnostic — confirm vocab pair is workable
bash 01_inspect.sh

# 2. Build same-vocab teacher cache (Qwen2.5-Coder-7B → Qwen2.5-Coder-0.5B)
bash 02_precompute_B.sh

# 3. Build CTD cache (DS-Coder-V2-Lite → Qwen2.5-Coder-0.5B via CTD)
bash 03_precompute_C.sh

# 4-6. Three training runs (sequential on solidPC 3090)
bash 04_train_A.sh   # SFT only
bash 05_train_B.sh   # same-vocab distill
bash 06_train_C.sh   # CTD distill

# 7. Eval all three
bash 07_eval_all.sh

# 8. Decision report
python 08_compare.py
```

## Status

Stub only. Run scripts to be filled in once core CTD modules are
implemented (tasks #75 and #72).

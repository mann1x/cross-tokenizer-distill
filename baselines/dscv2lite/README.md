# DeepSeek-Coder-V2-Lite-Instruct baselines

Eval JSONs for DSC-V2-Lite-Instruct (16B MoE, 2.4B active) on the canonical
HumanEval-164 + MBPP-378 test splits. Pinned to the repo because the
`experiments/validation/results/` tree is gitignored and pod-side state is
ephemeral — vast.ai pod destruction wipes any results JSON not previously
committed (lesson learned 2026-05-08, see
`memory/feedback_pin_baseline_jsons_to_repo.md`).

## Eval recipe (all entries)

- Base loaded from the locally-patched `/workspace/mythic-rdt/base/DeepSeek-Coder-V2-Lite-Instruct`
  (which has the vectorized MoE routing patched into `modeling_deepseek.py`).
- `06_eval_batched.py` flags: `--quant bf16 --chat-template --batch-size 8 --max-new-tokens 1024 --exec-timeout 30 --he-limit 164 --mbpp-limit 378`.
- Stack: FA2 (`attn_implementation="flash_attention_2"`) + `merge_and_unload()`
  for adapters (no-op for BASE). Wired in commit `c34662b`.
- bs=8 instead of 16 because BF16 16B + KV cache + max_new=1024 OOMed at HE/16
  on 44 GB VRAM with bs=16.

## Files

| File | HE pass@1 | MBPP pass@1 | Notes |
|---|---:|---:|---|
| `BASE_FA2_CHAT_v2_HE_MBPP.json` | 73.8% | 64.6% | DSC-V2-Lite-Instruct base, no adapter |
| `M115_DSCV2LITE_M37c_HE_MBPP.json` | 71.3% | 66.1% | M37c recipe (funcsig + --code-only-mask, lr=5e-5, r=16, 2 epochs) |

The doc `BASE_DEEPSEEK_CODER_V2_LITE.md` (commit 88164cc) reports 75.6 / 60.6
for the base — measured on a prior vast.ai pod (RTX 4090, 24GB) that was
later destroyed. The current re-eval reproduces within ~2pp on HE
(73.8 vs 75.6) and within ~4pp on MBPP (64.6 vs 60.6, possibly different
test-split definition or scorer rstrip vs strip nuance). Treat **64.6 / 73.8
as the canonical baseline** for any v2 adapter on this pod.

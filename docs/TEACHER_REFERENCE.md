# Teacher reference numbers

Reported HumanEval+ / HumanEval / MBPP / LiveCodeBench scores for the
teacher candidates we've used or considered. All sourced from the
respective technical reports / model cards. Use as ceiling estimates
for distillation targets.

## Used in our experiments

| Teacher | HumanEval+ | HumanEval | MBPP+ | LiveCodeBench v5 | Tokenizer / vocab | Source |
|---|---:|---:|---:|---:|---|---|
| **DS-Coder-V2-Lite-Instruct** (16B/2.4B-A) | n/a (~88) | 81.1 | 68.8 | n/a | DeepSeek (102K) | [paper](https://arxiv.org/abs/2406.11931) |
| **DeepSeek-Coder-6.7B-Instruct** | n/a (~80) | 78.6 | 65.4 | n/a | DeepSeek (32K) | [paper](https://arxiv.org/abs/2401.14196) |
| **Qwen2.5-Coder-7B-Instruct** | **84.1** | 88.4 | 73.3 | n/a | Qwen (152K) | [tech report](https://arxiv.org/html/2409.12186v1) |
| **DeepCoder-14B-Preview** | **92.6** | n/a | n/a | **60.6** | Qwen (152K, inherited from DeepSeek-R1-Distill-Qwen-14B) | [model card](https://huggingface.co/agentica-org/DeepCoder-14B-Preview) |

## Reference tier (not used)

| Teacher | HumanEval+ | HumanEval | MBPP+ | LiveCodeBench v5 | Notes |
|---|---:|---:|---:|---:|---|
| Qwen2.5-Coder-32B-Instruct | 86.0 | 92.7 | 76.5 | n/a | bigger but slower precompute |
| Qwen3-Coder-30B-A3B | n/a | ~90 | n/a | n/a | MoE; transformers ≥ 4.57 |
| Qwen3-Coder-Next-80B-A3B | n/a | ~88-92 (est.) | n/a | n/a | MoE 3B-active; SOTA on SWE-Bench |
| DeepSeek-Coder-V2 236B | n/a (~94) | 90.2 | 76.2 | n/a | huge; needs 2× H100 |
| GPT-4o (ref baseline) | 86.6 | 90.2 | 81.0 | n/a | API |
| o3-mini (DeepCoder peer) | 92.6 | n/a | n/a | 60.9 | API |

## Our student baseline

| Model | HumanEval | MBPP | Tokenizer / vocab |
|---|---:|---:|---|
| **DS-Coder-1.3B-Instruct** (the student) | **59.8** | **61.1** | DeepSeek (32K) |

## Implications for distillation ceiling

The teacher's reported HE-pass@1 sets a hard upper bound on what
distillation can reach. With DS-Coder-1.3B base at 59.8 % HE:

- Qwen2.5-Coder-7B-Inst (88.4 / 84.1+) → distill ceiling ≈ ~80 % if we
  capture most of the teacher's signal. Our best M6b at 53 % captures
  ~10 % of the gap (53 = 59.8 + (88.4-59.8) × 0.0). We have a long way
  to go.
- DeepCoder-14B-Preview (92.6 HE+) → distill ceiling ≈ ~85 %.
  Strongest teacher we've considered. Reasoning model — emits
  `<think>...</think>` blocks before code, must strip before
  scoring/training.
- DS-Coder-V2-Lite (~88 HE+) → distill ceiling ≈ ~80 %. Same-vocab
  with our student (both DeepSeek tokenizer) — no projection cost.

## Note on author of this document

I previously stated Qwen2.5-Coder-7B-Inst was at "88-90 % HE+" in
chat — that's wrong. **The official report has it at 84.1 % HumanEval+
(88.4 % HumanEval)**. The 88-92 % range applies to Qwen2.5-Coder-32B
and the much larger Qwen3-Coder-Next-80B-A3B / Coder-480B variants.
Don't confuse model sizes when reasoning about teacher ceilings.

Sources:
- [Qwen2.5-Coder Technical Report (arxiv)](https://arxiv.org/html/2409.12186v1)
- [Qwen2.5-Coder Series blog post](https://qwenlm.github.io/blog/qwen2.5-coder-family/)
- [Qwen2.5-Coder-7B-Instruct on HF](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- [llm-stats Qwen2.5-Coder-7B-Instruct benchmarks](https://llm-stats.com/models/qwen-2.5-coder-7b-instruct)
- [DeepCoder-14B-Preview model card](https://huggingface.co/agentica-org/DeepCoder-14B-Preview)

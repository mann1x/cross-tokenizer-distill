# Teacher token masking

Generic mechanism to suppress specific teacher-vocab tokens across every
CTD code path that uses a teacher (KL training, teacher generation,
teacher eval). Built primarily to handle **thinking-mode reasoning
teachers** (DeepSeek-R1-Distill family, DeepCoder-14B, Gemma 3 thinking,
etc.) but works for any case where you want to ban tokens.

## Why

Reasoning teachers were RL-trained to emit `<think>...</think>` blocks
before their answer. When such a teacher is plugged into a CTD recipe
two things go wrong:

1. **KL training**: at each student-sampled position the teacher's
   output distribution puts meaningful mass on `<think>` markers and
   on reasoning vocabulary. The student is pushed toward emitting
   them too — pure noise for downstream code/MBPP eval, and a
   structural style shift the student never wanted.
2. **Teacher generation** (SFT-on-teacher recipes, cache regen,
   reward models): the teacher emits literal `<think>` blocks in its
   completions. Strip-as-postprocessing works for eval but not for
   training data — the student trains on the prefix-context where
   `<think>` appeared, learning that "code is preceded by reasoning".

Banning the marker IDs at *generation* time and the same IDs at *KL
scoring* time fixes both — same blacklist, two application sites.

## API

`ctd.util.make_teacher_token_blacklist(tokenizer, names_csv, ids_csv) -> list[int]`

Resolves a comma-separated list of token strings (and/or raw IDs) into
teacher-vocab token IDs.

- **Path 1 — vocab lookup**: if a name is present as a single key in
  `tokenizer.get_vocab()`, that ID is added directly. This is the path
  that picks up dedicated special tokens like `<think>` in
  DeepSeek-R1-Distill / DeepCoder.
- **Path 2 — single-piece encode**: if a name is *not* in vocab, try
  `tokenizer.encode(name, add_special_tokens=False)`. Add the resulting
  ID **only if it tokenizes to exactly one piece**. Otherwise emit a
  `UserWarning` and skip — banning sub-pieces of a multi-piece
  encoding would over-restrict (e.g. encoding `<think>` on Qwen-Coder
  splits to `[<, think, >]`, banning `>` kills all comparison code).
- `ids_csv` adds raw IDs unconditionally — escape hatch when you
  *do* want to ban specific known IDs the resolver wouldn't otherwise
  catch.

`ctd.util.bad_words_ids_for_generate(blacklist) -> list[list[int]] | None`

Wraps a flat list of IDs into the nested format `model.generate(bad_words_ids=...)`
expects. Returns `None` for an empty blacklist (no-op).

## CLI flags

Three scripts take the same two flags:

| Flag | Type | Effect |
|---|---|---|
| `--mask-teacher-tokens` | csv strings | Resolve via `make_teacher_token_blacklist` |
| `--mask-teacher-token-ids` | csv ints | Add raw IDs (skip resolver) |

### `experiments/validation/06_train_grpo_kl_distill.py`

Application: teacher logits at scoring time get `index_fill(-1, mask_ids, -inf)`
*before* softmax. The masked tokens contribute zero probability to the
projected teacher distribution, and the rest renormalizes naturally.

### `experiments/validation/gen_teacher_completions.py`

Application: IDs are passed as `bad_words_ids` to `model.generate()`.
Hard ban — those tokens have probability 0 at every decoding step.
The teacher physically cannot emit them in its sampled completions.

### `experiments/validation/eval_teacher_chat.py`

Same as `gen_teacher_completions.py` — `bad_words_ids` to chat-template
generation. Useful when you want to FORCE no-think mode for a
reasoning teacher (rather than relying on the THINK_RE post-strip).

## Examples

### Train GRPO+KL Distill against DeepCoder-14B with thinking off

```bash
python -u 06_train_grpo_kl_distill.py --output-dir runs/onpolicy/M28_grpo_deepcoder_nothink \
    --student deepseek-ai/deepseek-coder-1.3b-instruct \
    --teacher agentica-org/DeepCoder-14B-Preview \
    --corpus data/mbpp_train_val_prompt.jsonl \
    --mapper-cache cache_mapper/deepcoder14b_to_dscoder1.3b_first_token.pt \
    --multi-token first_token \
    --K 4 --lambda-kl-ref 0.1 --lambda-distill 0.5 \
    --mask-teacher-tokens '<think>,</think>'
```

At launch you'll see:

```
[grpo-kl] masking 2 teacher token IDs from KL: [151648, 151649]
```

### Generate teacher completions with no thinking blocks

```bash
python -u gen_teacher_completions.py \
    --teacher agentica-org/DeepCoder-14B-Preview \
    --corpus data/mbpp_funcsig_prompts.jsonl \
    --output data/mbpp_funcsig_deepcoder14b_T07_nothink.jsonl \
    --max-new-tokens 256 --temperature 0.7 \
    --mask-teacher-tokens '<think>,</think>'
```

### Force no-think mode in teacher eval (vs post-strip)

```bash
python -u eval_teacher_chat.py \
    --model agentica-org/DeepCoder-14B-Preview \
    --output eval_results/deepcoder14b_nothink_he_mbpp.json \
    --he-limit 164 --mbpp-limit 378 \
    --batch-size 4 --max-new-tokens 1024 \
    --mask-teacher-tokens '<think>,</think>'
```

The default `eval_teacher_chat.py` keeps the post-hoc `THINK_RE` strip
for compatibility, but with the mask the model never emits the markers
in the first place — saves the tokens budget that would have been
spent on the reasoning trace.

### Use raw IDs when you know them

```bash
# Skip the string resolver; ban explicit teacher IDs
python -u 06_train_grpo_kl_distill.py ... \
    --mask-teacher-token-ids '151648,151649,151650'
```

### Combined string + ID list

```bash
# Block <think> markers AND a custom set of internal-reasoning IDs
python -u 06_train_grpo_kl_distill.py ... \
    --mask-teacher-tokens '<think>,</think>' \
    --mask-teacher-token-ids '151700,151701,151702'
```

## Behavior on tokenizers without dedicated think tokens

If you point this at a teacher whose tokenizer lacks a single-piece
`<think>` token (e.g. Qwen2.5-Coder-7B-Instruct, DS-Coder-6.7B-Instruct,
plain Qwen-base) the resolver returns `[]` and emits a warning:

```
UserWarning: make_teacher_token_blacklist: '<think>' is not a single
token in this tokenizer (encodes to 3 pieces: [13708, 766, 29]).
Skipping — banning sub-pieces would over-restrict generation.
If you really want it, pass --mask-teacher-token-ids with the resolved
IDs you want to ban.
```

Result: the run is unchanged (no masking). This is the desired no-op
behavior for non-thinking teachers — the same recipe works for both
without a code change.

## Verification

After launch, look for one of:

- `[grpo-kl] masking N teacher token IDs from KL: [...]` (trainer)
- `[gen] banning N teacher token IDs from generation: [...]` (gen)
- `[teacher-eval] banning N token IDs from generation: [...]` (eval)

If the line is missing despite passing the flag, the resolver returned
`[]` — check the warning above the launch banner.

## Programmatic use

```python
from ctd import make_teacher_token_blacklist, bad_words_ids_for_generate
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("agentica-org/DeepCoder-14B-Preview")
ids = make_teacher_token_blacklist(tok, "<think>,</think>")
# → [151648, 151649]

model = AutoModelForCausalLM.from_pretrained(...)
out = model.generate(
    input_ids=...,
    bad_words_ids=bad_words_ids_for_generate(ids),
)
```

Or apply directly to logits in a custom training loop:

```python
import torch
mask_t = torch.tensor(ids, dtype=torch.long, device="cuda")
logits = teacher(input_ids).logits  # [B, L, V_t]
logits = logits.index_fill(-1, mask_t, float("-inf"))
log_probs = torch.log_softmax(logits, dim=-1)  # masked tokens get -inf prob
```

## Why not just post-process?

For *eval* you can post-strip thinking blocks (`THINK_RE` regex) and
score the surviving code. That works for one-shot eval. It does NOT
work for:

- **Training data generation**: even if you strip after, the teacher
  spent its tokens budget on thinking — you get less code, often
  truncated. Banning the markers up front lets the budget go to the
  answer.
- **KL training signal**: there's no `THINK_RE` you can apply to a
  per-position log-probability tensor. Masking the marker IDs in the
  distribution is the only intervention that scales to dense
  per-token KL.
- **Reward modeling / RLHF**: any reward function that scores teacher
  text needs that text to be code, not chain-of-thought.

That's why the masking lives in `ctd.util` and is shared, not as a
recipe-local hack.

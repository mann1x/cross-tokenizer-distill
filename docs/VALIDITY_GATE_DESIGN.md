# Validity-gate design for on-policy KL trainer

Council audit `csl-2026-05-07-2207-0a17` finding: in `06_train_onpolicy_xv_v2.py:87`
`sample_continuation()`, when the student samples a non-code token (e.g. `<think>`,
`<|endoftext|>`, control characters, runaway repeats), the teacher's logits at that position
are **noise** — the teacher was never asked to predict those tokens, so its distribution there
is meaningless. Computing KL(student || teacher) on noise = aimless drift.

## Trigger pattern

This bites M39 (and any future on-policy KL recipe) more than SFT because:

- SFT loss is teacher-forced: positions are determined by the teacher corpus, never by student
  samples → noise positions never occur.
- On-policy KL loss is student-driven: the student's own chosen tokens determine where teacher
  logits get queried → garbage student samples → garbage teacher gradient.

This may explain a meaningful chunk of M39's catastrophic regression (HE -1.8 / MBPP -4.0 vs
QC-1.5B base).

## Implementation sketch

In the per-position KL loop (after `sample_continuation` returns student outputs):

```python
# Build a token-level "is this token plausibly code" mask.
# Cheap: blacklist of token ids known to be non-code.
NONCODE_TOKEN_IDS = build_noncode_blacklist(student_tok)
# blacklist content (per tokenizer):
#   - all special tokens (eos, pad, bos, unk, system tokens)
#   - <think>, </think>, <|im_start|>, <|im_end|>, channel tokens
#   - any token whose decoded string is pure whitespace > 4 chars or contains \x00..\x1f
#   - chat-template artifact tokens decoded by the tokenizer

valid_token_mask = ~torch.isin(sampled_token_ids, NONCODE_TOKEN_IDS)

# In KL accumulation:
kl_per_position = kl_div(student_logp, teacher_logp, reduction='none').sum(-1)
kl_loss = (kl_per_position * valid_token_mask.float()).sum() / valid_token_mask.float().sum().clamp(min=1)
```

## CLI

Add `--validity-gate` (default OFF; opt-in until A/B'd) to `06_train_onpolicy_xv_v2.py`.

Also log `valid_pos_share` per logging step so we can see how much gate-suppression actually
fires. If `valid_pos_share > 0.95` the gate is doing nothing useful; if `< 0.5` we've got a
sampling-quality crisis worth a separate fix.

## A/B plan (once cross-vocab on-policy KL is back in scope)

- M50: re-run M39 recipe with `--validity-gate`. Same student (QC-1.5B), same teacher
  (QC-14B-NF4), same corpus, same hyperparams. Single delta.
- Decision: if M50 ≥ M39 + 1pp on either HE or MBPP → ship gate as default ON. If flat → log
  it as a no-op for QC-1.5B and de-prioritize.

## Priority

LOW for the immediate path: cross-vocab SFT (M37/M37c) is the winning track and on-policy KL is
on the back burner. Implement opportunistically, A/B if/when on-policy KL is revived (e.g. as
the second stage of an M43-style 2-stage curriculum).


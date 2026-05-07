# Phase 3 design — mixed-loss recipes (post-Phase-2.5)

Council follow-up `csl-2026-05-07-2207-0a17` recommends mixed-loss as the next move if Path C
(Phase 2.5 code-only mask) fails to invert the QC-1.5B HE-drain. M40c first data point regressed
M40 by -2.4 HE / -2.1 MBPP, so the mixed-loss recipe is now likely on-deck.

## Mechanism

Anchor the student against its own raw-mode prior so distillation cannot drag it off-distribution
into the teacher's chat-template space.

```
L_total = α · L_anchor + (1 - α) · L_distill
```

- `L_distill`: existing chat-mode SFT teacher CE (unchanged from M40/M41).
- `L_anchor`: token-level CE against the **frozen student base** logits at the same positions
  (raw mode, NO chat template). Effectively distillation toward the unmodified pretrained model.
- `α`: anchor weight. Sweep `{0.1, 0.3, 0.5}`. Higher α = stronger raw-mode preservation, weaker
  distillation. α=0 reduces to current SFT; α=1 reduces to "do nothing." Council suggested
  starting at 0.3.

## Why this fits the observed regression

QC-1.5B in chat mode learns "be verbose, prefix code with `Here is the implementation:`". The
raw-mode prior is what HumanEval scoring (extract_code_chat → exec) actually rewards. By keeping
gradient pressure on the raw-mode logits we resist the chat drift while still absorbing the
teacher's *content*.

## Implementation sketch

In `06_train_sft_on_teacher.py`:

1. Add `--anchor-base PATH` + `--anchor-weight α`.
2. At init, load a frozen copy of the student base (no LoRA, eval mode):
   ```python
   anchor = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.bfloat16,
                                                 device_map={"": "cuda"})
   anchor.eval()
   for p_ in anchor.parameters(): p_.requires_grad = False
   ```
3. Build a parallel **raw-mode** input batch: re-tokenize the same `prompt + completion` pairs
   *without* `apply_chat_template`. Same loss mask (continuation slice).
4. In the training step:
   ```python
   # existing chat-mode forward (distill)
   s_out_chat = student(input_ids=input_ids, attention_mask=attention_mask)
   loss_distill = ce_at_mask(s_out_chat.logits, targets, slice_mask)

   # new raw-mode forward (anchor)
   with torch.no_grad():
       a_out = anchor(input_ids=raw_input_ids, attention_mask=raw_attn_mask)
   s_out_raw = student(input_ids=raw_input_ids, attention_mask=raw_attn_mask)
   loss_anchor = kl_at_mask(s_out_raw.logits, a_out.logits.detach(), raw_slice_mask)

   loss = args.anchor_weight * loss_anchor + (1 - args.anchor_weight) * loss_distill
   ```

VRAM cost: +student_base (~3 GB BF16) + 2× forward pass. On 48 GB Ada 6000 with bs=2/ga=8 this
fits without trouble; chat + raw forwards can share gradient checkpointing.

## Recipe matrix (Phase 3 mixed-loss)

| ID | Student | Data | α | Notes |
|---|---|---|---|---|
| **M45** | QC1.5B | mbpp_train (code-only)  | 0.3 | A/B vs M40c |
| **M46** | QC1.5B | funcsig    (code-only)  | 0.3 | A/B vs M41c (the QC-1.5B headline) |
| **M47** | QC1.5B | funcsig    (code-only)  | 0.5 | sweep — does heavier anchor help HE further |
| **M48** | QC1.5B | funcsig    (code-only)  | 0.1 | sweep — minimum viable anchor |
| **M49** | DSC1.3B | funcsig    (code-only)  | 0.3 | confirm anchor doesn't *hurt* the M37 winner |

## Decision rule

- If M46 (α=0.3) inverts QC-1.5B HE-drain (≥ baseline 63.4) AND keeps Phase 1+2 MBPP gains → α=0.3 is the recipe; do M47/M48 only if budget allows.
- If M46 still regresses HE → mixed-loss is not the bottleneck; investigate teacher chat-template
  itself (try a different system prompt, or distill from raw-mode teacher completions).
- M49 must NOT regress M37c. If it does, mixed-loss is bad for cross-vocab too and we need a
  per-track recipe.

## Compute budget

5 SFT runs × ~25-30 min each (chat+raw double forward ~1.7× slower than vanilla SFT) ≈ 2.5-3 hr
sequential. Eval ~5-7 min per recipe ≈ +30-40 min. Total Phase 3 ~3.5-4 hr.

## Out of scope for Phase 3

- KL-anchor variant (`L_anchor = KL(student_chat || student_raw)` instead of CE-vs-base): defer.
  Council called this out as a possibility but said CE-vs-base is the simpler first cut.
- Validity gate for on-policy KL (#113): orthogonal — only matters if we revive M39-class recipes.


# SFT-on-teacher style-shift collapse — postmortem

Documents the failure pattern across 6 cross-vocab SFT attempts
(M21, M21b, M21c, M22, M23, M25) so future experiments don't repeat
it. Distilled from the 2026-05-06 work block.

## TL;DR

**Causal-LM CE on a cross-vocab teacher's free-text generations
*always* shifts the student's output style toward the teacher's
style.** This shift is independent of:

- Corpus shape (MBPP-docstring-style vs HumanEval-function-sig-style)
- Corpus size (374 → 474 → 1845 → 2422 prompts)
- Completion cleanliness (raw vs aggressively stripped of fences /
  asserts / `__main__` blocks)
- Teacher mode (Instruct chat-aware vs base completion mode)
- Teacher max_new_tokens budget (128 / 192 / 256 / 768)
- Function signature shape rewrite (Fix A)

For our specific pairing — DS-Coder-1.3B-Instruct (terse, body-only
output) trained on Qwen2.5-Coder-7B-Instruct generations (verbose,
comment-heavy, often emits trailing test-runner code or markdown
fences) — the style shift breaks HE eval scoring even when the
training data has been visibly cleaned.

## What "style shift" means concretely

Even with TRAINING DATA stripped of obvious pollution, the student
post-SFT emits:

1. **Trailing `def test_*():` after the function body.** Student saw
   thousands of teacher generations where "after a function body,
   more code follows" (verbose explanation, asserts, test functions).
   Even with those stripped from the training corpus, the student's
   internal style adapts toward "code is followed by more code."
2. **Markdown fences after the body** (` ``` `). When ANY fraction
   of the training data has fences, student picks them up. Stripping
   them in data still leaves residual style shift if other related
   conventions remain.
3. **Over-indented continuations.** Student trained on Qwen's
   verbose nested logic emits 8-12 space indents instead of 4 →
   IndentationError at exec time.

For HE eval, the scorer is `exec(prompt + generation + tests)`.
Any of the above kills it: extra `def`s redefine the entry point
without matching signature; fences cause SyntaxError; bad indents
cause IndentationError.

## Why on-policy KL doesn't have this problem

Empirically: **M6b (on-policy KL with Qwen2.5-Coder-7B-Instruct
teacher, byte_anchor + first_token) holds at HE 53.0**, while every
SFT recipe with the SAME teacher collapses to 14-32%.

Mechanism: on-policy KL has the student generate its own
continuation at every step. The teacher is invoked only to compute
per-token logits over the student's text. **The student never sees
teacher's free generations**, so teacher's verbose conventions
cannot bleed into student's output style. The student keeps its
base generation behavior; only its per-token distribution is nudged
toward the teacher's at each position.

This is also why M16 (SFT on **same-vocab** DS-Coder-6.7B-Instruct
teacher) survives at HE 58.5: the teacher and student share style
conventions (both DeepSeek-family), so the style shift is small.

## Failure-mode evidence

### M22: smoke-passed-but-failed

Training corpus passed my eyeball smoke (HE-style → indented body,
MBPP-style → fresh module). Student post-train emitted module-level
code after function headers. **HE 14.0 / MBPP 38.6.** Cause:
synthetic HE-style prompts had Qwen completions ending in
`if __name__ == "__main__":\n    doctest.testmod()` past byte 200,
which my truncated peek didn't see.

### M25: distribution-clean-but-failed

Function-signature prompts (Fix A) + completions cleaned to 0%
fences, 0% asserts, 0% pass-stubs (verified by distribution check).
Student STILL emitted `def test_make_palindrome():\n    assert ...`
after the body. **HE 22.6 / MBPP 38.9.** Cause: style shift
operates beyond explicit pollution markers — the model's internal
representation of "what follows code" shifts even when the training
data ends abruptly mid-context.

### M23: clean MBPP-only

No synthetic prompts, no contamination, single-source corpus
(MBPP train+val+prompt = 474). Same teacher (Qwen-Inst). **HE 29.9.**
Confirms it's not the corpus *content* but the teacher *style* that
shifts the student.

## Prescription

If you want to use a cross-vocab teacher's signal, do NOT do SFT
on its text. Use one of:

- **On-policy KL** (M6b family): student samples, teacher scores.
  Ceiling ~53% (recipe-family) but stable.
- **Off-policy KL with cached teacher logits** (M26 family, in
  progress): student trains on teacher-tokenized completions but
  loss is KL between student-pred and cached-teacher-distribution
  at each position, NOT CE on teacher tokens. Preserves
  distribution signal without ingesting teacher's style.
- **Same-vocab SFT** (M16 family): if the teacher is same family
  (DS-Coder → DS-Coder), style match is good and SFT works (58.5).

If you MUST use cross-vocab SFT (e.g., teacher logits unavailable),
post-process the teacher generations to match student style:

- Truncate aggressively at the function body's natural end (not at
  marker patterns — at the indent-dedent boundary of the function)
- Filter out any completion that doesn't look like the student's
  base would produce
- Mix in 30-50% reference-solution data (no teacher style at all)
  to anchor the student

## Smoke-procedure update

Added to `docs/SMOKE_PROCEDURE.md`:

> **Style-shift detection (post-tiny-train check)**: after the
> tiny-train probe (smoke check #3), generate 5 completions on real
> HE prompts and compare mean length + structural features against
> base model outputs. If student's outputs differ structurally from
> base's (e.g., contains `def test_`, `assert ` after body, fences),
> the recipe has style shift and the full run will collapse —
> abort.

## Six runs, one lesson

| Run | Recipe variation | HE | Failure mode |
|---|---|---:|---|
| M21 | Qwen-Inst, mnt=128 | 32.3 | Module rewrite from docstring |
| M21b | Qwen-base, mnt=256 | 0.0 | `pass`-stub training |
| M21c | Qwen-base, mnt=768 | 0.6 | Same as M21b (natural EOS at 70 tok) |
| M22 | mix-corpus + Qwen-Inst | 14.0 | Synth-HE pollution + style shift |
| M23 | clean MBPP-only + Qwen-Inst | 29.9 | Pure style shift, no pollution |
| M25 | Fix A func-sig + cleaned | 22.6 | Style shift survives data cleaning |
| **M16** | same-vocab DS-Coder + SFT | **58.5** | (works because teacher style ≈ student style) |
| **M6b** | cross-vocab Qwen + on-policy KL | **53.0** | (works because student keeps own style) |

The only recipes that don't collapse are the ones that don't ingest
cross-vocab teacher style as training text. Lesson learned the hard
way.

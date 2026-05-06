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

## Off-policy KL on cache also has style shift (M26, 2026-05-06)

**Hypothesis tested**: off-policy KL on cached teacher logits would
side-step style shift by matching distributions instead of CE-fitting
teacher tokens. **Result: false.** M26 collapsed to **HE 3.7 / MBPP
0.8** — worst result in the entire CTD validation, worse than every
SFT recipe.

### Why off-policy KL still ingests style

The "KL doesn't ingest style" claim from M6b (53 % cross-vocab) only
holds because **M6b is on-policy**: the student samples its own
continuation, then teacher scores per-position logits. The student
never sees teacher text as input.

**Off-policy KL on cache (M26) is structurally different**: training
positions ARE teacher text (the cached completions). At every
teacher-text position, we ask the student to match the projected
teacher distribution. This means the student is conditioned on
teacher-text prefixes and trained to predict teacher-style
continuations — same exposure as SFT, just with KL instead of CE on
top.

### M26's stacked failure modes

Sample HE generation (post-train, no fences stripped):

```
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False

# Test the function with the provided test cases
assert has__close__elements([1.0, 2.0, 3.0], 0.5) == False
assert has__close__elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True
```

Two pathologies:

1. **Style shift (same as SFT)**: trailing markdown, `assert` lines
   after function body, verbose explanatory prose. Identical pattern
   to M21–M25 SFT collapses.
2. **NEW — projection-ambiguity identifier mangling**:
   `has__close__elements` (double underscore), `separate__paren__groups`
   etc. With 81 % of teacher tokens being multi-piece in student
   vocab, `first_token` projection chops them inconsistently
   (teacher's `_close` re-encodes to `_cl` + `ose` in student vocab,
   we KL-match against `_cl` only). Across many positions this
   teaches the student a corrupted naming convention that real code
   never uses.

### Implication for the prescription

The **Prescription** section above lists "Off-policy KL with cached
teacher logits" as a viable recipe. **M26 invalidates this for
cross-vocab.** Strike it from the menu. Updated viability matrix:

| Recipe | Cross-vocab? | Status |
|---|---|---|
| On-policy KL (M6b) | ✅ | Works (53 % ceiling, recipe-floor) |
| Same-vocab SFT (M16) | N/A — same vocab only | Works (58.5) |
| Off-policy KL on cache (M26) | ❌ | Collapsed (3.7 / 0.8) |
| GRPO+KL Distill (M27) — teacher-LL reward (NAIVE) | ❌ | Collapsed (18.3 / 33.6) — wrong reward, see below |
| GRPO+KL Distill (M28) — verified exec reward | ✅ planned | In flight — correct GRPO+ recipe per DeepCoder |

The only structural fix for off-policy cross-vocab would be:
- Multi-token-aware projection (sum log-probs over all sub-pieces of
  each teacher token, not just first_token), and
- Filter the cached corpus to exclude teacher-style chatter
  (markdown, asserts, test functions) before KL training,
- AND mix in same-vocab-style anchor data so the conditioning prefixes
  resemble student-base output, not teacher-style text.

Open question whether all three together would beat M6b. M27 sidesteps
this entirely by going back to on-policy sampling.

## GRPO+KL Distill with TEACHER-LL reward (M27, naive variant) — 2026-05-06

**Important framing**: this section documents an *implementation
mistake*, not a flaw in GRPO+KL Distill the recipe. The recipe (per
DeepCoder / standard GRPO+) uses **verified rewards** (exec-pass on
unit tests) — that's what makes it style-immune. M27 substituted
teacher-log-likelihood as reward because it's "free" (same forward
already needed for the KL term, no sandbox required). That
substitution alone broke style-shift immunity. M28 = same recipe with
correct verified-exec reward.

**Hypothesis tested with M27**: GRPO+ on-policy sampling + teacher-LL
as reward + KL anchor would side-step style shift because the student
samples its OWN text (like M6b on-policy KL works at 53 %). **Result:
false** — but only because of the wrong reward choice. M27 collapsed
to **HE 18.3 / MBPP 33.6** — worse than M6b's 53 % cross-vocab
ceiling, worse than base.

### Why on-policy didn't save us

M6b's on-policy KL works because the loss is **distribution-matching**
at every position — the student's own samples are the input, and the
teacher's per-token distribution (projected) is the target. The
student keeps its own style by construction; only its per-token
distribution is nudged toward teacher's at each position.

M27's policy gradient term is **categorically different**. The reward
function is teacher-log-likelihood of the sampled tokens. The policy
gradient explicitly upweights sampled trajectories the teacher
considers high-likelihood. **Teacher's high-likelihood completions
are teacher-style** — verbose, markdown-heavy, with the teacher's
typography conventions (Qwen2.5-Coder-7B uses Chinese fullwidth `（`,
`：`, mathematical Unicode `𝟏`, etc.). PG signal pulls student into
teacher's output mode just as effectively as SFT-on-teacher-text does.

### Diagnostic signature in the eval

Failure modes from `humaneval` + `mbpp` cache:

```
SyntaxError: invalid character '（' (U+FF08)
SyntaxError: invalid character '：' (U+FF1A)
SyntaxError: invalid character '𝟏' (U+1D7CF)
IndentationError: unindent does not match any outer indentation level
NameError: name 'is_allowed_specific_char' is not defined
SyntaxError: 'return' outside function
```

CJK punctuation + math Unicode are **teacher style markers** — Qwen
emits them in verbose explanatory prose. The student now does too.
NameError on undefined functions = same M21 docstring-rewrite-as-prose
pattern, just induced via GRPO reward instead of CE on teacher tokens.

### Healthy training metrics, broken outputs

The deceiving thing: M27's training metrics looked fine throughout
2 hours of training:

```
step=5/118  pg=-0.074 kl_ref=0.0016 distill=0.30 reward(mean/std)=-0.49/0.15
step=60/118 pg=-0.012 kl_ref=0.138  distill=0.50 reward(mean/std)=-0.72/0.26
step=110/118 pg=-0.049 kl_ref=0.235 distill=0.30 reward(mean/std)=-0.58/0.16
```

`pg` consistently negative (advantages working), `kl_ref` bounded
(< 0.25 throughout — anchor holding), `reward std` 0.10-0.33
(GRPO advantage actually discriminating). All textbook-healthy.

**The metrics measure adherence to the loss; they cannot measure
whether the loss is the right objective.** When reward = teacher LL on
mismatched-style teacher, healthy metrics = student style → teacher
style → exec eval breaks.

### Implication for the prescription

The viability matrix has now ruled out two more cross-vocab recipes:

- ❌ Off-policy KL on cache (M26) — student conditioned on teacher
  text → SFT-class style ingestion + projection mangling.
- ❌ GRPO with teacher-LL reward (M27) — on-policy sampling but
  reward function pulls student into teacher style.

**The pattern**: any recipe that uses teacher's distribution as
either (a) training input for the student or (b) reward signal will
inherit teacher's style — even if KL anchor and on-policy sampling
are layered on top.

**M6b survives because it uses teacher only as a per-position
log-prob source over the STUDENT's tokens, in a distribution-matching
loss.** No reward, no input conditioning — just dense distillation.
That's a structural property of M6b's loss, not something we can
graft onto richer recipes.

### What would actually work for cross-vocab + RL

If we want RL/GRPO benefits without style ingestion, the reward
signal must be **task-level, not teacher-level**:

- **Exec-based reward**: 1.0 if student generation passes the unit
  tests, 0.0 otherwise. Requires unit tests at training time
  (MBPP-train works; HE-train doesn't exist). Slow (sandbox per
  rollout) but style-agnostic.
- **Test-pass-rate reward** with multi-test suites (MBPP+/HE+).
- **Code-quality reward** (linter/type-check pass) — proxies
  correctness without sandbox.

None of these were what M27 implemented. M27 used teacher LL because
it's free (same forward as the KL term), but free is the wrong price
when the signal teaches the wrong style. **M28 implements the first
option** (verified exec reward) — same GRPO+KL framework, swap teacher
LL for `1.0 if exec(prompt+gen+test) passes else 0.0`. Style-agnostic
by construction. MBPP-train has `test_list` per problem so the
sandbox is plumbing-only.

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
| M26 | cross-vocab Qwen + off-policy KL on cache | **3.7** | (off-policy KL on teacher text = SFT-class style ingestion + identifier mangling) |
| M27 | cross-vocab Qwen + GRPO+KL Distill (teacher-LL reward) | **18.3** | (PG reward = "be teacher-like" → CJK punct + math Unicode in outputs; healthy training metrics, broken eval) |

The only recipes that don't collapse are the ones that don't ingest
cross-vocab teacher style as training text. Lesson learned the hard
way.

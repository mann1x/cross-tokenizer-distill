# CTD design — alignment math, projection semantics, loss derivations

## The fundamental problem

Knowledge distillation defines a loss between a teacher distribution
`p_T(y | x)` and a student distribution `p_S(y | x)`. Both `y` are
"the next token" — but if teacher and student have different
tokenizers, "the next token" means different things in each model.

Concretely, given a piece of text:

```
text:           "def hello_world():\n    "
teacher_tok:    [d, ef, ▁hello, _world, (), :, \n, ▁▁▁▁]
student_tok:    [def, ▁hell, o_, world, (, ), :, \n, ▁, ▁, ▁, ▁]
```

The teacher's logit at position 3 predicts "what comes after `def ef
▁hello _world`". The student's logit at position 3 predicts "what
comes after `def ▁hell o_ world`". Both prefixes represent the same
*string* up to byte offset 18, but the position indices differ, the
token IDs differ, and the vocabularies differ.

To distill correctly we have to solve **three** sub-problems
independently:

1. **Vocabulary projection** — map teacher's distribution over its
   vocab to a distribution over the student's vocab.
2. **Position alignment** — for each student position `j`, determine
   which teacher position `k` (or which teacher state) corresponds to
   the same byte prefix.
3. **Loss formulation** — define a meaningful divergence between the
   projected, aligned teacher distribution and the student's
   distribution.

## 1. Vocabulary projection

Build a sparse matrix `M ∈ R^(V_S × V_T)` where `M[s, t]` is the
weight with which teacher token `t`'s probability mass is transferred
onto student token `s`.

Construction algorithm (`ctd.mapper.VocabMapper`):

```
for t_id in range(V_T):
    s = teacher_tokenizer.decode([t_id])      # string form
    s_ids = student_tokenizer.encode(s, add_special_tokens=False)
    if len(s_ids) == 1:
        # Clean single-token match
        M[s_ids[0], t_id] = 1.0
    elif len(s_ids) > 1:
        # Multi-token: teacher's t_id corresponds to several
        # student tokens concatenated. Apply a strategy:
        if strategy == "strict":
            pass                               # drop this mass
        elif strategy == "distribute":
            for s_id in s_ids:
                M[s_id, t_id] += 1.0 / len(s_ids)
        elif strategy == "first_token":
            M[s_ids[0], t_id] = 1.0            # all mass to first sub-token
    else:
        # Empty decode (rare, special tokens)
        pass
```

Apply: `student_dist = M @ teacher_dist`. For top-K teacher logits
with indices `(t_idx, t_val)`, project as

```
sparse_student_dist[batch, pos] = sum over k of softmax(t_val)[k] * M[:, t_idx[k]]
```

`coverage_report()` returns the fraction of teacher's average mass
that survives projection — useful as a go/no-go signal before paying
for a full precompute.

## 2. Position alignment

We need a function `align(j) → (k, residual_bytes)` that, for each
student position `j`, returns either:

- The teacher position `k` such that `teacher_byte_offset[k] ==
  student_byte_offset[j]` (clean alignment, residual_bytes = 0), or
- The largest teacher position `k*` whose byte offset is `≤
  student_byte_offset[j]`, plus the residual byte string between them.

### Naive approach: byte-anchor

Only distill at student positions where a clean teacher boundary
exists. Build:

```
teacher_offsets = cumulative byte length of each teacher token
student_offsets = cumulative byte length of each student token
aligned_positions = {j : student_offsets[j] in teacher_offsets}
```

Use teacher's natural logit at the matched position. Skip
non-aligned positions (mask out of loss). Coverage is typically
50-70% for code corpora.

### Smart approach: student_offset + suffix re-encode

For non-aligned student positions, we still have the byte-prefix; we
just need teacher's prediction "what comes after byte `student_offsets[j]`".

```
for j in range(N_student):
    target_byte = student_offsets[j]
    if target_byte in teacher_offsets:
        teacher_logit_at_j = natural_teacher_logits[lookup(target_byte)]
    else:
        # Find largest teacher boundary k* below target_byte
        k_star = largest_teacher_pos_with_offset_lte(target_byte)
        suffix_bytes = text[teacher_offsets[k_star] : target_byte]
        suffix_ids = teacher_tokenizer.encode(suffix_bytes, add_special_tokens=False)
        # Reuse KV cache from k_star, run forward on suffix_ids
        logit = teacher.forward(suffix_ids, past_kv=cached[k_star]).logits[-1]
        teacher_logit_at_j = logit
```

KV cache hand-off requires keeping teacher's `past_key_values` at every
position, which is `O(N_teacher × hidden × layers)` memory. For
precompute, we do this in a streaming fashion (one example at a time,
discarding cache after) so memory stays bounded.

Coverage with smart approach: 100% of student positions get a
teacher-derived target. Compute cost: roughly `1.5-2×` a single full
teacher forward pass per example.

### Edge cases

- **Special tokens (BOS, EOS, system markers)** never align across
  tokenizers. Treat as alignment misses; either skip or feed the
  decoded string through suffix re-encode.
- **Multi-byte UTF-8 codepoints split across teacher tokens** — the
  byte-offset approach handles these naturally because we work in
  bytes, not codepoints.
- **Whitespace normalisation** — some tokenizers normalise leading
  whitespace differently (`▁` prefix, `Ġ` prefix). Decode/encode
  round-trip MUST preserve byte-exact strings; if it doesn't, the
  tokenizer pair is incompatible and we error out at `VocabMapper`
  construction time with a clear message.

## 3. Loss formulation

Once we have an aligned, projected teacher distribution `p_T_proj[j]`
and the student's distribution `p_S[j]` at every position `j`, the
loss options are:

### KL divergence (standard)

```
L_kl[j] = sum_v p_T_proj[j, v] * (log p_T_proj[j, v] - log_softmax(student_logits[j])[v])
```

Standard temperature scaling can be applied to teacher's logits
before projection.

### JS divergence (symmetric)

Useful when projection drops mass and the projected distribution
isn't a true probability distribution. JS handles partial mass
gracefully.

### Top-K only KL (memory-frugal)

Project only teacher's top-K (e.g. K=32) onto student vocab; compute
KL only on the projected support, renormalise both distributions
over that support. Same shape as our existing top-K cache.

### ULD-style sorted KL (vocab-agnostic fallback)

Sort both teacher and student logits descending, take top-K from each,
KL on the sorted ranks. Vocab-agnostic — no projection matrix needed.
Provided as a fallback when projection coverage is too low.

## Default recipe

For our headline use-case (Qwen3-Coder-Next → DeepSeek-Coder-V2-Lite):

- `multi_token = "distribute"` (recover ~85-90% mass)
- `alignment = "student_offset"` with `suffix_reencode = True`
- `loss = "kl"` with temperature 1.0, K = 32

## Open questions / TODOs

- Can we learn the projection matrix? Treat `M` as parameters of a
  small linear projection trained jointly with the student. Higher
  capacity, possibly better fidelity.
- Calibrate teacher temperature per-position based on alignment
  quality?
- For positions where projection coverage is low, downweight the
  loss contribution rather than drop entirely?

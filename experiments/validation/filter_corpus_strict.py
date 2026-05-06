"""Filter teacher-generated completions for language consistency + readability.

Inspired by DeepSeek-R1 §2.3.3 (filter for "language consistency, readability,
no language-mix"). Aggressive defaults aimed at killing the style-shift
failure modes documented in docs/STYLE_SHIFT_ISSUE.md (CJK punctuation, math
Unicode, markdown chatter, trailing test_*, doctest, etc.).

Input  : JSONL with {"prompt": str, "teacher_completion": str, ...}
Output : JSONL with same schema, only records that survive ALL filters.
         Completion field is rewritten to the cleaned version.

Design: each filter runs in sequence; record is kept only if all filters
either pass it or successfully rewrite it. Drop counts are reported so
you can see which filter is doing the work.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path


def _strip_markdown_fences(s):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", s, re.DOTALL)
    if m:
        s = m.group(1)
    s = re.split(r"```", s)[0]
    return s.rstrip(), None


def _truncate_at_function_end(s):
    """Keep code-shaped top-level lines (def/class/@/import) and indented bodies.
    Truncate at the first top-level line that is NOT one of those — that's
    where chatter, prints, asserts, or test runners typically start."""
    lines = s.split("\n")
    out = []
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            out.append(ln); continue
        if ln[:1] in (" ", "\t"):  # indented: part of a body
            out.append(ln); continue
        # Top-level (col 0): keep code structure, drop chatter
        if re.match(r"^(def|class|@|from\s|import\s)\b", ln):
            out.append(ln); continue
        # Anything else at col 0 = chatter / prints / asserts / module-level expr
        break
    return "\n".join(out).rstrip(), None


def _drop_if_test_helpers(s):
    if re.search(r"^\s+def\s+(test_|check_)", s, re.MULTILINE):
        return None, "nested-test-def"
    if re.search(r"\bdoctest\.testmod\b", s):
        return None, "doctest-runner"
    if re.search(r'^\s*if\s+__name__\s*==\s*[\'"]__main__[\'"]', s, re.MULTILINE):
        return None, "main-guard"
    return s, None


def _ascii_only(s):
    try:
        s.encode("ascii")
        return s, None
    except UnicodeEncodeError as e:
        return None, f"non-ascii@{e.start}"


def _length_cap(s, min_chars, max_chars):
    n = len(s)
    if n < min_chars: return None, f"too-short({n}<{min_chars})"
    if n > max_chars: return None, f"too-long({n}>{max_chars})"
    return s, None


def _has_def(s):
    if not re.search(r"^\s*def\s+\w+\s*\(", s, re.MULTILINE):
        return None, "no-def"
    return s, None


def _drop_assert_chains(s, max_asserts):
    n = len(re.findall(r"^\s*assert\s+", s, re.MULTILINE))
    if n > max_asserts:
        return None, f"too-many-asserts({n}>{max_asserts})"
    return s, None


def filter_one(s, min_chars, max_chars, max_asserts):
    pipeline = [
        ("md", lambda x: _strip_markdown_fences(x)),
        ("trunc", lambda x: _truncate_at_function_end(x)),
        ("test", lambda x: _drop_if_test_helpers(x)),
        ("ascii", lambda x: _ascii_only(x)),
        ("def", lambda x: _has_def(x)),
        ("asserts", lambda x: _drop_assert_chains(x, max_asserts)),
        ("len", lambda x: _length_cap(x, min_chars, max_chars)),
    ]
    cur = s
    for name, fn in pipeline:
        cur, drop = fn(cur)
        if cur is None:
            return None, f"{name}:{drop}"
    return cur, "kept"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--completion-field", default="teacher_completion")
    p.add_argument("--min-chars", type=int, default=30)
    p.add_argument("--max-chars", type=int, default=900)
    p.add_argument("--max-asserts", type=int, default=2)
    args = p.parse_args()

    in_p = Path(args.input)
    out_p = Path(args.output); out_p.parent.mkdir(parents=True, exist_ok=True)

    counts = {"total": 0, "kept": 0}
    drop_reasons = {}

    with open(in_p) as fi, open(out_p, "w") as fo:
        for line in fi:
            counts["total"] += 1
            rec = json.loads(line)
            comp = rec.get(args.completion_field, "")
            cleaned, status = filter_one(comp, args.min_chars, args.max_chars, args.max_asserts)
            if cleaned is None:
                drop_reasons[status] = drop_reasons.get(status, 0) + 1
                continue
            rec[args.completion_field] = cleaned
            fo.write(json.dumps(rec) + "\n")
            counts["kept"] += 1

    kept_pct = 100.0 * counts["kept"] / max(1, counts["total"])
    print(f"[filter] in={counts['total']} kept={counts['kept']} ({kept_pct:.1f}%)")
    print(f"[filter] dropped={counts['total'] - counts['kept']}")
    for reason, n in sorted(drop_reasons.items(), key=lambda x: -x[1])[:15]:
        print(f"  {n:5d}  {reason}")
    print(f"[filter] -> {out_p}")


if __name__ == "__main__":
    main()

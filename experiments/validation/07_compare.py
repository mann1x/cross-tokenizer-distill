"""Step 7 — Compare A/B/C results and emit decision report.

Reads results/{A,B,C}.json from step 6 and prints:
  - Pass@1 table per benchmark
  - Δ_B = B - A, Δ_C = C - A (gain from same-vocab vs CTD)
  - ratio = Δ_C / Δ_B (CTD efficiency)
  - GO / CAUTION / NO-GO recommendation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_run(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--A", default="results/A.json")
    parser.add_argument("--B", default="results/B.json")
    parser.add_argument("--C", default="results/C.json")
    parser.add_argument("--output", default="results/decision.md")
    args = parser.parse_args()

    A = load_run(args.A)
    B = load_run(args.B)
    C = load_run(args.C)

    benches = sorted(set(A) & set(B) & set(C) - {"adapter"})
    if not benches:
        raise SystemExit("No common benchmarks across A/B/C.")

    lines = []
    lines.append("# CTD validation — A/B/C decision report\n")
    lines.append("| Benchmark | A (no distill) | B (same-vocab) | C (CTD) | Δ_B | Δ_C | C/B ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    deltas_B = []
    deltas_C = []

    for bench in benches:
        a = A[bench]["pass@1"]
        b = B[bench]["pass@1"]
        c = C[bench]["pass@1"]
        dB = b - a
        dC = c - a
        ratio = (dC / dB) if abs(dB) > 1e-9 else float("nan")
        lines.append(
            f"| {bench} | {a:.1%} | {b:.1%} | {c:.1%} | "
            f"{dB:+.1%} | {dC:+.1%} | {ratio:.2f} |"
        )
        deltas_B.append(dB)
        deltas_C.append(dC)

    avg_dB = sum(deltas_B) / len(deltas_B)
    avg_dC = sum(deltas_C) / len(deltas_C)
    ratio = (avg_dC / avg_dB) if abs(avg_dB) > 1e-9 else float("nan")

    lines.append("")
    lines.append(f"**Averaged Δ_B = {avg_dB:+.1%}, Δ_C = {avg_dC:+.1%}, "
                 f"ratio = {ratio:.2f}**\n")

    lines.append("## Decision\n")
    if avg_dC <= 0:
        decision = "NO-GO — CTD destroys signal vs SFT-only baseline. Fall back to same-vocab teacher."
    elif ratio >= 0.8:
        decision = (
            "GO — CTD recovers ≥80% of same-vocab distill gain. "
            "Proceed with Mythic-RDT v6U using Qwen3-Coder-Next teacher via CTD."
        )
    elif ratio >= 0.5:
        decision = (
            "PARTIAL — CTD recovers 50-80% of same-vocab gain. Consider re-running C with "
            "different multi_token strategy or alignment mode before Mythic-RDT v6U."
        )
    else:
        decision = (
            "CAUTION — CTD recovers <50% of same-vocab gain. "
            "Investigate vocab-pair-specific issues (coverage, byte-roundtrip) before committing."
        )
    lines.append(decision)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[07-compare] Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

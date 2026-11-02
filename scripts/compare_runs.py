#!/usr/bin/env python
"""Paired comparison of evaluation runs that share the same samples (same dataset /
limit / seed). Reports Acc@0.5 difference with a paired bootstrap 95% CI, the number
of samples that flipped, and an exact McNemar p-value - much more sensitive than
comparing two independent accuracies.

    python scripts/compare_runs.py runs/p1_qwen3-vl-4b-instruct_dior_test runs/p2_dior_px_up1024
    python scripts/compare_runs.py --base runs/p1_qwen3-vl-4b-instruct_dior_test runs/p2_dior_px_*
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
from pathlib import Path

from _common import ROOT  # noqa: E402


def load_hits(run: Path):
    hits, ious = {}, {}
    for line in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            hits[r["id"]] = bool(r["hit"])
            ious[r["id"]] = float(r["iou"])
    return hits, ious


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test on discordant pairs (b: A right/B wrong, c: A wrong/B right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def bootstrap_ci(diffs, iters=5000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="*")
    p.add_argument("--base", default=None, help="baseline run; default = first positional run")
    p.add_argument("--iters", type=int, default=5000)
    args = p.parse_args()
    runs = []
    for r in args.runs:
        runs += [Path(x) for x in glob.glob(r)] if any(ch in r for ch in "*?[") else [Path(r)]
    runs = [r for r in runs if (r / "predictions.jsonl").exists()]
    base = Path(args.base) if args.base else runs[0]
    others = [r for r in runs if r.resolve() != base.resolve()]
    if not others:
        raise SystemExit("need at least two runs")

    bh, bi = load_hits(base)
    print(f"baseline: {base.name}  n={len(bh)}  Acc@0.5={sum(bh.values())/len(bh):.4f}")
    print(f"{'run':40s} {'n':>5s} {'acc':>7s} {'delta':>7s} {'95% CI':>18s} {'A>B':>5s} {'A<B':>5s} {'p(McNemar)':>11s} {'dIoU':>7s}")
    for r in others:
        oh, oi = load_hits(r)
        ids = sorted(set(bh) & set(oh))
        if not ids:
            print(f"{r.name:40s}  no shared sample ids")
            continue
        diffs = [float(oh[i]) - float(bh[i]) for i in ids]
        b = sum(1 for i in ids if bh[i] and not oh[i])   # baseline right, other wrong
        c = sum(1 for i in ids if oh[i] and not bh[i])   # other right, baseline wrong
        lo, hi = bootstrap_ci(diffs, args.iters)
        acc = sum(oh[i] for i in ids) / len(ids)
        delta = sum(diffs) / len(ids)
        diou = sum(oi[i] - bi[i] for i in ids) / len(ids)
        print(f"{r.name:40s} {len(ids):5d} {acc:7.4f} {delta:+7.4f} [{lo:+.4f}, {hi:+.4f}] {b:5d} {c:5d} {mcnemar_exact(b, c):11.4f} {diou:+7.4f}")
    print("\nA>B: baseline correct & run wrong;  A<B: run correct & baseline wrong;  CI excludes 0 => difference is reliable")


if __name__ == "__main__":
    main()

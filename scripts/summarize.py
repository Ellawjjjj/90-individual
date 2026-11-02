#!/usr/bin/env python
"""Aggregate metrics.json from many runs into one Markdown + CSV table.

    python scripts/summarize.py runs/*            (PowerShell: python scripts/summarize.py (Get-Item runs/*))
    python scripts/summarize.py --glob "runs/sweep_pixels_*" --out runs/summary_pixels
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

from _common import ROOT, load_json  # noqa: E402

COLS = ["run", "model", "adapter", "dataset", "split", "n", "acc@0.5", "acc@0.75", "mean_iou",
        "parse_fail_rate", "sec_per_sample", "input_hw", "coord_mode", "thinking", "prompt", "max_pixels",
        "min_pixels", "temperature", "max_new_tokens", "dtype", "tag"]


def row_of(run_dir: Path):
    m = load_json(run_dir / "metrics.json")
    a, s = m["args"], m["summary"]
    return {
        "run": run_dir.name, "model": m["model"], "adapter": m.get("adapter") or "",
        "dataset": a["dataset"], "split": a["split"], "n": s["n"],
        "acc@0.5": s["acc@0.5"], "acc@0.75": s["acc@0.75"], "mean_iou": s["mean_iou"],
        "parse_fail_rate": s["parse_fail_rate"], "sec_per_sample": s["sec_per_sample"],
        "input_hw": "x".join(str(v) for v in (m.get("model_input") or {}).get("median_hw", [])) or "",
        "coord_mode": m["coord_mode"], "thinking": m.get("thinking"), "prompt": a.get("prompt"),
        "max_pixels": a.get("max_pixels"), "min_pixels": a.get("min_pixels"),
        "temperature": a.get("temperature"), "max_new_tokens": m.get("max_new_tokens"),
        "dtype": a.get("dtype"), "tag": a.get("tag", ""),
    }


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return "" if v is None else str(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="*")
    p.add_argument("--glob", default=None)
    p.add_argument("--out", default="runs/summary")
    p.add_argument("--sort", default="run")
    args = p.parse_args()
    dirs = [Path(r) for r in args.runs]
    if args.glob:
        dirs += [Path(d) for d in glob.glob(args.glob)]
    dirs = [d for d in dirs if (d / "metrics.json").exists()]
    rows = sorted((row_of(d) for d in dirs), key=lambda r: (r.get(args.sort) is None, r.get(args.sort)))
    if not rows:
        raise SystemExit("no runs with metrics.json found")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out) + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    md = ["| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    for r in rows:
        md.append("| " + " | ".join(fmt(r[c]) for c in COLS) + " |")
    Path(str(out) + ".md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {out}.md and {out}.csv")


if __name__ == "__main__":
    main()

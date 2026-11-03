#!/usr/bin/env python
"""Build the reproduction table: model x RefCOCO split (Acc@0.5), plus the 8-split average,
next to the numbers reported in the Qwen3-VL technical report (Table 4, RefCOCO-avg).

    python tools/p0_table.py --out runs/table_p0_vs_paper.md
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPLITS = ["refcoco_val", "refcoco_testA", "refcoco_testB",
          "refcoco_plus_val", "refcoco_plus_testA", "refcoco_plus_testB",
          "refcocog_val", "refcocog_test"]
SHORT = ["RC val", "RC testA", "RC testB", "RC+ val", "RC+ testA", "RC+ testB", "RCg val", "RCg test"]

# RefCOCO-avg reported in the Qwen3-VL Technical Report (arXiv 2511.21631, Table 4)
PAPER_AVG = {
    "qwen3-vl-2b-instruct": 85.6, "qwen3-vl-2b-thinking": 84.8,
    "qwen3-vl-4b-instruct": 89.0, "qwen3-vl-4b-thinking": 88.2,
    "qwen3-vl-8b-instruct": 89.1, "qwen3-vl-8b-thinking": 88.2,
}
# per-split numbers for the Thinking checkpoints as measured in NVIDIA's EGM model cards
# (not from the Qwen report; shown for reference only)
EGM_SPLITS = {
    "qwen3-vl-4b-thinking": [90.0, 92.7, 85.6, 85.2, 89.5, 79.3, 87.0, 87.7],
    "qwen3-vl-8b-thinking": [91.0, 92.5, 86.6, 86.2, 91.2, 80.5, 87.8, 88.6],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="runs/p0*_*")
    p.add_argument("--out", default="runs/table_p0_vs_paper.md")
    args = p.parse_args()

    table = {}  # model -> split -> (acc, n)
    for d in glob.glob(args.glob):
        mp = Path(d) / "metrics.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        model = m["model"].split("/")[-1].lower()
        name = Path(d).name
        split = next((s for s in SPLITS if name.endswith("_" + s)), None)
        if split is None:
            continue
        table.setdefault(model, {})[split] = (100 * m["summary"]["acc@0.5"], m["summary"]["n"])

    lines = ["| model | " + " | ".join(SHORT) + " | avg (ours) | avg (paper) | Δ |", "|" + "---|" * (len(SHORT) + 4)]
    for model in sorted(table):
        row = table[model]
        cells, vals = [], []
        for s in SPLITS:
            if s in row:
                cells.append(f"{row[s][0]:.1f}")
                vals.append(row[s][0])
            else:
                cells.append("–")
        avg = sum(vals) / len(vals) if vals else float("nan")
        paper = PAPER_AVG.get(model)
        note = "" if len(vals) == len(SPLITS) else f" ({len(vals)}/8 splits)"
        lines.append(f"| {model} | " + " | ".join(cells) + f" | {avg:.1f}{note} | "
                     + (f"{paper:.1f}" if paper else "–") + " | "
                     + (f"{avg - paper:+.1f}" if paper and vals else "–") + " |")
    for model, vals in EGM_SPLITS.items():
        lines.append(f"| {model} (EGM card, ref.) | " + " | ".join(f"{v:.1f}" for v in vals)
                     + f" | {sum(vals)/len(vals):.1f} | {PAPER_AVG.get(model, float('nan')):.1f} | – |")
    n_note = sorted({n for row in table.values() for _, n in row.values()})
    lines.append("")
    lines.append(f"Acc@0.5 (%). Ours: random subsets of n={n_note} expressions per split (seed 0). "
                 "Paper: Qwen3-VL Technical Report Table 4 (full splits). "
                 "EGM rows: per-split numbers from NVIDIA EGM model cards for the Thinking checkpoints.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Static figures for the report (matplotlib, PNG @ 200 dpi) from metrics.json files.

    python tools/plot_curves.py            # writes docs/figures/fig_steps.png, fig_resolution.png, fig_buckets.png

Numbers are read from runs/*/metrics.json so the figures stay in sync with the runs.
Palette: validated 3-slot categorical (blue / orange / aqua); text in neutral ink.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "docs" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
})


def acc(run: str, key: str = "acc@0.5", bucket=None):
    p = RUNS / run / "metrics.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text(encoding="utf-8"))["summary"]
    if bucket:
        grp, name = bucket
        b = s["buckets"].get(grp, {}).get(name)
        return 100 * b["acc@0.5"] if b else None
    return 100 * s[key]


def line(ax, xs, ys, color, label, marker="o", dy=8, skip_first=False):
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not pts:
        return
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=6, label=label,
            markerfacecolor="white", markeredgewidth=2)
    for i, (x, y) in enumerate(pts):
        if skip_first and i == 0:
            continue
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, dy), ha="center",
                    va="bottom" if dy > 0 else "top", fontsize=8.5, color=INK)


def fig_steps():
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    # same 1000 samples, more epochs  vs  more data at 1 epoch
    line(ax, [125, 375], [acc("p3_data_1000_dior"), acc("p3_data_1000_ep3_dior")], BLUE, "1000 samples, 1 → 3 epochs",
         dy=-10, skip_first=True)
    line(ax, [38, 125, 375, 1125], [acc("p3_data_300_dior"), acc("p3_data_1000_dior"), acc("p3_data_3000_dior"),
                                   acc("p3_data_3000_ep3_dior")], ORANGE, "300 / 1000 / 3000 samples × 1 epoch, then 3000 × 3", marker="s")
    z = acc("p3_zeroshot_dior")
    if z is not None:
        ax.axhline(z, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(f"zero-shot {z:.1f}", (38, z), textcoords="offset points", xytext=(0, -12), fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([38, 125, 375, 1125])
    ax.set_xticklabels(["38", "125", "375", "1125"])
    ax.set_xlabel("optimisation steps (effective batch 8)")
    ax.set_ylabel("Acc@0.5 on DIOR-RSVG test (%)")
    ax.set_ylim(50, 84)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_steps.png", dpi=200)
    plt.close(fig)


def fig_resolution():
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    runs = ["p2_dior_px_down448", "p1_qwen3-vl-4b-instruct_dior_test", "p2_dior_px_up1024", "p2_dior_px_up1280"]
    xs = [0, 1, 2, 3]
    line(ax, xs, [acc(r) for r in runs], BLUE, "all targets", dy=9)
    line(ax, xs, [acc(r, bucket=("spatial_words", "with_spatial")) for r in runs], AQUA,
         "expressions with spatial words", marker="^", dy=-10)
    line(ax, xs, [acc(r, bucket=("gt_area", "small")) for r in runs], ORANGE, "small targets (<1% of image)",
         marker="s", dy=-10)
    ax.set_xticks(xs)
    ax.set_xticklabels(["448\u00b2\n196 tok", "800\u00b2 (native)\n625 tok", "1024\u00b2\n1024 tok", "1280\u00b2\n1600 tok"])
    ax.set_xlim(-0.35, 3.35)
    ax.set_xlabel("model input resolution (visual tokens)")
    ax.set_ylabel("Acc@0.5 on DIOR-RSVG test (%)")
    ax.set_ylim(28, 66)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_resolution.png", dpi=200)
    plt.close(fig)


def fig_buckets():
    """Zero-shot vs LLM-LoRA vs all-LoRA per bucket (grouped bars, 2px gaps)."""
    cfgs = [("zero-shot", "p3_zeroshot_dior", INK2), ("LoRA · LLM", "p3_lora_llm_dior", BLUE), ("LoRA · all", "p3_lora_all_dior", ORANGE)]
    buckets = [("small", ("gt_area", "small")), ("medium", ("gt_area", "medium")), ("large", ("gt_area", "large")),
               ("no spatial", ("spatial_words", "no_spatial")), ("with spatial", ("spatial_words", "with_spatial")),
               ("long expr.", ("expr_length", "long(>=8w)"))]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    w = 0.26
    for i, (label, run, color) in enumerate(cfgs):
        ys = [acc(run, bucket=b) or 0 for _, b in buckets]
        xs = [j + (i - 1) * (w + 0.02) for j in range(len(buckets))]
        bars = ax.bar(xs, ys, width=w, color=color, label=label, linewidth=0)
        for b, y in zip(bars, ys):
            ax.annotate(f"{y:.0f}", (b.get_x() + b.get_width() / 2, y), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels([n for n, _ in buckets])
    ax.set_ylabel("Acc@0.5 on DIOR-RSVG test (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_buckets.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_steps()
    fig_resolution()
    fig_buckets()
    for f in ("fig_steps.png", "fig_resolution.png", "fig_buckets.png"):
        print("wrote", OUT / f)


if __name__ == "__main__":
    main()

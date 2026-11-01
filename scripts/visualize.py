#!/usr/bin/env python
"""Render good / bad cases from an evaluation run and write a failure summary.

    python scripts/visualize.py --run runs/q3vl4b_refcoco_val --n-good 8 --n-bad 16

Writes into <run>/vis/:
    good_cases.jpg, bad_<category>.jpg, cases/<id>.jpg (individual), failure_summary.md
Optionally compare several runs on the same samples (same dataset/limit/seed):
    python scripts/visualize.py --run runs/A --compare runs/B runs/C --ids 12_0 57_0
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from _common import ROOT, load_json  # noqa: E402

from grounding.data import load_samples
from grounding.metrics import failure_category
from grounding.vis import draw_case, make_grid


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--compare", nargs="*", default=[], help="other run dirs evaluated on the same samples")
    p.add_argument("--ids", nargs="*", default=None, help="only these sample ids")
    p.add_argument("--n-good", type=int, default=8)
    p.add_argument("--n-bad", type=int, default=16)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--max-side", type=int, default=448)
    p.add_argument("--save-individual", action="store_true")
    return p.parse_args()


def load_run(run: Path):
    metrics = load_json(run / "metrics.json")
    preds = {}
    for line in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            preds[r["id"]] = r
    return metrics, preds


def reload_samples(metrics):
    a = metrics["args"]
    kw = dict(split=a["split"], limit=a["limit"], seed=a["seed"], image_field=a["image_field"],
              expr_field=a["expr_field"], bbox_field=a["bbox_field"], expr_policy=a["expr_policy"],
              id_field=a.get("id_field"))
    if a.get("bbox_format"):
        kw["bbox_format"] = a["bbox_format"]
    elif a["dataset"].startswith("hf:"):
        kw["bbox_format"] = "xywh"
    return {s.id: s for s in load_samples(a["dataset"], **kw)}


def caption_for(r, tag=""):
    pred = "none" if r.get("pred") is None else "[" + ", ".join(f"{v:.0f}" for v in r["pred"]) + "]"
    head = f"{tag} " if tag else ""
    return f"{head}IoU={r['iou']:.2f} [{r['category']}] \"{r['expression']}\" pred={pred}"


def main():
    args = parse_args()
    run = Path(args.run)
    vis = run / "vis"
    vis.mkdir(exist_ok=True)
    metrics, preds = load_run(run)
    samples = reload_samples(metrics)

    recs = list(preds.values())
    if args.ids:
        recs = [r for r in recs if r["id"] in set(args.ids)]
    for r in recs:
        r["category"] = failure_category(r)

    # --- comparison mode: one row per sample, one column per run
    if args.compare:
        others = [(Path(c).name, load_run(Path(c))[1]) for c in args.compare]
        tiles = []
        chosen = recs if args.ids else sorted(recs, key=lambda r: r["iou"])[: args.n_bad]
        for r in chosen:
            s = samples[r["id"]]
            img = s.load_image()
            tiles.append(draw_case(img, r["gt"], r.get("pred"), caption_for(r, run.name), args.max_side))
            for name, op in others:
                o = op.get(r["id"])
                if o is None:
                    continue
                o["category"] = failure_category(o)
                tiles.append(draw_case(img, o["gt"], o.get("pred"), caption_for(o, name), args.max_side))
        grid = make_grid(tiles, cols=1 + len(others))
        grid.save(vis / "compare.jpg", quality=90)
        print("wrote", vis / "compare.jpg")
        return

    # --- good cases
    good = sorted([r for r in recs if r["iou"] >= 0.5], key=lambda r: -r["iou"])[: args.n_good]
    tiles = []
    for r in good:
        img = samples[r["id"]].load_image()
        t = draw_case(img, r["gt"], r.get("pred"), caption_for(r), args.max_side)
        tiles.append(t)
        if args.save_individual:
            (vis / "cases").mkdir(exist_ok=True)
            t.save(vis / "cases" / f"good_{r['id']}.jpg", quality=90)
    make_grid(tiles, cols=args.cols).save(vis / "good_cases.jpg", quality=90)

    # --- bad cases grouped by automatic category
    by_cat = defaultdict(list)
    for r in recs:
        if r["iou"] < 0.5:
            by_cat[r["category"]].append(r)
    per_cat = max(1, args.n_bad // max(1, len(by_cat)))
    for cat, rs in by_cat.items():
        rs = sorted(rs, key=lambda r: r["iou"])[:per_cat]
        tiles = []
        for r in rs:
            img = samples[r["id"]].load_image()
            t = draw_case(img, r["gt"], r.get("pred"), caption_for(r), args.max_side)
            tiles.append(t)
            if args.save_individual:
                (vis / "cases").mkdir(exist_ok=True)
                t.save(vis / "cases" / f"bad_{cat}_{r['id']}.jpg", quality=90)
        make_grid(tiles, cols=args.cols).save(vis / f"bad_{cat}.jpg", quality=90)

    # --- summary markdown
    s = metrics["summary"]
    lines = [f"# {run.name}", "",
             f"model: `{metrics['model']}`  adapter: `{metrics.get('adapter')}`  coord_mode: `{metrics['coord_mode']}`", "",
             f"n={s['n']}  Acc@0.5={s['acc@0.5']:.4f}  Acc@0.75={s['acc@0.75']:.4f}  mIoU={s['mean_iou']:.4f}  "
             f"parse_fail={s['parse_fail_rate']:.4f}", "",
             "## Failure categories (IoU < 0.5)", "", "| category | n | share of failures |", "|---|---|---|"]
    n_fail = sum(len(v) for v in by_cat.values()) or 1
    for cat, rs in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| {cat} | {len(rs)} | {len(rs)/n_fail:.1%} |")
    lines += ["", "## Buckets", ""]
    for name, b in s["buckets"].items():
        lines += [f"### {name}", "", "| bucket | n | Acc@0.5 | mIoU |", "|---|---|---|---|"]
        for k, v in b.items():
            lines.append(f"| {k} | {v['n']} | {v['acc@0.5']:.3f} | {v['mean_iou']:.3f} |")
        lines.append("")
    lines += ["## Worst 10 (raw model output)", ""]
    for r in sorted(recs, key=lambda r: r["iou"])[:10]:
        raw = (r.get("raw_text") or "").replace("\n", " ")[:300]
        lines.append(f"- `{r['id']}` IoU={r['iou']:.2f} [{r['category']}] \"{r['expression']}\" -> `{raw}`")
    (vis / "failure_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", vis)


if __name__ == "__main__":
    main()

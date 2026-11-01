#!/usr/bin/env python
"""Dump a Hugging Face grounding dataset to the repo's JSONL format (images saved
as JPEG next to it). Converting once makes later evals / visualisation / LoRA
training independent of the HF cache and lets you check the boxes by eye.

RefCOCO family (lmms-lab uploads; check columns with scripts/inspect_dataset.py first):
    python tools/convert_hf_to_jsonl.py --dataset hf:lmms-lab/RefCOCO --split val ^
        --expr-field answer --bbox-field bbox --bbox-format xywh --out data/refcoco/val
ScreenSpot (GUI grounding; rootsautomation upload stores normalised [x1, y1, x2, y2]):
    python tools/convert_hf_to_jsonl.py --dataset hf:rootsautomation/ScreenSpot --split test ^
        --expr-field instruction --bbox-field bbox --bbox-format xyxy_norm --out data/screenspot/test

Writes <out>.jsonl and <out>_images/<row>.jpg. Use --limit for a subset (fixed seed)
and --preview N to also render N random samples with the GT box drawn so you can
verify the bbox format before running anything expensive.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grounding.data import load_hf  # noqa: E402
from grounding.vis import draw_case, make_grid  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="hf:<repo_id>[:<config>]")
    p.add_argument("--split", required=True)
    p.add_argument("--image-field", default="image")
    p.add_argument("--expr-field", default="answer")
    p.add_argument("--bbox-field", default="bbox")
    p.add_argument("--bbox-format", default="xywh")
    p.add_argument("--expr-policy", default="all", choices=["all", "first"])
    p.add_argument("--id-field", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output prefix, e.g. data/refcoco/val")
    p.add_argument("--preview", type=int, default=8)
    p.add_argument("--jpeg-quality", type=int, default=92)
    args = p.parse_args()

    spec = args.dataset[3:] if args.dataset.startswith("hf:") else args.dataset
    config = None
    if ":" in spec:
        spec, config = spec.split(":", 1)
    samples = load_hf(spec, args.split, image_field=args.image_field, expr_field=args.expr_field,
                      bbox_field=args.bbox_field, bbox_format=args.bbox_format, expr_policy=args.expr_policy,
                      limit=args.limit, seed=args.seed, config=config, id_field=args.id_field)
    out = Path(args.out)
    img_dir = Path(str(out) + "_images")
    img_dir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    saved = {}
    n = 0
    with open(str(out) + ".jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            row = s.meta.get("row")
            if row not in saved:
                path = img_dir / f"{row}.jpg"
                if not path.exists():
                    s.load_image().save(path, quality=args.jpeg_quality)
                saved[row] = path
            rec = {"id": s.id, "image": str(saved[row].relative_to(out.parent)).replace("\\", "/"),
                   "expression": s.expression, "bbox": [round(v, 2) for v in s.bbox_xyxy],
                   "bbox_format": "xyxy", "width": s.width, "height": s.height, "source_row": row}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} samples ({len(saved)} images) to {out}.jsonl")

    if args.preview:
        import random

        rng = random.Random(args.seed)
        pick = rng.sample(samples, min(args.preview, len(samples)))
        tiles = [draw_case(s.load_image(), s.bbox_xyxy, None, f"GT: {s.expression}", 384) for s in pick]
        prev = Path(str(out) + "_preview.jpg")
        make_grid(tiles, cols=4).save(prev, quality=90)
        print(f"preview with GT boxes: {prev}  <- open it and confirm the green boxes are correct")


if __name__ == "__main__":
    main()

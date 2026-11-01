#!/usr/bin/env python
"""Generate a tiny synthetic grounding dataset (coloured shapes on a plain background).

Use it to (a) dry-run the whole pipeline with `--model fake` and (b) as a sanity
benchmark where a working VLM should score close to 100% Acc@0.5.

    python tools/make_toy_dataset.py --n 40 --out data/toy
    python scripts/eval_grounding.py --model fake --dataset data/toy/toy.jsonl --out runs/dryrun
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

COLORS = {"red": (220, 40, 40), "green": (40, 170, 70), "blue": (40, 90, 220), "yellow": (240, 200, 30),
          "purple": (140, 60, 180), "orange": (240, 130, 30)}
SHAPES = ("circle", "square", "triangle")


def draw_shape(d: ImageDraw.ImageDraw, shape: str, box, color):
    x1, y1, x2, y2 = box
    if shape == "circle":
        d.ellipse(box, fill=color)
    elif shape == "square":
        d.rectangle(box, fill=color)
    else:
        d.polygon([(x1, y2), (x2, y2), ((x1 + x2) / 2, y1)], fill=color)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--out", default="data/toy")
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.n):
        W = H = args.size
        img = Image.new("RGB", (W, H), (245, 245, 240))
        d = ImageDraw.Draw(img)
        k = rng.randint(2, 4)
        used = []
        objs = []
        tries = 0
        while len(objs) < k and tries < 100:
            tries += 1
            s = rng.randint(50, 160)
            x1, y1 = rng.randint(10, W - s - 10), rng.randint(10, H - s - 10)
            box = [x1, y1, x1 + s, y1 + s]
            if any(not (box[2] < u[0] or box[0] > u[2] or box[3] < u[1] or box[1] > u[3]) for u in used):
                continue
            used.append(box)
            color = rng.choice(list(COLORS))
            shape = rng.choice(SHAPES)
            objs.append((color, shape, box))
        # expressions only for (color, shape) combos that are unique in the image
        combos = {}
        for c, s, b in objs:
            combos.setdefault((c, s), []).append(b)
        for c, s, b in objs:
            draw_shape(d, s, b, COLORS[c])
        name = f"{i:04d}.jpg"
        img.save(out / "images" / name, quality=92)
        for (c, s), boxes in combos.items():
            if len(boxes) == 1:
                rows.append({"id": f"{i}_{c}_{s}", "image": f"images/{name}", "expression": f"the {c} {s}",
                             "bbox": boxes[0], "bbox_format": "xyxy", "width": W, "height": H})
    with open(out / "toy.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} samples over {args.n} images to {out / 'toy.jsonl'}")


if __name__ == "__main__":
    main()

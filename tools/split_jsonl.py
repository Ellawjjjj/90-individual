#!/usr/bin/env python
"""Split a JSONL file into train/val(/test) by image (all expressions of one image
stay in the same split).

    python tools/split_jsonl.py data/my_photos/all.jsonl --ratios 0.7 0.3
    python tools/split_jsonl.py data/screenspot/test.jsonl --ratios 0.5 0.5 --names train test
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--ratios", type=float, nargs="+", default=[0.8, 0.2])
    p.add_argument("--names", nargs="+", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    names = args.names or ["train", "val", "test"][: len(args.ratios)]
    assert len(names) == len(args.ratios)

    path = Path(args.path)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_img = defaultdict(list)
    for r in rows:
        by_img[r["image"]].append(r)
    imgs = sorted(by_img)
    random.Random(args.seed).shuffle(imgs)
    total = sum(args.ratios)
    bounds, acc = [], 0.0
    for rt in args.ratios:
        acc += rt / total
        bounds.append(int(round(acc * len(imgs))))
    start = 0
    for name, end in zip(names, bounds):
        chunk = imgs[start:end]
        start = end
        out = path.with_name(f"{path.stem}_{name}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            n = 0
            for im in chunk:
                for r in by_img[im]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n += 1
        print(f"{out}: {len(chunk)} images, {n} samples")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build your own grounding dataset from LabelMe annotations.

Annotate photos with LabelMe (pip install labelme): draw a *rectangle* per target and
type the referring expression as the label (e.g. "the blue bicycle next to the door").
One image may contain several rectangles.

    python tools/convert_labelme.py --root data/my_photos --out data/my_photos/all.jsonl
    python tools/split_jsonl.py data/my_photos/all.jsonl --ratios 0.7 0.3   (optional)

Each LabelMe JSON sits next to its image (or set --image-dir). The description field
of a shape, if non-empty, overrides the label as the expression.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="directory with LabelMe *.json files")
    p.add_argument("--image-dir", default=None, help="images directory if not next to the JSONs")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for js in sorted(root.rglob("*.json")):
            try:
                d = json.loads(js.read_text(encoding="utf-8"))
            except Exception:
                continue
            if "shapes" not in d:
                continue
            img_name = d.get("imagePath") or (js.stem + ".jpg")
            img_path = (Path(args.image_dir) / Path(img_name).name) if args.image_dir else (js.parent / Path(img_name).name)
            w, h = d.get("imageWidth"), d.get("imageHeight")
            if (not w or not h) and img_path.exists():
                from PIL import Image

                with Image.open(img_path) as im:
                    w, h = im.size
            for k, sh in enumerate(d["shapes"]):
                if sh.get("shape_type", "rectangle") != "rectangle":
                    continue
                pts = sh["points"]
                xs = [pt[0] for pt in pts]
                ys = [pt[1] for pt in pts]
                expr = (sh.get("description") or "").strip() or (sh.get("label") or "").strip()
                if not expr:
                    continue
                rel = os.path.relpath(img_path.resolve(), out.parent.resolve()).replace("\\", "/")
                rec = {"id": f"{js.stem}_{k}", "image": rel, "expression": expr,
                       "bbox": [min(xs), min(ys), max(xs), max(ys)], "bbox_format": "xyxy",
                       "width": w, "height": h}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} samples to {out}")


if __name__ == "__main__":
    main()

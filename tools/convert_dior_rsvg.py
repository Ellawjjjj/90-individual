#!/usr/bin/env python
"""Convert DIOR-RSVG (remote-sensing referring expressions) to JSONL.

Expected layout (from the RSVG release, github.com/ZhanYang-nwpu/RSVG-pytorch):
    <root>/JPEGImages/*.jpg
    <root>/Annotations/*.xml      one XML per image, several <object> each with a
                                  <bndbox> and a <description> sentence
    <root>/train.txt, val.txt, test.txt   (optional) object indices per split

    python tools/convert_dior_rsvg.py --root data/DIOR_RSVG --out data/dior_rsvg

Writes data/dior_rsvg/{train,val,test}.jsonl with image paths pointing at JPEGImages.
If the split files are missing (or have a different meaning), a seeded 70/10/20 split
by image is produced instead and a warning is printed. Verify a preview before use.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grounding.vis import draw_case, make_grid  # noqa: E402

DESC_TAGS = ("description", "desc", "sentence", "caption", "text", "expression")


def parse_xml(path: Path):
    tree = ET.parse(path)
    r = tree.getroot()
    fname = (r.findtext("filename") or path.stem)
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        fname = fname + ".jpg"
    size = r.find("size")
    w = int(float(size.findtext("width"))) if size is not None else None
    h = int(float(size.findtext("height"))) if size is not None else None
    objs = []
    # NOTE: every <object> is appended (expression may be None) so that the running
    # index matches the official split files, which count *all* objects in order
    # (RSVG-pytorch data_loader.py: `for member in root.findall('object'): ... count += 1`).
    for o in r.findall("object"):
        bb = o.find("bndbox")
        box = None
        if bb is not None:
            try:
                box = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
            except (TypeError, ValueError):
                box = None
        elif len(o) > 2 and len(o[2]) >= 4:  # positional fallback used by the original loader
            try:
                box = [float(o[2][k].text) for k in range(4)]
            except (TypeError, ValueError):
                box = None
        desc = None
        for t in DESC_TAGS:
            desc = o.findtext(t)
            if desc and desc.strip():
                break
        if not (desc and desc.strip()) and len(o) > 3 and o[3].text and o[3].text.strip():
            desc = o[3].text  # original loader reads member[3].text
        objs.append({"name": o.findtext("name"), "bbox": box,
                     "expression": desc.strip() if desc and desc.strip() else None})
    return fname, w, h, objs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preview", type=int, default=8)
    args = p.parse_args()

    root = Path(args.root)
    ann_dir = root / "Annotations"
    img_dir = root / "JPEGImages"
    xmls = sorted(ann_dir.glob("*.xml"))
    if not xmls:
        raise SystemExit(f"no XML files under {ann_dir}")

    flat = []  # one entry per (image, object)
    for x in xmls:
        fname, w, h, objs = parse_xml(x)
        img_path = img_dir / fname
        if (w is None or h is None) and img_path.exists():
            from PIL import Image

            with Image.open(img_path) as im:
                w, h = im.size
        for j, o in enumerate(objs):
            flat.append({"id": f"{x.stem}_{j}", "image": img_path, "expression": o["expression"],
                         "bbox": o["bbox"], "width": w, "height": h, "category": o["name"], "xml": x.stem})
    n_valid = sum(1 for f in flat if f["expression"] and f["bbox"])
    print(f"parsed {len(flat)} objects ({n_valid} with expression+box) from {len(xmls)} annotation files")
    if n_valid == 0:
        o = ET.parse(xmls[0]).getroot().find("object")
        raise SystemExit("no expressions found; children of the first <object> are: "
                         + str([c.tag for c in o] if o is not None else None))

    def valid(rows):
        return [r for r in rows if r["expression"] and r["bbox"]]

    splits = {}
    have_split_files = all((root / f"{s}.txt").exists() for s in ("train", "val", "test"))
    if have_split_files:
        for s in ("train", "val", "test"):
            toks = (root / f"{s}.txt").read_text(encoding="utf-8").split()
            idx = []
            names = set()
            for t in toks:
                if t.isdigit():
                    idx.append(int(t))
                else:
                    names.add(Path(t).stem)
            if idx:
                bad = [i for i in idx if not (0 <= i < len(flat))]
                if bad:
                    print(f"WARNING: {len(bad)} indices in {s}.txt are out of range (max {len(flat)-1}); "
                          "the XML ordering may differ from the official one")
                splits[s] = valid([flat[i] for i in idx if 0 <= i < len(flat)])
            else:
                splits[s] = valid([f for f in flat if f["xml"] in names])
        print("split sizes from txt files:", {k: len(v) for k, v in splits.items()},
              "(official DIOR-RSVG: train 26991 / val 3829 / test 7500)")
        if sum(len(v) for v in splits.values()) == 0:
            have_split_files = False
    if not have_split_files:
        print("WARNING: no usable split files; making a seeded 70/10/20 split by image")
        imgs = sorted({f["xml"] for f in flat})
        rng = random.Random(args.seed)
        rng.shuffle(imgs)
        n = len(imgs)
        tr, va = set(imgs[: int(0.7 * n)]), set(imgs[int(0.7 * n): int(0.8 * n)])
        splits = {"train": valid([f for f in flat if f["xml"] in tr]),
                  "val": valid([f for f in flat if f["xml"] in va]),
                  "test": valid([f for f in flat if f["xml"] not in tr and f["xml"] not in va])}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for s, rows in splits.items():
        with open(out / f"{s}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                rec = dict(r)
                rec["image"] = os.path.relpath(Path(r["image"]).resolve(), out.resolve()).replace("\\", "/")
                rec["bbox_format"] = "xyxy"
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} -> {out / (s + '.jsonl')}")

    if args.preview and splits.get("train"):
        from PIL import Image

        rng = random.Random(args.seed)
        pick = rng.sample(splits["train"], min(args.preview, len(splits["train"])))
        tiles = [draw_case(Image.open(r["image"]).convert("RGB"), r["bbox"], None, f"GT: {r['expression']}", 384)
                 for r in pick]
        make_grid(tiles, cols=4).save(out / "preview.jpg", quality=90)
        print("preview:", out / "preview.jpg")


if __name__ == "__main__":
    main()

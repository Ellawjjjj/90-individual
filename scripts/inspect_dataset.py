#!/usr/bin/env python
"""Look at the columns and first rows of a Hugging Face dataset (or a jsonl) before
deciding the field mapping for eval / conversion.

    python scripts/inspect_dataset.py --dataset hf:lmms-lab/RefCOCO --split val --n 3
    python scripts/inspect_dataset.py --dataset data/refcoco/val.jsonl --n 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=3)
    args = p.parse_args()

    if args.dataset.startswith("hf:"):
        from datasets import get_dataset_split_names, load_dataset

        spec = args.dataset[3:]
        config = None
        if ":" in spec:
            spec, config = spec.split(":", 1)
        try:
            print("available splits:", get_dataset_split_names(spec, config))
        except Exception as e:  # noqa: BLE001
            print("could not list splits:", e)
        ds = load_dataset(spec, config, split=args.split) if config else load_dataset(spec, split=args.split)
        print("num rows:", len(ds))
        print("features:")
        for k, v in ds.features.items():
            print(f"  {k}: {v}")
        for i in range(min(args.n, len(ds))):
            row = ds[i]
            print(f"\n--- row {i} ---")
            for k, v in row.items():
                if hasattr(v, "size") and hasattr(v, "mode"):
                    print(f"  {k}: <PIL image {v.size[0]}x{v.size[1]} {v.mode}>")
                else:
                    s = json.dumps(v, ensure_ascii=False, default=str)
                    print(f"  {k}: {s[:300]}")
        print("\nHint: pass --image-field / --expr-field / --bbox-field / --bbox-format accordingly.\n"
              "If bbox values look like [x, y, w, h] in pixels use --bbox-format xywh; if they are in 0..1 use *_norm.")
    else:
        lines = Path(args.dataset).read_text(encoding="utf-8").splitlines()
        print("num rows:", len([l for l in lines if l.strip()]))
        for l in lines[: args.n]:
            print(json.dumps(json.loads(l), ensure_ascii=False)[:500])


if __name__ == "__main__":
    main()

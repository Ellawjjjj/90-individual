"""Dataset loading.

Two sources are supported:

1. JSONL (recommended; produced by tools/convert_*.py). One sample per line:
       {"id": "...", "image": "images/000123.jpg", "expression": "the red cup on the left",
        "bbox": [x1, y1, x2, y2], "bbox_format": "xyxy", "width": 640, "height": 480}
   `image` is relative to the JSONL file's directory (absolute paths also work).
   `bbox_format` may be omitted and given globally. width/height are optional
   (read from the image when missing).

2. Hugging Face datasets, e.g. `hf:lmms-lab/RefCOCO` with a split name. Field names
   are configurable because every community upload uses different column names.
   Run `scripts/inspect_dataset.py` first to look at the columns.

Both return a list of `Sample` objects with boxes in absolute xyxy pixels of the
original image.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

BBOX_FORMATS = ("xyxy", "xywh", "xyxy_norm", "xywh_norm", "cxcywh", "cxcywh_norm")


def to_xyxy(box: Sequence[float], fmt: str, w: int, h: int) -> List[float]:
    b = [float(v) for v in box]
    if fmt == "xyxy":
        x1, y1, x2, y2 = b
    elif fmt == "xywh":
        x1, y1, x2, y2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    elif fmt == "xyxy_norm":
        x1, y1, x2, y2 = b[0] * w, b[1] * h, b[2] * w, b[3] * h
    elif fmt == "xywh_norm":
        x1, y1, x2, y2 = b[0] * w, b[1] * h, (b[0] + b[2]) * w, (b[1] + b[3]) * h
    elif fmt == "cxcywh":
        x1, y1, x2, y2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    elif fmt == "cxcywh_norm":
        x1, y1 = (b[0] - b[2] / 2) * w, (b[1] - b[3] / 2) * h
        x2, y2 = (b[0] + b[2] / 2) * w, (b[1] + b[3] / 2) * h
    else:
        raise ValueError(f"unknown bbox format {fmt}; choose from {BBOX_FORMATS}")
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


@dataclass
class Sample:
    id: str
    expression: str
    bbox_xyxy: List[float]
    width: int
    height: int
    image_path: Optional[str] = None
    _loader: Optional[Callable[[], Any]] = None  # lazy PIL loader (HF datasets)
    meta: Dict[str, Any] = field(default_factory=dict)

    def load_image(self):
        from PIL import Image

        if self._loader is not None:
            img = self._loader()
        else:
            img = Image.open(self.image_path)
        if isinstance(img, dict) and "bytes" in img:  # undecoded HF Image feature
            from io import BytesIO

            img = Image.open(BytesIO(img["bytes"]))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img


# ----------------------------------------------------------------------------- JSONL


def load_jsonl(path: str, bbox_format: Optional[str] = None, limit: Optional[int] = None,
               seed: int = 0) -> List[Sample]:
    path = Path(path)
    root = path.parent
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = _subset(rows, limit, seed)
    samples: List[Sample] = []
    for i, r in enumerate(rows):
        img_path = r["image"]
        if not os.path.isabs(img_path):
            img_path = str(root / img_path)
        w, h = r.get("width"), r.get("height")
        if not w or not h:
            from PIL import Image

            with Image.open(img_path) as im:
                w, h = im.size
        fmt = r.get("bbox_format") or bbox_format or "xyxy"
        samples.append(
            Sample(
                id=str(r.get("id", i)),
                expression=str(r["expression"]).strip(),
                bbox_xyxy=to_xyxy(r["bbox"], fmt, w, h),
                width=int(w),
                height=int(h),
                image_path=img_path,
                meta={k: v for k, v in r.items() if k not in ("image", "expression", "bbox", "bbox_format")},
            )
        )
    return samples


# ----------------------------------------------------------------------------- HF


def _get_field(row: Dict, name: str):
    """Supports dotted access like 'meta.bbox'."""
    cur = row
    for part in name.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def load_hf(
    name: str,
    split: str,
    image_field: str = "image",
    expr_field: str = "answer",
    bbox_field: str = "bbox",
    bbox_format: str = "xywh",
    expr_policy: str = "all",  # if the field is a list: 'all' | 'first'
    limit: Optional[int] = None,
    seed: int = 0,
    config: Optional[str] = None,
    id_field: Optional[str] = None,
) -> List[Sample]:
    from datasets import load_dataset

    ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    idx = subset_indices(len(ds), limit, seed)
    samples: List[Sample] = []
    for i in idx:
        row = ds[i]
        img = _get_field(row, image_field)
        if img is None:
            raise KeyError(f"image field '{image_field}' not in row keys {list(row.keys())}")
        if isinstance(img, dict) and "bytes" in img:
            from io import BytesIO

            from PIL import Image

            img = Image.open(BytesIO(img["bytes"]))
        w, h = img.size
        loader = (lambda _i=i: _get_field(ds[_i], image_field))
        exprs = _get_field(row, expr_field)
        if exprs is None:
            raise KeyError(f"expression field '{expr_field}' not in row keys {list(row.keys())}")
        if isinstance(exprs, str):
            exprs = [exprs]
        elif expr_policy == "first":
            exprs = exprs[:1]
        bbox = _get_field(row, bbox_field)
        if bbox is None:
            raise KeyError(f"bbox field '{bbox_field}' not in row keys {list(row.keys())}")
        if isinstance(bbox, str):
            bbox = json.loads(bbox)
        if isinstance(bbox, (list, tuple)) and bbox and isinstance(bbox[0], (list, tuple)):
            bbox = bbox[0]  # some uploads wrap the box in a list
        xyxy = to_xyxy(bbox, bbox_format, w, h)
        base_id = str(_get_field(row, id_field)) if id_field else str(i)
        for j, e in enumerate(exprs):
            samples.append(
                Sample(
                    id=f"{base_id}_{j}" if len(exprs) > 1 else base_id,
                    expression=str(e).strip(),
                    bbox_xyxy=xyxy,
                    width=int(w),
                    height=int(h),
                    image_path=None,
                    _loader=loader,
                    meta={"row": i},
                )
            )
    return samples


# ----------------------------------------------------------------------------- generic


def subset_indices(n: int, limit: Optional[int], seed: int) -> List[int]:
    """Deterministic subset that *nests*: limit=200 is a subset of limit=500 for the
    same seed (prefix of one seeded permutation), so runs with different sizes can be
    compared on their shared samples."""
    idx = list(range(n))
    if limit is None or limit >= n:
        return idx
    rng = random.Random(seed)
    rng.shuffle(idx)
    return sorted(idx[:limit])


def _subset(rows: List, limit: Optional[int], seed: int) -> List:
    return [rows[i] for i in subset_indices(len(rows), limit, seed)]


def load_samples(dataset: str, **kw) -> List[Sample]:
    """`dataset` is either a path ending in .jsonl or 'hf:<repo_id>[:<config>]'."""
    if dataset.startswith("hf:"):
        spec = dataset[3:]
        config = None
        if ":" in spec:
            spec, config = spec.split(":", 1)
        hf_kw = {k: v for k, v in kw.items() if k in (
            "split", "image_field", "expr_field", "bbox_field", "bbox_format", "expr_policy",
            "limit", "seed", "id_field")}
        return load_hf(spec, config=config, **hf_kw)
    jl_kw = {k: v for k, v in kw.items() if k in ("bbox_format", "limit", "seed")}
    return load_jsonl(dataset, **jl_kw)


def describe(samples: Sequence[Sample], n: int = 3) -> str:
    lines = [f"{len(samples)} samples"]
    for s in list(samples)[:n]:
        lines.append(f"  id={s.id} size={s.width}x{s.height} bbox={[round(v,1) for v in s.bbox_xyxy]} expr={s.expression!r}")
    return "\n".join(lines)

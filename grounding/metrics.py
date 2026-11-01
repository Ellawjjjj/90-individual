"""IoU, Acc@IoU metrics and failure-mode bucketing (torch-free)."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

SPATIAL_WORDS = {
    "left", "right", "top", "bottom", "above", "below", "behind", "front", "next",
    "near", "beside", "between", "middle", "center", "centre", "closest", "farthest",
    "nearest", "back", "corner", "under", "over", "inside", "outside", "second",
    "third", "first", "last", "upper", "lower", "leftmost", "rightmost", "far",
    "closer", "further", "end", "side", "edge",
}
_WORD = re.compile(r"[a-zA-Z']+")


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def area_bucket(box: Sequence[float], w: int, h: int) -> str:
    """small (<1% of image), medium (1-10%), large (>10%)."""
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    ratio = area / float(max(1, w * h))
    if ratio < 0.01:
        return "small"
    if ratio < 0.10:
        return "medium"
    return "large"


def length_bucket(expr: str) -> str:
    n = len(_WORD.findall(expr or ""))
    if n <= 3:
        return "short(<=3w)"
    if n <= 7:
        return "medium(4-7w)"
    return "long(>=8w)"


def has_spatial(expr: str) -> bool:
    words = {w.lower() for w in _WORD.findall(expr or "")}
    return bool(words & SPATIAL_WORDS)


def _rate(xs: Iterable[float]) -> Optional[float]:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else None


def summarize(records: List[Dict]) -> Dict:
    """records: dicts with keys
        iou (float), parsed (bool), expression (str), gt (xyxy), width, height,
        optional latency_s (float), n_boxes (int)
    """
    n = len(records)
    ious = [r["iou"] for r in records]
    parsed = [bool(r.get("parsed", r.get("iou", 0) > 0)) for r in records]
    out = {
        "n": n,
        "acc@0.5": _rate(i >= 0.5 for i in ious),
        "acc@0.75": _rate(i >= 0.75 for i in ious),
        "mean_iou": _rate(ious),
        "parse_fail_rate": _rate(not p for p in parsed),
        "mean_iou_parsed_only": _rate(r["iou"] for r in records if r.get("parsed", True)),
        "multi_box_rate": _rate((r.get("n_boxes") or 0) > 1 for r in records),
        "sec_per_sample": _rate(r["latency_s"] for r in records if r.get("latency_s") is not None),
    }
    buckets: Dict[str, Dict[str, List[float]]] = {
        "gt_area": defaultdict(list),
        "expr_length": defaultdict(list),
        "spatial_words": defaultdict(list),
    }
    for r in records:
        gt = r.get("gt")
        if gt and r.get("width") and r.get("height"):
            buckets["gt_area"][area_bucket(gt, r["width"], r["height"])].append(r["iou"])
        buckets["expr_length"][length_bucket(r.get("expression", ""))].append(r["iou"])
        buckets["spatial_words"]["with_spatial" if has_spatial(r.get("expression", "")) else "no_spatial"].append(r["iou"])
    out["buckets"] = {
        name: {
            k: {"n": len(v), "acc@0.5": _rate(i >= 0.5 for i in v), "mean_iou": _rate(v)}
            for k, v in sorted(b.items())
        }
        for name, b in buckets.items()
    }
    return out


def failure_category(rec: Dict) -> str:
    """Coarse automatic label for a bad case; refined manually in the report."""
    if not rec.get("parsed", True):
        return "parse_failure"
    iou = rec["iou"]
    if iou >= 0.5:
        return "correct"
    gt = rec.get("gt")
    pred = rec.get("pred")
    if gt and pred:
        if iou == 0.0:
            return "wrong_object"
        if 0.2 <= iou < 0.5:
            return "loose_localisation"
        return "partial_overlap"
    return "unknown"

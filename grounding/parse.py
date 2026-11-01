"""Parse bounding boxes out of free-form VLM text and convert coordinate systems.

Coordinate modes (what the *model* is assumed to output):
    relative_1000     - [x1, y1, x2, y2] normalised to 0..1000 w.r.t. the image
                        (Qwen2-VL, Qwen3-VL, Qwen3.5 style)
    absolute_input    - absolute pixels w.r.t. the image *after* the processor's
                        smart-resize (Qwen2.5-VL style)
    absolute_original - absolute pixels w.r.t. the original image

Because conventions differ between model generations and are easy to get wrong,
`eval_grounding.py` can calibrate the mode automatically on a few samples.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Sequence, Tuple

COORD_MODES: Tuple[str, ...] = ("relative_1000", "absolute_input", "absolute_original")

_THINK_END = "</think>"
_NUM = r"-?\d+(?:\.\d+)?"
_BOX4 = re.compile(
    r"\[\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*\]"
)
# Qwen2-VL style "<|box_start|>(x1,y1),(x2,y2)<|box_end|>" or plain "(x1,y1),(x2,y2)"
_PAIR = re.compile(
    r"\(\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*\)\s*,?\s*\(\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*\)"
)
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)
_BOX_KEYS = ("bbox_2d", "bbox", "box", "bounding_box", "box_2d")


def strip_thinking(text: str) -> str:
    """Return the part of the output after the last </think> tag (if any)."""
    if _THINK_END in text:
        return text.rsplit(_THINK_END, 1)[1]
    return text


def _is_box(x) -> bool:
    return (
        isinstance(x, (list, tuple))
        and len(x) == 4
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in x)
    )


def _walk(obj) -> List[List[float]]:
    out: List[List[float]] = []
    if isinstance(obj, dict):
        for k in _BOX_KEYS:
            if k in obj and _is_box(obj[k]):
                out.append([float(v) for v in obj[k]])
        for k, v in obj.items():
            if k in _BOX_KEYS:
                continue
            if isinstance(v, (dict, list)):
                out.extend(_walk(v))
    elif isinstance(obj, list):
        if _is_box(obj):
            out.append([float(v) for v in obj])
        else:
            for v in obj:
                out.extend(_walk(v))
    return out


def _from_json(text: str) -> List[List[float]]:
    candidates = _FENCE.findall(text)
    candidates.append(text.strip())
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        boxes = _walk(obj)
        if boxes:
            return boxes
    return []


def extract_boxes(text: str, strip_think: bool = True) -> List[List[float]]:
    """Extract every 4-number box found in the model output, in order of appearance.

    Strategy: JSON (fenced or raw, any nesting) -> "[a, b, c, d]" regex -> "(a,b),(c,d)" regex.
    Returns raw numbers in whatever coordinate convention the model used.
    """
    if text is None:
        return []
    if strip_think:
        text = strip_thinking(text)
    boxes = _from_json(text)
    if boxes:
        return boxes
    boxes = [[float(v) for v in m.groups()] for m in _BOX4.finditer(text)]
    if boxes:
        return boxes
    return [[float(v) for v in m.groups()] for m in _PAIR.finditer(text)]


def first_box(text: str) -> Optional[List[float]]:
    boxes = extract_boxes(text)
    return boxes[0] if boxes else None


def to_original_xyxy(
    box: Sequence[float],
    mode: str,
    orig_w: int,
    orig_h: int,
    input_w: Optional[int] = None,
    input_h: Optional[int] = None,
    clip: bool = True,
) -> List[float]:
    """Map a raw model box to absolute xyxy pixels in the original image."""
    if mode == "relative_1000":
        sx, sy = orig_w / 1000.0, orig_h / 1000.0
    elif mode == "absolute_input":
        if not input_w or not input_h:
            raise ValueError("absolute_input mode needs input_w/input_h")
        sx, sy = orig_w / float(input_w), orig_h / float(input_h)
    elif mode == "absolute_original":
        sx = sy = 1.0
    else:
        raise ValueError(f"unknown coord mode: {mode}")
    x1, y1, x2, y2 = box
    xa, xb = sorted((x1 * sx, x2 * sx))
    ya, yb = sorted((y1 * sy, y2 * sy))
    if clip:
        xa, xb = max(0.0, min(xa, orig_w)), max(0.0, min(xb, orig_w))
        ya, yb = max(0.0, min(ya, orig_h)), max(0.0, min(yb, orig_h))
    return [xa, ya, xb, yb]


def from_original_xyxy(
    box: Sequence[float],
    mode: str,
    orig_w: int,
    orig_h: int,
    input_w: Optional[int] = None,
    input_h: Optional[int] = None,
) -> List[int]:
    """Inverse of `to_original_xyxy`: produce the integer box a model should emit
    (used to build fine-tuning targets)."""
    if mode == "relative_1000":
        sx, sy = 1000.0 / orig_w, 1000.0 / orig_h
        hi_x, hi_y = 1000, 1000
    elif mode == "absolute_input":
        if not input_w or not input_h:
            raise ValueError("absolute_input mode needs input_w/input_h")
        sx, sy = input_w / float(orig_w), input_h / float(orig_h)
        hi_x, hi_y = input_w, input_h
    elif mode == "absolute_original":
        sx = sy = 1.0
        hi_x, hi_y = orig_w, orig_h
    else:
        raise ValueError(f"unknown coord mode: {mode}")
    x1, y1, x2, y2 = box
    out = [
        int(round(max(0.0, min(x1 * sx, hi_x)))),
        int(round(max(0.0, min(y1 * sy, hi_y)))),
        int(round(max(0.0, min(x2 * sx, hi_x)))),
        int(round(max(0.0, min(y2 * sy, hi_y)))),
    ]
    return out

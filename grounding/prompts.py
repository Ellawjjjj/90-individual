"""Prompt templates and fine-tuning target format.

The same template name is used for evaluation and LoRA training so that the
model sees a consistent instruction / answer format.
"""

from __future__ import annotations

import json
from typing import Sequence

PROMPTS = {
    # Qwen3-VL cookbook style grounding instruction
    "json": 'Locate the "{expr}" in the image and output its bbox coordinates in JSON format.',
    # stricter: one object, one key
    "json_strict": (
        'Locate "{expr}" in the image. Output only a JSON list containing one object '
        'with the key "bbox_2d" whose value is [x1, y1, x2, y2].'
    ),
    # plain list, no JSON
    "plain": 'Output the bounding box of "{expr}" in the image as [x1, y1, x2, y2].',
    # lmms-eval / RefCOCO-style phrasing
    "refcoco": "Please provide the bounding box coordinate of the region this sentence describes: {expr}",
    # chain-of-thought nudge for instruct models (not needed for *-Thinking)
    "cot": (
        'First describe where "{expr}" is in the image in one sentence, then output its '
        'bbox coordinates in JSON format with the key "bbox_2d".'
    ),
}


def build_prompt(name: str, expr: str) -> str:
    if name not in PROMPTS:
        raise KeyError(f"unknown prompt template '{name}', choose from {list(PROMPTS)}")
    return PROMPTS[name].format(expr=expr.strip())


def build_target(expr: str, box: Sequence[int], style: str = "json") -> str:
    """Assistant answer used as the supervised target."""
    x1, y1, x2, y2 = [int(v) for v in box]
    if style in ("json", "json_strict", "refcoco", "cot"):
        obj = [{"bbox_2d": [x1, y1, x2, y2], "label": expr.strip()}]
        return "```json\n" + json.dumps(obj, ensure_ascii=False) + "\n```"
    if style == "plain":
        return f"[{x1}, {y1}, {x2}, {y2}]"
    raise KeyError(style)

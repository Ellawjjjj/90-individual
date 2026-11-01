"""Drawing helpers for qualitative analysis (PIL only)."""

from __future__ import annotations

import textwrap
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

GT_COLOR = (46, 204, 113)     # green
PRED_COLOR = (231, 76, 60)    # red
TEXT_BG = (0, 0, 0)


def _font(size: int = 14):
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_case(image: Image.Image, gt: Optional[Sequence[float]], pred: Optional[Sequence[float]],
              caption: str, max_side: int = 512) -> Image.Image:
    img = image.copy()
    scale = min(1.0, max_side / float(max(img.size)))
    if scale < 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BILINEAR)
    d = ImageDraw.Draw(img)
    w = max(2, int(3 * scale))
    if gt:
        d.rectangle([v * scale for v in gt], outline=GT_COLOR, width=w)
    if pred:
        d.rectangle([v * scale for v in pred], outline=PRED_COLOR, width=w)
    # caption strip
    font = _font(14)
    lines = textwrap.wrap(caption, width=max(20, img.width // 8))
    strip_h = 6 + 17 * len(lines)
    canvas = Image.new("RGB", (img.width, img.height + strip_h), TEXT_BG)
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    y = img.height + 3
    for ln in lines:
        d.text((4, y), ln, fill=(255, 255, 255), font=font)
        y += 17
    return canvas


def make_grid(tiles: List[Image.Image], cols: int = 4, pad: int = 6, bg=(255, 255, 255)) -> Image.Image:
    if not tiles:
        return Image.new("RGB", (10, 10), bg)
    tw = max(t.width for t in tiles)
    th = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    grid = Image.new("RGB", (cols * tw + (cols + 1) * pad, rows * th + (rows + 1) * pad), bg)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        grid.paste(t, (pad + c * (tw + pad), pad + r * (th + pad)))
    return grid

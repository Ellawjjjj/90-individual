#!/usr/bin/env python
"""Smoke test: ground one expression on one image, print the raw output under every
coordinate convention and save a visualisation. Run this first after installing.

    python scripts/demo_single.py --model Qwen/Qwen3-VL-4B-Instruct ^
        --image data/sample.jpg --expr "the person on the left" --out runs/demo.jpg
"""

from __future__ import annotations

import argparse

from _common import ROOT, add_model_args  # noqa: E402

from grounding.parse import COORD_MODES, extract_boxes, to_original_xyxy
from grounding.prompts import build_prompt
from grounding.vis import draw_case


def main():
    p = argparse.ArgumentParser()
    add_model_args(p)
    p.add_argument("--image", required=True)
    p.add_argument("--expr", required=True)
    p.add_argument("--coord-mode", default=None, help="draw this mode (default: family default)")
    p.add_argument("--out", default="runs/demo.jpg")
    args = p.parse_args()

    from PIL import Image

    from grounding.models import GenConfig, default_coord_mode, make_grounder

    is_thinking = "thinking" in args.model.lower() or args.enable_thinking == "on"
    g, family = make_grounder(
        args.model, dtype="bf16" if args.dtype == "4bit" else args.dtype, load_4bit=args.dtype == "4bit",
        adapter=args.adapter, merge_adapter=args.merge_adapter, attn_impl=args.attn,
        min_pixels=args.min_pixels, max_pixels=args.max_pixels, trust_remote_code=args.trust_remote_code)
    cfg = GenConfig(max_new_tokens=args.max_new_tokens or (4096 if is_thinking else 128),
                    temperature=args.temperature, top_p=args.top_p,
                    enable_thinking=None if args.enable_thinking == "auto" else args.enable_thinking == "on")
    img = Image.open(args.image).convert("RGB")
    res = g.generate([img], [build_prompt(args.prompt, args.expr)], cfg)[0]
    print("=== raw output ===")
    print(res.text)
    print(f"=== original size (W,H)={img.size}  model input (H,W)={res.input_hw}  patch={g.patch}  "
          f"new_tokens={res.n_new_tokens}  latency={res.latency_s:.2f}s")
    boxes = extract_boxes(res.text)
    print("parsed boxes (raw numbers):", boxes)
    if not boxes:
        return
    for mode in COORD_MODES:
        try:
            print(f"  {mode:18s} -> {[round(v, 1) for v in to_original_xyxy(boxes[0], mode, img.width, img.height, res.input_hw[1], res.input_hw[0])]}")
        except ValueError as e:
            print(f"  {mode:18s} -> {e}")
    mode = args.coord_mode or default_coord_mode(family)
    pred = to_original_xyxy(boxes[0], mode, img.width, img.height, res.input_hw[1], res.input_hw[0])
    draw_case(img, None, pred, f"[{mode}] \"{args.expr}\"", max_side=768).save(args.out, quality=92)
    print("saved", args.out, "- open it and check that the red box is on the right object.")


if __name__ == "__main__":
    main()

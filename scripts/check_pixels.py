#!/usr/bin/env python
"""Verify that --min-pixels / --max-pixels really change the model's input resolution.
Loads only the processor (no GPU, a few seconds) and prints the resized grid for one image.

    python scripts/check_pixels.py --model Qwen/Qwen3-VL-4B-Instruct --image data/dior_rsvg/preview.jpg
    python scripts/check_pixels.py --model Qwen/Qwen3-VL-4B-Instruct --image some.jpg --max-pixels 200704
    python scripts/check_pixels.py --model Qwen/Qwen3-VL-4B-Instruct --image some.jpg --min-pixels 1638400
"""

from __future__ import annotations

import argparse

from _common import ROOT  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--min-pixels", type=int, default=None)
    p.add_argument("--max-pixels", type=int, default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()

    import transformers
    from PIL import Image
    from transformers import AutoProcessor

    from grounding.models import current_pixel_budget, image_kwargs_of, patch_size_of, set_pixel_budget

    print("transformers", transformers.__version__)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    ip = proc.image_processor
    print("image processor:", type(ip).__name__, " size attr:", getattr(ip, "size", None),
          " default budget (min,max):", current_pixel_budget(proc))
    img = Image.open(args.image).convert("RGB")
    patch = patch_size_of(proc)
    merge = getattr(ip, "merge_size", 2)

    def grid(kw):
        out = ip(images=[img], return_tensors="pt", **kw)
        t, h, w = out["image_grid_thw"][0].tolist()
        return h * patch, w * patch, (h * w) // (merge * merge)

    H, W, tok = grid({})
    print(f"image {img.width}x{img.height} -> default input (H,W)=({H},{W}), {tok} visual tokens")
    kw = set_pixel_budget(proc, args.min_pixels, args.max_pixels)
    if kw:
        H2, W2, tok2 = grid({})
        print(f"after attribute patch (implicit): (H,W)=({H2},{W2}), {tok2} tokens")
        H3, W3, tok3 = grid(image_kwargs_of(proc))
        print(f"with explicit kwargs {kw}: (H,W)=({H3},{W3}), {tok3} tokens   <- this is what eval/train use")
        # also through the full processor (text+image), exactly like Grounder.generate
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hi"}]}]
        txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = proc(text=[txt], images=[img], return_tensors="pt", **image_kwargs_of(proc))
        t, h, w = enc["image_grid_thw"][0].tolist()
        print(f"through the full processor: (H,W)=({h * patch},{w * patch}), {(h * w) // (merge * merge)} tokens")
        native = img.width * img.height
        should_change = (args.min_pixels or 0) > native or (args.max_pixels or 10**12) < native
        if should_change and (H3, W3) == (H, W):
            print("WARNING: resolution did not change - the budget is not being applied for this processor/version")
    else:
        print("no --min-pixels/--max-pixels given; nothing to check")


if __name__ == "__main__":
    main()

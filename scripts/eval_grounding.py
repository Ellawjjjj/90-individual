#!/usr/bin/env python
"""Evaluate a VLM on referring-expression grounding (Acc@0.5 / mean IoU).

Example (RefCOCO val subset, Qwen3-VL-4B-Instruct):
    python scripts/eval_grounding.py --model Qwen/Qwen3-VL-4B-Instruct ^
        --dataset data/refcoco/val.jsonl --limit 500 --out runs/q3vl4b_refcoco_val

Outputs in --out:
    raw.jsonl          one line per sample: raw model text, input size, latency (resumable)
    predictions.jsonl  parsed box, IoU, hit, failure category
    metrics.json       args + summary + buckets + coordinate-mode calibration
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from _common import ROOT, add_dataset_args, add_model_args, dataset_kwargs, save_json  # noqa: E402

from grounding.data import load_samples
from grounding.metrics import failure_category, iou_xyxy, summarize
from grounding.parse import COORD_MODES, extract_boxes, to_original_xyxy
from grounding.prompts import build_prompt

log = logging.getLogger("eval")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_args(p)
    add_model_args(p)
    p.add_argument("--coord-mode", default="auto", choices=["auto", *COORD_MODES],
                   help="how to interpret model coordinates; auto = calibrate on the first --calib-n samples")
    p.add_argument("--calib-n", type=int, default=32)
    p.add_argument("--out", required=True, help="output run directory")
    p.add_argument("--resume", action="store_true", help="skip samples already in raw.jsonl")
    p.add_argument("--tag", default="", help="free-form note stored in metrics.json")
    return p.parse_args()


def calibrate(records, candidates=COORD_MODES):
    """Pick the coordinate mode that maximises mean IoU on calibration records."""
    scores = {}
    for mode in candidates:
        ious = []
        for r in records:
            if not r["raw_boxes"]:
                continue
            try:
                pred = to_original_xyxy(r["raw_boxes"][0], mode, r["width"], r["height"],
                                        r["input_hw"][1], r["input_hw"][0])
            except ValueError:
                continue
            ious.append(iou_xyxy(pred, r["gt"]))
        scores[mode] = (sum(ious) / len(ious)) if ious else 0.0
    best = max(scores, key=scores.get)
    return best, scores


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw.jsonl"

    samples = load_samples(args.dataset, **dataset_kwargs(args))
    log.info("loaded %d samples from %s", len(samples), args.dataset)
    by_id = {s.id: s for s in samples}

    # ---- resume support
    raw = {}
    if args.resume and raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                raw[r["id"]] = r
        log.info("resuming: %d samples already done", len(raw))
    todo = [s for s in samples if s.id not in raw]

    # ---- model
    from grounding.models import GenConfig, detect_family, make_grounder

    family = detect_family(args.model)
    is_thinking = "thinking" in args.model.lower() or args.enable_thinking == "on"
    max_new = args.max_new_tokens or (4096 if is_thinking else 128)
    enable_thinking = None if args.enable_thinking == "auto" else (args.enable_thinking == "on")
    cfg = GenConfig(max_new_tokens=max_new, temperature=args.temperature, top_p=args.top_p,
                    top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                    enable_thinking=enable_thinking)

    if todo:
        grounder, family = make_grounder(
            args.model, dtype="bf16" if args.dtype == "4bit" else args.dtype, load_4bit=args.dtype == "4bit",
            adapter=args.adapter, merge_adapter=args.merge_adapter, attn_impl=args.attn,
            min_pixels=args.min_pixels, max_pixels=args.max_pixels, trust_remote_code=args.trust_remote_code)
        log.info("family=%s patch=%d thinking=%s max_new_tokens=%d", family, grounder.patch, is_thinking, max_new)

        from tqdm import tqdm

        bs = max(1, args.batch_size)
        with open(raw_path, "a", encoding="utf-8") as f:
            for i in tqdm(range(0, len(todo), bs), desc="generate"):
                batch = todo[i:i + bs]
                images = [s.load_image() for s in batch]
                prompts = [build_prompt(args.prompt, s.expression) for s in batch]
                try:
                    results = grounder.generate(images, prompts, cfg)
                except Exception as e:  # keep going, record the failure
                    log.exception("generation failed for batch starting at %s: %s", batch[0].id, e)
                    results = None
                if i == 0 and results is not None:
                    s0, r0 = batch[0], results[0]
                    log.info("first sample: original (H,W)=(%d,%d) -> model input (H,W)=%s  [%d visual patches of %dpx]",
                             s0.height, s0.width, tuple(r0.input_hw),
                             (r0.input_hw[0] // grounder.patch) * (r0.input_hw[1] // grounder.patch), grounder.patch)
                    log.info("first raw output: %s", r0.text[:300].replace("\n", " "))
                for j, s in enumerate(batch):
                    if results is None:
                        rec = {"id": s.id, "raw_text": "", "input_hw": [s.height, s.width],
                               "n_new_tokens": 0, "latency_s": None, "error": "generation_failed"}
                    else:
                        r = results[j]
                        rec = {"id": s.id, "raw_text": r.text, "input_hw": list(r.input_hw),
                               "n_new_tokens": r.n_new_tokens, "latency_s": r.latency_s}
                    raw[s.id] = rec
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()

    # ---- parse + calibrate
    recs = []
    for s in samples:
        r = raw.get(s.id)
        if r is None:
            continue
        boxes = extract_boxes(r.get("raw_text", ""))
        recs.append({
            "id": s.id, "expression": s.expression, "gt": s.bbox_xyxy, "width": s.width, "height": s.height,
            "input_hw": r["input_hw"], "raw_boxes": boxes, "n_boxes": len(boxes),
            "raw_text": r.get("raw_text", ""), "n_new_tokens": r.get("n_new_tokens"),
            "latency_s": r.get("latency_s"), "image_path": s.image_path, "row": s.meta.get("row"),
        })

    if not recs:
        raise SystemExit("no generated outputs found (nothing to evaluate)")

    calib = None
    if args.coord_mode == "auto":
        n = min(args.calib_n, len(recs))
        mode, scores = calibrate(recs[:n])
        calib = {"n": n, "scores": scores, "chosen": mode}
        log.info("coordinate-mode calibration on %d samples: %s -> %s", n, scores, mode)
        if scores[mode] < 0.2:
            log.warning("best calibration IoU is low (%.3f); check prompt/parsing with --coord-mode set explicitly",
                        scores[mode])
    else:
        mode = args.coord_mode

    for r in recs:
        if r["raw_boxes"]:
            r["pred"] = to_original_xyxy(r["raw_boxes"][0], mode, r["width"], r["height"],
                                         r["input_hw"][1], r["input_hw"][0])
            r["parsed"] = True
            r["iou"] = iou_xyxy(r["pred"], r["gt"])
        else:
            r["pred"] = None
            r["parsed"] = False
            r["iou"] = 0.0
        r["hit"] = r["iou"] >= 0.5
        r["category"] = failure_category(r)

    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = summarize(recs)
    px = sorted(r["input_hw"][0] * r["input_hw"][1] for r in recs if r.get("input_hw"))
    input_stats = None
    if px:
        input_stats = {"min_pixels": px[0], "median_pixels": px[len(px) // 2], "max_pixels": px[-1],
                       "median_hw": next(r["input_hw"] for r in recs
                                         if r["input_hw"][0] * r["input_hw"][1] == px[len(px) // 2])}
    try:
        import transformers

        tf_version = transformers.__version__
    except Exception:  # noqa: BLE001
        tf_version = None
    metrics = {
        "args": vars(args),
        "model": args.model, "adapter": args.adapter, "family": family,
        "coord_mode": mode, "calibration": calib,
        "max_new_tokens": max_new, "thinking": is_thinking,
        "model_input": input_stats, "transformers_version": tf_version,
        "summary": summary,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(metrics, out / "metrics.json")

    print("\n==== " + args.out + " ====")
    print(f"model={args.model} adapter={args.adapter} dataset={args.dataset} split={args.split} n={summary['n']}")
    if input_stats:
        print(f"model input: median (H,W)={tuple(input_stats['median_hw'])}  pixels min/median/max="
              f"{input_stats['min_pixels']}/{input_stats['median_pixels']}/{input_stats['max_pixels']}")
    print(f"coord_mode={mode}  Acc@0.5={summary['acc@0.5']:.4f}  Acc@0.75={summary['acc@0.75']:.4f}  "
          f"mIoU={summary['mean_iou']:.4f}  parse_fail={summary['parse_fail_rate']:.4f}  "
          f"sec/sample={summary['sec_per_sample'] if summary['sec_per_sample'] is None else round(summary['sec_per_sample'], 3)}")
    for name, b in summary["buckets"].items():
        print(f"  [{name}] " + "  ".join(f"{k}: acc={v['acc@0.5']:.3f} (n={v['n']})" for k, v in b.items()))


if __name__ == "__main__":
    main()

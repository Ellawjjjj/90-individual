"""Shared CLI helpers (argument groups, run-dir utilities)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# make `grounding` importable when running scripts from the repo root or elsewhere
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def add_dataset_args(p: argparse.ArgumentParser):
    g = p.add_argument_group("dataset")
    g.add_argument("--dataset", required=True,
                   help="path/to/file.jsonl  or  hf:<repo_id>[:<config>]")
    g.add_argument("--split", default="val", help="HF split name (ignored for jsonl)")
    g.add_argument("--limit", type=int, default=None, help="random subset size (fixed seed)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--image-field", default="image")
    g.add_argument("--expr-field", default="answer")
    g.add_argument("--bbox-field", default="bbox")
    g.add_argument("--bbox-format", default=None,
                   help="xyxy|xywh|xyxy_norm|xywh_norm|cxcywh|cxcywh_norm (HF default xywh, jsonl default xyxy)")
    g.add_argument("--expr-policy", default="all", choices=["all", "first"])
    g.add_argument("--id-field", default=None)


def dataset_kwargs(args) -> dict:
    kw = dict(split=args.split, limit=args.limit, seed=args.seed,
              image_field=args.image_field, expr_field=args.expr_field,
              bbox_field=args.bbox_field, expr_policy=args.expr_policy, id_field=args.id_field)
    if args.bbox_format:
        kw["bbox_format"] = args.bbox_format
    elif args.dataset.startswith("hf:"):
        kw["bbox_format"] = "xywh"
    return kw


def add_model_args(p: argparse.ArgumentParser, with_generation: bool = True):
    g = p.add_argument_group("model")
    g.add_argument("--model", required=True, help="HF id or local path, e.g. Qwen/Qwen3-VL-4B-Instruct")
    g.add_argument("--adapter", default=None, help="LoRA adapter directory (PEFT)")
    g.add_argument("--merge-adapter", action="store_true")
    g.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32", "4bit"])
    g.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    g.add_argument("--min-pixels", type=int, default=None, help="e.g. 256*28*28=200704")
    g.add_argument("--max-pixels", type=int, default=None, help="e.g. 1280*28*28=1003520")
    g.add_argument("--trust-remote-code", action="store_true")
    if with_generation:
        h = p.add_argument_group("generation")
        h.add_argument("--prompt", default="json", help="prompt template name (grounding/prompts.py)")
        h.add_argument("--max-new-tokens", type=int, default=None,
                       help="default 128 (instruct) / 4096 (thinking)")
        h.add_argument("--temperature", type=float, default=0.0)
        h.add_argument("--top-p", type=float, default=1.0)
        h.add_argument("--top-k", type=int, default=0)
        h.add_argument("--repetition-penalty", type=float, default=1.0)
        h.add_argument("--enable-thinking", default="auto", choices=["auto", "on", "off"],
                       help="pass enable_thinking to the chat template (Qwen3.5 hybrid models)")
        h.add_argument("--batch-size", type=int, default=1)


def save_json(obj, path: str | os.PathLike):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

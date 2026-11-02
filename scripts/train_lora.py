#!/usr/bin/env python
"""LoRA fine-tuning of a Qwen-VL model for referring-expression grounding.

Example (DIOR-RSVG train split, LLM-side LoRA only):
    python scripts/train_lora.py --model Qwen/Qwen3-VL-4B-Instruct ^
        --train data/dior_rsvg/train.jsonl --target-scope llm ^
        --lora-r 16 --lr 1e-4 --epochs 1 --max-pixels 602112 --out runs/lora_dior_llm

Then evaluate with:  scripts/eval_grounding.py --model <same> --adapter runs/lora_dior_llm/adapter ...

Ablation knobs: --target-scope llm|vision|all|custom (which side adapts),
--lora-r, --lr, --train-fraction, --max-pixels, --coord-mode.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path

from _common import ROOT, add_model_args, save_json  # noqa: E402

from grounding.data import load_jsonl
from grounding.parse import from_original_xyxy
from grounding.prompts import build_prompt, build_target

log = logging.getLogger("train")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p, with_generation=False)
    p.add_argument("--train", required=True, help="training jsonl (see grounding/data.py)")
    p.add_argument("--bbox-format", default=None)
    p.add_argument("--train-fraction", type=float, default=1.0, help="use a random fraction of the train set")
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt", default="json")
    p.add_argument("--coord-mode", default=None,
                   help="target convention; default = family default (relative_1000 / absolute_input)")
    p.add_argument("--enable-thinking", default="off", choices=["auto", "on", "off"])
    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-scope", default="llm", choices=["llm", "vision", "all", "custom"])
    p.add_argument("--target-regex", default=None, help="regex over module names when --target-scope custom")
    p.add_argument("--list-modules", action="store_true", help="print candidate Linear module names and exit")
    # optimisation
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--out", required=True)
    return p.parse_args()


import torch  # noqa: E402  (training needs torch anyway)


class GroundingDataset(torch.utils.data.Dataset):
    """Builds (image, prompt, answer) triples. Target boxes are expressed in the
    coordinate convention the model is expected to emit."""

    def __init__(self, samples, processor, prompt_name, coord_mode, patch):
        super().__init__()
        self.samples = samples
        self.processor = processor
        self.prompt_name = prompt_name
        self.coord_mode = coord_mode
        self.patch = patch

    def __len__(self):
        return len(self.samples)

    def _input_hw(self, image):
        """Size after the processor's smart-resize (needed for absolute_input targets)."""
        if self.coord_mode != "absolute_input":
            return None
        from grounding.models import image_kwargs_of

        feats = self.processor.image_processor(images=[image], return_tensors="pt", **image_kwargs_of(self.processor))
        g = feats["image_grid_thw"][0].tolist()
        return int(g[-2]) * self.patch, int(g[-1]) * self.patch

    def __getitem__(self, i):
        s = self.samples[i]
        image = s.load_image()
        hw = self._input_hw(image)
        box = from_original_xyxy(s.bbox_xyxy, self.coord_mode, s.width, s.height,
                                 input_w=hw[1] if hw else None, input_h=hw[0] if hw else None)
        return {
            "image": image,
            "prompt": build_prompt(self.prompt_name, s.expression),
            "answer": build_target(s.expression, box, style=self.prompt_name),
        }


class Collator:
    def __init__(self, processor, enable_thinking):
        self.processor = processor
        self.tok = processor.tokenizer
        self.tok.padding_side = "right"
        self.kw = {} if enable_thinking is None else {"enable_thinking": enable_thinking}

    def _template(self, messages, add_generation_prompt):
        try:
            return self.processor.apply_chat_template(messages, tokenize=False,
                                                      add_generation_prompt=add_generation_prompt, **self.kw)
        except TypeError:
            return self.processor.apply_chat_template(messages, tokenize=False,
                                                      add_generation_prompt=add_generation_prompt)

    def __call__(self, batch):
        texts, suffixes, images = [], [], []
        for ex in batch:
            user = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ex["prompt"]}]}]
            full_msgs = user + [{"role": "assistant", "content": [{"type": "text", "text": ex["answer"]}]}]
            prompt_text = self._template(user, True)
            full_text = self._template(full_msgs, False)
            if full_text.startswith(prompt_text):
                suffix = full_text[len(prompt_text):]
            else:  # template inserted something (e.g. empty <think> block); mask up to the answer
                k = full_text.rfind(ex["answer"])
                suffix = full_text[k:] if k >= 0 else full_text
            texts.append(full_text)
            suffixes.append(suffix)
            images.append(ex["image"])
        from grounding.models import image_kwargs_of

        enc = self.processor(text=texts, images=images, padding=True, return_tensors="pt",
                             **image_kwargs_of(self.processor))
        labels = enc["input_ids"].clone()
        attn = enc["attention_mask"]
        for i, suffix in enumerate(suffixes):
            n_total = int(attn[i].sum().item())
            n_ans = len(self.tok(suffix, add_special_tokens=False)["input_ids"])
            n_ans = min(n_ans, n_total)
            labels[i, : n_total - n_ans] = -100
        labels[attn == 0] = -100
        enc["labels"] = labels
        return enc


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    from grounding.models import (default_coord_mode, detect_family, linear_module_names,
                                  load_model_and_processor, patch_size_of)

    family = detect_family(args.model)
    coord_mode = args.coord_mode or default_coord_mode(family)
    log.info("family=%s coord_mode=%s", family, coord_mode)

    model, processor = load_model_and_processor(
        args.model, dtype="bf16" if args.dtype == "4bit" else args.dtype, load_4bit=args.dtype == "4bit",
        attn_impl=args.attn, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        device_map={"": 0} if torch.cuda.is_available() else None, trust_remote_code=args.trust_remote_code)

    names = linear_module_names(model, args.target_scope, args.target_regex)
    if args.list_modules:
        for n in names:
            print(n)
        print(f"\n{len(names)} Linear modules selected with scope={args.target_scope}")
        return
    if not names:
        raise SystemExit("no Linear modules matched; use --list-modules / --target-regex")
    log.info("LoRA on %d Linear modules (scope=%s), e.g. %s", len(names), args.target_scope, names[:3])

    if args.dtype == "4bit":
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=not args.no_gradient_checkpointing)

    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=names, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    if not args.no_gradient_checkpointing:
        model.enable_input_require_grads()

    # ---- data
    samples = load_jsonl(args.train, bbox_format=args.bbox_format)
    rng = random.Random(args.seed)
    rng.shuffle(samples)
    if args.train_fraction < 1.0:
        samples = samples[: max(1, int(len(samples) * args.train_fraction))]
    if args.train_limit:
        samples = samples[: args.train_limit]
    log.info("training on %d samples", len(samples))
    ds = GroundingDataset(samples, processor, args.prompt, coord_mode, patch_size_of(processor))
    enable_thinking = None if args.enable_thinking == "auto" else (args.enable_thinking == "on")
    collator = Collator(processor, enable_thinking)

    # quick sanity print of one example
    ex = ds[0]
    log.info("example prompt: %s", ex["prompt"])
    log.info("example answer: %s", ex["answer"])

    # integer warm-up steps work on every transformers version (warmup_ratio was removed in v5)
    steps_per_epoch = math.ceil(len(ds) / (args.batch_size * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(steps_per_epoch * args.epochs)
    warmup_steps = int(round(args.warmup_ratio * total_steps))
    log.info("total optimisation steps=%d warmup=%d", total_steps, warmup_steps)

    targ_kwargs = dict(
        output_dir=str(out / "trainer"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.dtype != "fp32",
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,  # Windows-safe
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )
    # drop anything this transformers version does not know (names change between majors)
    supported = set(getattr(TrainingArguments, "__dataclass_fields__", {}).keys())
    if supported:
        dropped = [k for k in targ_kwargs if k not in supported]
        if dropped:
            log.warning("TrainingArguments on transformers %s does not accept %s; dropping",
                        transformers.__version__, dropped)
            targ_kwargs = {k: v for k, v in targ_kwargs.items() if k in supported}
    targs = TrainingArguments(**targ_kwargs)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()

    adapter_dir = out / "adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    save_json({"args": vars(args), "family": family, "coord_mode": coord_mode,
               "n_train": len(samples), "lora_modules": len(names),
               "log_history": trainer.state.log_history}, out / "train_info.json")
    log.info("saved adapter to %s", adapter_dir)


if __name__ == "__main__":
    main()

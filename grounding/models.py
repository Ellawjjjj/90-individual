"""Model loading and batched generation for Qwen2.5-VL / Qwen3-VL / Qwen3.5
(and any other HF image-text-to-text model that follows the same processor API).

Everything here imports torch/transformers lazily so the rest of the package stays
importable on machines without a GPU stack.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

FAMILIES = ("qwen3_5", "qwen3vl", "qwen2_5vl", "qwen2vl", "generic")


def detect_family(model_id: str) -> str:
    m = model_id.lower().replace("_", "-")
    if "qwen3.5" in m or "qwen3-5" in m:
        return "qwen3_5"
    if "qwen3-vl" in m or "qwen3vl" in m:
        return "qwen3vl"
    if "qwen2.5-vl" in m or "qwen2-5-vl" in m:
        return "qwen2_5vl"
    if "qwen2-vl" in m:
        return "qwen2vl"
    return "generic"


def default_coord_mode(family: str) -> str:
    """Best-guess output convention per family. Verify with --coord-mode auto."""
    return "absolute_input" if family == "qwen2_5vl" else "relative_1000"


def _torch_dtype(name: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _dtype_kwargs(dtype):
    """transformers renamed torch_dtype -> dtype in 4.56; pass the right one."""
    import transformers
    from packaging import version

    if version.parse(transformers.__version__) >= version.parse("4.56.0"):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def set_pixel_budget(processor, min_pixels: Optional[int] = None, max_pixels: Optional[int] = None):
    """Control the dynamic-resolution token budget of Qwen-style image processors."""
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return
    for key, val in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
        if val is None:
            continue
        if hasattr(ip, key):
            try:
                setattr(ip, key, int(val))
            except Exception:
                pass
        size = getattr(ip, "size", None)
        if isinstance(size, dict):
            if key == "min_pixels":
                for k in ("shortest_edge", "min_pixels"):
                    if k in size:
                        size[k] = int(val)
            else:
                for k in ("longest_edge", "max_pixels"):
                    if k in size:
                        size[k] = int(val)


def patch_size_of(processor) -> int:
    ip = getattr(processor, "image_processor", None)
    for attr in ("patch_size",):
        v = getattr(ip, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return 14


def load_model_and_processor(
    model_id: str,
    dtype: str = "bf16",
    load_4bit: bool = False,
    adapter: Optional[str] = None,
    merge_adapter: bool = False,
    attn_impl: str = "sdpa",
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    device_map: Any = "auto",
    trust_remote_code: bool = False,
):
    import torch
    import transformers
    from transformers import AutoProcessor

    kwargs: Dict[str, Any] = dict(device_map=device_map, trust_remote_code=trust_remote_code)
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    if load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs.update(_dtype_kwargs(torch.bfloat16))
    else:
        kwargs.update(_dtype_kwargs(_torch_dtype(dtype)))

    # Try the generic auto classes first; fall back to family-specific classes.
    candidates = [
        "AutoModelForImageTextToText",
        "AutoModelForMultimodalLM",  # transformers >= 5 (Qwen3.5)
        "AutoModelForVision2Seq",
        "Qwen3VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
    ]
    model = None
    errors = []
    for name in candidates:
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_id, **kwargs)
            log.info("loaded %s with %s", model_id, name)
            break
        except (ValueError, KeyError, AttributeError, TypeError) as e:  # unrecognised config etc.
            errors.append(f"{name}: {type(e).__name__}: {str(e)[:200]}")
            continue
    if model is None:
        raise RuntimeError(
            "Could not load model with any known class. Upgrade transformers "
            "(`pip install -U transformers`). Errors:\n" + "\n".join(errors)
        )

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        if merge_adapter and not load_4bit:
            model = model.merge_and_unload()
    model.eval()

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    set_pixel_budget(processor, min_pixels, max_pixels)
    return model, processor


@dataclass
class GenConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    enable_thinking: Optional[bool] = None  # None -> don't pass the template kwarg

    def to_generate_kwargs(self) -> Dict[str, Any]:
        kw: Dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        if self.temperature and self.temperature > 0:
            kw.update(do_sample=True, temperature=self.temperature, top_p=self.top_p)
            if self.top_k:
                kw["top_k"] = self.top_k
        else:
            kw.update(do_sample=False)
            # silence sampling-param warnings on greedy decode
            kw.update(temperature=None, top_p=None, top_k=None)
        if self.repetition_penalty and self.repetition_penalty != 1.0:
            kw["repetition_penalty"] = self.repetition_penalty
        return kw


@dataclass
class GenResult:
    text: str
    input_hw: Tuple[int, int]  # (H, W) of the image as the model actually saw it
    n_new_tokens: int
    latency_s: float


class Grounder:
    """Thin wrapper: (PIL image, prompt) -> raw text + resized input size."""

    def __init__(self, model, processor, family: str = "generic"):
        self.model = model
        self.processor = processor
        self.family = family
        self.patch = patch_size_of(processor)
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            tok.padding_side = "left"

    def _chat_text(self, prompt: str, enable_thinking: Optional[bool]) -> str:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        kw: Dict[str, Any] = {}
        if enable_thinking is not None:
            kw["enable_thinking"] = bool(enable_thinking)
        try:
            return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
        except TypeError:
            return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @staticmethod
    def _input_hw(inputs, i: int, patch: int, fallback: Tuple[int, int]) -> Tuple[int, int]:
        grid = inputs.get("image_grid_thw", None)
        if grid is None:
            return fallback
        g = grid[i].tolist() if hasattr(grid[i], "tolist") else list(grid[i])
        t, h, w = g[-3], g[-2], g[-1]
        return int(h) * patch, int(w) * patch

    def generate(self, images: Sequence, prompts: Sequence[str], cfg: GenConfig) -> List[GenResult]:
        import torch

        assert len(images) == len(prompts)
        texts = [self._chat_text(p, cfg.enable_thinking) for p in prompts]
        inputs = self.processor(text=texts, images=list(images), padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        in_len = inputs["input_ids"].shape[1]
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**inputs, **cfg.to_generate_kwargs())
        dt = time.perf_counter() - t0
        new = out[:, in_len:]
        texts_out = self.processor.batch_decode(new, skip_special_tokens=True)
        results = []
        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        for i, txt in enumerate(texts_out):
            n_new = int((new[i] != pad_id).sum().item()) if pad_id is not None else int(new.shape[1])
            fallback = (images[i].height, images[i].width)
            results.append(GenResult(text=txt, input_hw=self._input_hw(inputs, i, self.patch, fallback),
                                     n_new_tokens=n_new, latency_s=dt / len(images)))
        return results


class FakeGrounder:
    """`--model fake`: no GPU, no download. Emits a random JSON box in relative-1000
    coordinates so the whole pipeline (parsing, calibration, metrics, visualisation,
    summaries) can be exercised before the real model is available."""

    patch = 16

    def __init__(self, seed: int = 0):
        import random

        self.rng = random.Random(seed)

    def generate(self, images, prompts, cfg) -> List[GenResult]:
        out = []
        for img in images:
            x1, y1 = self.rng.randint(0, 600), self.rng.randint(0, 600)
            x2, y2 = self.rng.randint(x1 + 50, 1000), self.rng.randint(y1 + 50, 1000)
            txt = '```json\n[{"bbox_2d": [%d, %d, %d, %d], "label": "fake"}]\n```' % (x1, y1, x2, y2)
            if self.rng.random() < 0.05:  # occasional parse failure
                txt = "I am not sure where that is."
            out.append(GenResult(text=txt, input_hw=(img.height, img.width), n_new_tokens=30, latency_s=0.001))
        return out


def make_grounder(model_id: str, **load_kwargs):
    """Load a real model or the fake one; returns (grounder, family)."""
    family = detect_family(model_id)
    if model_id.lower() == "fake":
        return FakeGrounder(), "fake"
    model, processor = load_model_and_processor(model_id, **load_kwargs)
    return Grounder(model, processor, family), family


def linear_module_names(model, scope: str = "llm", custom_regex: Optional[str] = None,
                        exclude_regex: str = r"(lm_head|embed_tokens|patch_embed)") -> List[str]:
    """Names of nn.Linear modules to wrap with LoRA.

    scope: 'llm' (language model only), 'vision' (vision tower + merger), 'all', or 'custom'.
    Works across naming schemes by looking for 'visual' / 'vision' in the module path.
    """
    import re

    import torch.nn as nn

    vis_pat = re.compile(r"(visual|vision_tower|vision_model|vision_encoder|merger|projector)", re.I)
    excl = re.compile(exclude_regex) if exclude_regex else None
    cust = re.compile(custom_regex) if custom_regex else None
    names = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if excl and excl.search(name):
            continue
        is_vis = bool(vis_pat.search(name))
        if scope == "llm" and is_vis:
            continue
        if scope == "vision" and not is_vis:
            continue
        if scope == "custom":
            if cust is None or not cust.search(name):
                continue
        names.append(name)
    return names

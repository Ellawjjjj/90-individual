# P2 - hyper-parameter analysis on one model / one dataset.
# Each sweep varies ONE factor; everything else stays at the P0 defaults.
$ErrorActionPreference = "Stop"
$M = "Qwen/Qwen3-VL-8B-Instruct"
$D = "data/refcoco/val.jsonl"
$N = 500

# 1) visual token budget (dynamic resolution). 28*28 = one merged patch for Qwen2.5-VL,
#    Qwen3-VL uses 32*32; the numbers below are just pixel counts and work for both.
foreach ($px in @(200704, 401408, 802816, 1605632)) {      # 256, 512, 1024, 2048 patches of 28x28
  python scripts/eval_grounding.py --model $M --dataset $D --limit $N --batch-size 2 `
    --max-pixels $px --out "runs/p2_pixels_$px" --resume --tag "p2 max_pixels=$px"
}

# 1b) same sweep on DIOR-RSVG (800x800, many small targets). Native resolution is
#     640k pixels; --min-pixels forces UPSCALING so the model sees more tokens per object.
$DD = "data/dior_rsvg/test.jsonl"
$M4 = "Qwen/Qwen3-VL-4B-Instruct"
python scripts/eval_grounding.py --model $M4 --dataset $DD --limit $N --batch-size 4 --max-pixels 200704 `
  --out "runs/p2_dior_px_down448" --resume --tag "p2 dior max_pixels=448^2"
python scripts/eval_grounding.py --model $M4 --dataset $DD --limit $N --batch-size 4 `
  --out "runs/p2_dior_px_native800" --resume --tag "p2 dior native"
python scripts/eval_grounding.py --model $M4 --dataset $DD --limit $N --batch-size 2 --min-pixels 1048576 `
  --out "runs/p2_dior_px_up1024" --resume --tag "p2 dior min_pixels=1024^2"
python scripts/eval_grounding.py --model $M4 --dataset $DD --limit $N --batch-size 2 --min-pixels 1638400 `
  --out "runs/p2_dior_px_up1280" --resume --tag "p2 dior min_pixels=1280^2"

# 2) decoding
foreach ($t in @(0.0, 0.3, 0.7, 1.0)) {
  python scripts/eval_grounding.py --model $M --dataset $D --limit $N --batch-size 2 `
    --temperature $t --top-p 0.9 --out "runs/p2_temp_$t" --resume --tag "p2 temperature=$t"
}

# 3) prompt template
foreach ($pr in @("json", "json_strict", "plain", "refcoco", "cot")) {
  python scripts/eval_grounding.py --model $M --dataset $D --limit $N --batch-size 2 `
    --prompt $pr --max-new-tokens 256 --out "runs/p2_prompt_$pr" --resume --tag "p2 prompt=$pr"
}

# 4) model scale and precision
foreach ($m in @("Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-8B-Instruct")) {
  $tag = ($m -split "/")[-1].ToLower()
  python scripts/eval_grounding.py --model $m --dataset $D --limit $N --batch-size 2 `
    --out "runs/p2_scale_$tag" --resume --tag "p2 scale"
}
python scripts/eval_grounding.py --model $M --dataset $D --limit $N --batch-size 2 `
  --dtype 4bit --out "runs/p2_4bit_8b" --resume --tag "p2 4bit"

# 5) thinking on/off at equal scale (Instruct vs Thinking checkpoints)
python scripts/eval_grounding.py --model "Qwen/Qwen3-VL-8B-Thinking" --dataset $D --limit 200 `
  --batch-size 1 --out "runs/p2_thinking_8b" --resume --tag "p2 thinking"

python scripts/summarize.py --glob "runs/p2_*" --out runs/summary_p2

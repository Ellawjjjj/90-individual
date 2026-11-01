# P1 - out-of-domain datasets + three model generations under one protocol.
# Prerequisite: data/dior_rsvg/test.jsonl (tools/convert_dior_rsvg.py),
#               data/screenspot/test.jsonl (tools/convert_hf_to_jsonl.py ... --bbox-format xyxy_norm),
#               optional data/my_photos/all.jsonl
$ErrorActionPreference = "Stop"
$MODELS = @("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3.5-9B")
$DATASETS = @{
  "refcoco_val"   = "data/refcoco/val.jsonl";
  "dior_test"     = "data/dior_rsvg/test.jsonl";
  "screenspot"    = "data/screenspot/test.jsonl";
  # "myphotos"    = "data/my_photos/all.jsonl";
}

foreach ($m in $MODELS) {
  $tag = ($m -split "/")[-1].ToLower()
  foreach ($d in $DATASETS.Keys) {
    $extra = @()
    if ($m -like "*Qwen3.5*") { $extra += @("--enable-thinking", "off") }   # hybrid model: non-thinking mode
    python scripts/eval_grounding.py --model $m --dataset $DATASETS[$d] --limit 500 `
      --batch-size 2 --out "runs/p1_${tag}_$d" --resume --tag p1 @extra
  }
}
python scripts/summarize.py --glob "runs/p1_*" --out runs/summary_p1

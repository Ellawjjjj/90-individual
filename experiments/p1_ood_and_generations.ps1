# P1 - out-of-domain datasets + three model generations under one protocol (~45 min + downloads).
# Prerequisites:
#   data/dior_rsvg/test.jsonl   (tools/convert_dior_rsvg.py)
#   data/screenspot/test.jsonl  (tools/convert_hf_to_jsonl.py --bbox-format xyxy_norm, see 00_prepare_data.ps1)
#   models downloaded beforehand (in another terminal):
#     hf download Qwen/Qwen2.5-VL-7B-Instruct ; hf download Qwen/Qwen3.5-9B
#     (older CLI: huggingface-cli download ...)
$ErrorActionPreference = "Stop"
$N = 500
$MODELS = @("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3.5-9B")
$DATASETS = [ordered]@{
  "refcoco_val" = "data/refcoco/val.jsonl";
  "dior_test"   = "data/dior_rsvg/test.jsonl";
  "screenspot"  = "data/screenspot/test.jsonl";
  # "myphotos"  = "data/my_photos/all.jsonl";
}

foreach ($m in $MODELS) {
  $tag = ($m -split "/")[-1].ToLower()
  foreach ($d in $DATASETS.Keys) {
    $path = $DATASETS[$d]
    if (-not (Test-Path $path)) { Write-Host "missing $path - skipped"; continue }
    $extra = @()
    if ($m -like "*Qwen3.5*") { $extra += @("--enable-thinking", "off") }   # hybrid model: non-thinking mode
    python scripts/eval_grounding.py --model $m --dataset $path --limit $N `
      --batch-size 2 --out "runs/p1_${tag}_$d" --resume --tag p1 @extra
  }
}
python scripts/summarize.py --glob "runs/p1_*" --out runs/summary_p1
# paired comparisons per dataset (baseline = Qwen3-VL-8B)
foreach ($d in $DATASETS.Keys) {
  $runs = Get-ChildItem -Directory "runs" | Where-Object { $_.Name -like "p1_*_$d" -and $_.Name -notlike "p1_qwen3-vl-8b-instruct_*" } | ForEach-Object { $_.FullName }
  if ((Test-Path "runs/p1_qwen3-vl-8b-instruct_$d") -and $runs) {
    python scripts/compare_runs.py --base "runs/p1_qwen3-vl-8b-instruct_$d" @runs > "runs/compare_p1_$d.txt"
    Get-Content "runs/compare_p1_$d.txt"
  }
}

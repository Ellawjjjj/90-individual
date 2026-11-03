# P0 - reproduce Qwen3-VL Table 4 (RefCOCO-avg) with the Instruct models on the 8 standard splits.
# ~1.5 h on a 5090 (16 runs x 500 samples). Thinking models are in p0b_thinking.ps1 (slow, optional).
# Prerequisite: data/refcoco*/**.jsonl from experiments/00_prepare_data.ps1
$ErrorActionPreference = "Stop"
$LIMIT = 500
$SPLITS = @("refcoco/val", "refcoco/testA", "refcoco/testB",
            "refcoco_plus/val", "refcoco_plus/testA", "refcoco_plus/testB",
            "refcocog/val", "refcocog/test")
$MODELS = @("Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-8B-Instruct")

foreach ($m in $MODELS) {
  $tag = ($m -split "/")[-1].ToLower()
  foreach ($s in $SPLITS) {
    $name = ($s -replace "/", "_")
    if (-not (Test-Path "data/$s.jsonl")) { Write-Host "missing data/$s.jsonl - skipped"; continue }
    python scripts/eval_grounding.py --model $m --dataset "data/$s.jsonl" --limit $LIMIT `
      --batch-size 4 --out "runs/p0_${tag}_$name" --resume --tag p0
  }
}

python scripts/summarize.py --glob "runs/p0_*" --out runs/summary_p0
python tools/p0_table.py --out runs/table_p0_vs_paper.md

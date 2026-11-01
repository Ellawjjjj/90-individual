# P0 - reproduce Qwen3-VL Table 4 (RefCOCO-avg) on subsets of the 8 standard splits.
# Run from the repo root in PowerShell:  .\experiments\p0_reproduce.ps1
# Prerequisite: data/refcoco/*.jsonl produced by tools/convert_hf_to_jsonl.py
$ErrorActionPreference = "Stop"
$LIMIT = 500          # per split; raise to 1000+ for the final table if time allows
$SPLITS = @("refcoco/val", "refcoco/testA", "refcoco/testB",
            "refcoco_plus/val", "refcoco_plus/testA", "refcoco_plus/testB",
            "refcocog/val", "refcocog/test")
$MODELS = @("Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-8B-Instruct")

foreach ($m in $MODELS) {
  $tag = ($m -split "/")[-1].ToLower()
  foreach ($s in $SPLITS) {
    $name = ($s -replace "/", "_")
    python scripts/eval_grounding.py --model $m --dataset "data/$s.jsonl" --limit $LIMIT `
      --batch-size 4 --out "runs/p0_${tag}_$name" --resume --tag p0
  }
}

# Thinking variants are ~10x slower; use a smaller subset and batch-size 1.
foreach ($m in @("Qwen/Qwen3-VL-4B-Thinking", "Qwen/Qwen3-VL-8B-Thinking")) {
  $tag = ($m -split "/")[-1].ToLower()
  foreach ($s in @("refcoco/val", "refcoco_plus/val", "refcocog/val")) {
    $name = ($s -replace "/", "_")
    python scripts/eval_grounding.py --model $m --dataset "data/$s.jsonl" --limit 200 `
      --batch-size 1 --max-new-tokens 4096 --out "runs/p0_${tag}_$name" --resume --tag p0
  }
}

python scripts/summarize.py --glob "runs/p0_*" --out runs/summary_p0

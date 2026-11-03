# P4 - qualitative analysis: good / bad cases, failure buckets, cross-model and before/after comparisons.
# All CPU; ~5 min. Figures are copied to docs/figures for the report.
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force docs/figures | Out-Null

function Vis($run, $args) { if (Test-Path "$run/predictions.jsonl") { python scripts/visualize.py --run $run @args } else { Write-Host "skip $run" } }

# in-domain vs out-of-domain, best Instruct model
Vis runs/p0_qwen3-vl-8b-instruct_refcoco_val @("--n-good", "8", "--n-bad", "16", "--save-individual")
Vis runs/p1_qwen3-vl-8b-instruct_dior_test   @("--n-good", "8", "--n-bad", "16", "--save-individual")
Vis runs/p1_qwen3-vl-8b-instruct_screenspot  @("--n-good", "8", "--n-bad", "16")

# three generations on the same hard RefCOCO / DIOR samples
Vis runs/p1_qwen3-vl-8b-instruct_refcoco_val @("--compare", "runs/p1_qwen2.5-vl-7b-instruct_refcoco_val", "runs/p1_qwen3.5-9b_refcoco_val", "--n-bad", "10")
Vis runs/p1_qwen3-vl-8b-instruct_dior_test   @("--compare", "runs/p1_qwen2.5-vl-7b-instruct_dior_test",   "runs/p1_qwen3.5-9b_dior_test",   "--n-bad", "10")

# LoRA before/after on DIOR: fixed and broken cases
Vis runs/p3_zeroshot_dior @("--compare", "runs/p3_lora_llm_dior", "runs/p3_lora_all_dior", "--select", "fixed",  "--n-bad", "8")
Vis runs/p3_zeroshot_dior @("--compare", "runs/p3_lora_llm_dior", "runs/p3_lora_all_dior", "--select", "broken", "--n-bad", "8")

# collect figures for the report
$copies = @(
  @("runs/p0_qwen3-vl-8b-instruct_refcoco_val/vis/good_cases.jpg",      "refcoco_good.jpg"),
  @("runs/p0_qwen3-vl-8b-instruct_refcoco_val/vis/bad_wrong_object.jpg", "refcoco_bad_wrong_object.jpg"),
  @("runs/p1_qwen3-vl-8b-instruct_dior_test/vis/good_cases.jpg",        "dior_good.jpg"),
  @("runs/p1_qwen3-vl-8b-instruct_dior_test/vis/bad_wrong_object.jpg",  "dior_bad_wrong_object.jpg"),
  @("runs/p1_qwen3-vl-8b-instruct_screenspot/vis/bad_wrong_object.jpg", "screenspot_bad.jpg"),
  @("runs/p1_qwen3-vl-8b-instruct_refcoco_val/vis/compare.jpg",         "generations_refcoco.jpg"),
  @("runs/p1_qwen3-vl-8b-instruct_dior_test/vis/compare.jpg",           "generations_dior.jpg"),
  @("runs/p3_zeroshot_dior/vis/compare_fixed.jpg",                      "dior_lora_fixed.jpg"),
  @("runs/p3_zeroshot_dior/vis/compare_broken.jpg",                     "dior_lora_broken.jpg")
)
foreach ($c in $copies) { if (Test-Path $c[0]) { Copy-Item $c[0] ("docs/figures/" + $c[1]) -Force } }
Get-ChildItem docs/figures

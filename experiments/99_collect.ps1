# 99 - collect everything for the report: summary tables, paired tests, P0 table, figures. CPU only, ~1 min.
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force docs/figures | Out-Null

python scripts/summarize.py --glob "runs/p0*_*"        --out runs/summary_p0
python scripts/summarize.py --glob "runs/p1_*"         --out runs/summary_p1
python scripts/summarize.py --glob "runs/p2_*"         --out runs/summary_p2
python scripts/summarize.py --glob "runs/p3_*_dior"    --out runs/summary_p3_dior
python scripts/summarize.py --glob "runs/p3_*_refcoco" --out runs/summary_p3_forgetting
python tools/p0_table.py --out runs/table_p0_vs_paper.md

function Cmp($name, $base, $others) {
  $ok = @($others | Where-Object { Test-Path "$_/predictions.jsonl" })
  if ((Test-Path "$base/predictions.jsonl") -and $ok.Count -gt 0) {
    python scripts/compare_runs.py --base $base @ok > "runs/compare_$name.txt"; Write-Host "wrote runs/compare_$name.txt"
  } else { Write-Host "skip compare_$name" }
}
Cmp "resolution"    "runs/p1_qwen3-vl-4b-instruct_dior_test" @("runs/p2_dior_px_down448","runs/p2_dior_px_up1024","runs/p2_dior_px_up1280")
Cmp "p3_vs_zeroshot" "runs/p3_zeroshot_dior" @("runs/p3_lora_llm_dior","runs/p3_lora_vision_dior","runs/p3_lora_all_dior","runs/p3_rank_4_dior","runs/p3_rank_64_dior","runs/p3_data_300_dior","runs/p3_data_1000_dior","runs/p3_data_1000_ep3_dior","runs/p3_data_3000_ep3_dior","runs/p3_lora_llm_1024_dior")
Cmp "p3_vs_llm"      "runs/p3_lora_llm_dior" @("runs/p3_lora_all_dior","runs/p3_rank_64_dior","runs/p3_lora_llm_1024_dior","runs/p3_data_1000_ep3_dior","runs/p3_data_3000_ep3_dior")
Cmp "steps"          "runs/p3_data_1000_dior" @("runs/p3_data_1000_ep3_dior","runs/p3_data_3000_dior")
Cmp "rank"           "runs/p3_rank_4_dior" @("runs/p3_rank_16_dior","runs/p3_rank_64_dior")
Cmp "forgetting"     "runs/p0_qwen3-vl-4b-instruct_refcoco_val" @("runs/p3_lora_llm_refcoco","runs/p3_lora_vision_refcoco","runs/p3_lora_all_refcoco")
Cmp "scale_refcoco"  "runs/p0_qwen3-vl-4b-instruct_refcoco_val" @("runs/p0_qwen3-vl-8b-instruct_refcoco_val")
Cmp "scale_dior"     "runs/p1_qwen3-vl-4b-instruct_dior_test" @("runs/p1_qwen3-vl-8b-instruct_dior_test")

.\experiments\p4_visualize.ps1

Write-Host "`n==== files for the report ===="
Get-ChildItem runs -Filter "summary_*.md"; Get-ChildItem runs -Filter "table_*.md"; Get-ChildItem runs -Filter "compare_*.txt"; Get-ChildItem docs/figures

# P0b - Thinking vs Instruct at equal scale, on the SAME 100 RefCOCO-val samples
# (limit=100 is a nested subset of limit=500 with the same seed, so compare_runs.py can pair them).
# Thinking models generate long reasoning chains: ~10-40 s per sample -> ~1-1.5 h total. Run last.
$ErrorActionPreference = "Stop"
$D = "data/refcoco/val.jsonl"
foreach ($m in @("Qwen/Qwen3-VL-4B-Thinking", "Qwen/Qwen3-VL-8B-Thinking")) {
  $tag = ($m -split "/")[-1].ToLower()
  python scripts/eval_grounding.py --model $m --dataset $D --limit 100 --batch-size 1 `
    --max-new-tokens 4096 --out "runs/p0b_${tag}_refcoco_val" --resume --tag p0b
}
# paired comparison against the Instruct runs on the shared 100 samples
python scripts/compare_runs.py --base runs/p0_qwen3-vl-4b-instruct_refcoco_val runs/p0b_qwen3-vl-4b-thinking_refcoco_val > runs/compare_thinking_4b.txt
python scripts/compare_runs.py --base runs/p0_qwen3-vl-8b-instruct_refcoco_val runs/p0b_qwen3-vl-8b-thinking_refcoco_val > runs/compare_thinking_8b.txt
Get-Content runs/compare_thinking_4b.txt, runs/compare_thinking_8b.txt
# reasoning-chain excerpts for the report (worst cases of the thinking model vs instruct)
python scripts/visualize.py --run runs/p0b_qwen3-vl-4b-thinking_refcoco_val --compare runs/p0_qwen3-vl-4b-instruct_refcoco_val --n-bad 8

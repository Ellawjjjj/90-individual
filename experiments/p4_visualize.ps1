# P4 - qualitative analysis: good / bad cases, failure buckets, cross-model comparison
$ErrorActionPreference = "Stop"

python scripts/visualize.py --run runs/p0_qwen3-vl-8b-instruct_refcoco_val --n-good 8 --n-bad 24 --save-individual
python scripts/visualize.py --run runs/p1_qwen3-vl-8b-instruct_dior_test --n-good 8 --n-bad 24 --save-individual
python scripts/visualize.py --run runs/p1_qwen3-vl-8b-instruct_screenspot --n-good 8 --n-bad 24

# same samples, three generations side by side (runs must share dataset/limit/seed)
python scripts/visualize.py --run runs/p1_qwen3-vl-8b-instruct_refcoco_val `
  --compare runs/p1_qwen2.5-vl-7b-instruct_refcoco_val runs/p1_qwen3.5-9b_refcoco_val --n-bad 12

# thinking vs instruct on the thinking model's hardest cases
python scripts/visualize.py --run runs/p2_thinking_8b --compare runs/p0_qwen3-vl-8b-instruct_refcoco_val --n-bad 12

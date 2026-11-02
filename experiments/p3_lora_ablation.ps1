# P3 - LoRA adaptation to an out-of-domain dataset + component ablation.
# Question answered: when the model adapts to a new domain, which side carries the
# grounding ability - the LLM, the vision tower, or both? And how much is forgotten?
$ErrorActionPreference = "Stop"
$M = "Qwen/Qwen3-VL-4B-Instruct"          # 4B trains comfortably on 32 GB; 8B also fits with batch 1
$TRAIN = "data/dior_rsvg/train.jsonl"
$TEST  = "data/dior_rsvg/test.jsonl"
$INDOMAIN = "data/refcoco/val.jsonl"     # forgetting check
$PX = 655360                              # >= 800*800 so DIOR images stay at native resolution (625 visual tokens)

# baseline (zero-shot) on the target domain
python scripts/eval_grounding.py --model $M --dataset $TEST --limit 500 --batch-size 2 --max-pixels $PX `
  --out runs/p3_zeroshot_dior --resume --tag p3

# a) which modules adapt
foreach ($scope in @("llm", "vision", "all")) {
  python scripts/train_lora.py --model $M --train $TRAIN --train-limit 3000 --target-scope $scope `
    --lora-r 16 --lr 1e-4 --epochs 1 --batch-size 2 --grad-accum 4 --max-pixels $PX --out "runs/p3_lora_$scope"
  python scripts/eval_grounding.py --model $M --adapter "runs/p3_lora_$scope/adapter" --dataset $TEST `
    --limit 500 --batch-size 2 --max-pixels $PX --out "runs/p3_lora_${scope}_dior" --resume --tag "p3 scope=$scope"
  python scripts/eval_grounding.py --model $M --adapter "runs/p3_lora_$scope/adapter" --dataset $INDOMAIN `
    --limit 500 --batch-size 2 --max-pixels $PX --out "runs/p3_lora_${scope}_refcoco" --resume --tag "p3 forgetting scope=$scope"
}

# b) LoRA rank (LLM scope)
foreach ($r in @(4, 16, 64)) {
  python scripts/train_lora.py --model $M --train $TRAIN --train-limit 3000 --target-scope llm `
    --lora-r $r --lora-alpha (2*$r) --lr 1e-4 --epochs 1 --batch-size 2 --grad-accum 4 --max-pixels $PX --out "runs/p3_rank_$r"
  python scripts/eval_grounding.py --model $M --adapter "runs/p3_rank_$r/adapter" --dataset $TEST `
    --limit 500 --batch-size 2 --max-pixels $PX --out "runs/p3_rank_${r}_dior" --resume --tag "p3 rank=$r"
}

# c) training data size (LLM scope, r=16)
foreach ($n in @(300, 1000, 3000)) {
  python scripts/train_lora.py --model $M --train $TRAIN --train-limit $n --target-scope llm `
    --lora-r 16 --lr 1e-4 --epochs 1 --batch-size 2 --grad-accum 4 --max-pixels $PX --out "runs/p3_data_$n"
  python scripts/eval_grounding.py --model $M --adapter "runs/p3_data_$n/adapter" --dataset $TEST `
    --limit 500 --batch-size 2 --max-pixels $PX --out "runs/p3_data_${n}_dior" --resume --tag "p3 n_train=$n"
}

python scripts/summarize.py --glob "runs/p3_*_dior" --out runs/summary_p3_dior
python scripts/summarize.py --glob "runs/p3_*_refcoco" --out runs/summary_p3_forgetting

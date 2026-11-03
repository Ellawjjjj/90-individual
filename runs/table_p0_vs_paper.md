| model | RC val | RC testA | RC testB | RC+ val | RC+ testA | RC+ testB | RCg val | RCg test | avg (ours) | avg (paper) | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-vl-4b-instruct | 92.6 | 91.6 | 88.4 | 84.2 | 89.2 | 81.6 | 88.4 | 85.8 | 87.7 | 89.0 | -1.3 |
| qwen3-vl-8b-instruct | 92.2 | 91.6 | 88.4 | 86.8 | 90.4 | 79.8 | 87.8 | 87.8 | 88.1 | 89.1 | -1.0 |
| qwen3-vl-4b-thinking (EGM card, ref.) | 90.0 | 92.7 | 85.6 | 85.2 | 89.5 | 79.3 | 87.0 | 87.7 | 87.1 | 88.2 | – |
| qwen3-vl-8b-thinking (EGM card, ref.) | 91.0 | 92.5 | 86.6 | 86.2 | 91.2 | 80.5 | 87.8 | 88.6 | 88.0 | 88.2 | – |

Acc@0.5 (%). Ours: random subsets of n=[500] expressions per split (seed 0). Paper: Qwen3-VL Technical Report Table 4 (full splits). EGM rows: per-split numbers from NVIDIA EGM model cards for the Thinking checkpoints.
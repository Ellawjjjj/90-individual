<!--
DRAFT v3（提交前删除这段注释）
- 数字来自 runs/*/metrics.json 与 runs/compare_*.txt。
- 请通读并改成自己的表述，按课程规定申报 AI 辅助。
- 生成 Word：python tools/build_report.py
-->

# Visual Grounding with Qwen3-VL: Reproduction, Out-of-Domain Analysis, and Parameter-Efficient Adaptation

**Author:** Ella Wang · **Course assignment 1 (individual)** · Code: https://github.com/Ellawjjjj/90-individual

## Abstract

This report studies referring expression comprehension (REC) with Qwen3-VL, a vision-language model that answers grounding queries by writing box coordinates as text. We reproduce its RefCOCO/RefCOCO+/RefCOCOg results on a single RTX 5090 (8-split average 87.7 for 4B and 88.1 for 8B, against 89.0 and 89.1 in the paper), then evaluate it on two datasets that the paper does not use: remote-sensing images (DIOR-RSVG) and GUI screenshots (ScreenSpot). Accuracy drops from 92.6 on RefCOCO val to 53.6 and 56.0. Most of the remote-sensing errors come from relational expressions over many similar small objects. Increasing the input resolution helps a little (+3.4 at 1024²); a LoRA adapter on the language model alone helps a lot (+18.2 with 0.74% of the parameters). In the ablation of the adaptation recipe, the number of optimisation steps matters more than the amount of training data, the LoRA rank or the resolution. We also compare three model generations (Qwen2.5-VL, Qwen3-VL, Qwen3.5) and find that the gap between generations on out-of-domain data (14 to 31 points) is far larger than the gap between the 4B and 8B models.

## 1 Introduction

Visual grounding means finding the image region that a sentence refers to, for example "the chimney on the right of the chimney on the left". Detection-based methods such as MDETR and Grounding DINO attach a text-conditioned head to a detector. Generalist vision-language models (VLMs) work differently: the image becomes a sequence of visual tokens, the language model reads the expression, and the output is a string of coordinates. This makes grounding open-vocabulary and lets one model handle many tasks. It also means that localisation quality depends on how well the language model has learned to map tokens to pixel positions.

We chose Qwen3-VL (Qwen Team, 2025) as the main method. It is open, comes in sizes that fit on one GPU (2B to 8B), and its technical report describes the grounding evaluation in Section 5.5, which gives us numbers to reproduce. It also belongs to a model family whose coordinate format has changed twice (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5), so comparing generations under one protocol is straightforward to set up and interesting to look at.

We address four questions:

- **RQ1.** Can the paper's RefCOCO results be reproduced with an independent evaluation script on random subsets?
- **RQ2.** How much accuracy is lost on remote-sensing images and GUI screenshots, and which expressions and targets fail?
- **RQ3.** How sensitive is accuracy to input resolution, model size and the generation budget?
- **RQ4.** When the model is adapted to remote-sensing data with LoRA, where does the gain come from, and how do rank, data size, training steps and resolution affect accuracy and forgetting?

All experiments run on one RTX 5090 (32 GB) under Windows, using Hugging Face `transformers` and `peft`.

## 2 Background and Related Work

**Detection-based REC.** MDETR (Kamath et al., 2021) conditions a DETR detector on text. Grounding DINO (Liu et al., 2023) aligns a DINO detector with language for open-set detection. Florence-2 (Xiao et al., 2024) handles grounding together with other tasks in a sequence-to-sequence model. These methods predict boxes with a regression head and are fast at inference.

**Grounding with VLMs.** Qwen2-VL (Wang et al., 2024) introduced dynamic input resolution and output coordinates on a 0–1000 grid. Qwen2.5-VL switched to absolute pixel coordinates of the resized input. Qwen3-VL went back to relative coordinates and added interleaved-MRoPE and DeepStack, which feeds features from several ViT layers into the language model. Qwen3.5 is an early-fusion model with an optional thinking mode. Because the coordinate format differs between versions, an evaluation script has to know which one a model uses; Section 3.2 describes how we handle this.

**Vision encoders.** The vision towers of these models come from language-supervised encoders such as CLIP and SigLIP 2. Self-supervised encoders such as DINOv2 and DINOv3 are known to give better dense features. Our LoRA scope ablation (Section 4.4a) asks a related question: when grounding fails on a new domain, is the vision side or the language side the one that needs to change?

**Post-training for grounding.** EGM (NVIDIA, 2026) fine-tunes Qwen3-VL with SFT and reinforcement learning and raises the 8B model's RefCOCO average from 87.8 to 91.4. Our LoRA experiments are a much smaller version of the same idea.

**Datasets.** DIOR-RSVG (Zhan et al., 2023) contains 38,320 expressions over 800×800 remote-sensing images with 20 object classes. ScreenSpot (Cheng et al., 2024) contains 1,272 screenshots of desktop, web and mobile interfaces, each paired with an instruction that refers to one UI element.

## 3 Method

### 3.1 Model

Qwen3-VL consists of a ViT encoder with 16-pixel patches, a merger that turns each 2×2 group of patches into one visual token, and a decoder-only language model. Images keep their native resolution within a pixel budget, so an 800×800 image becomes 625 visual tokens. We use the 4B and 8B Instruct checkpoints.

### 3.2 Grounding as text generation

We use the prompt from the Qwen3-VL cookbook: `Locate the "<expression>" in the image and output its bbox coordinates in JSON format.` The model typically answers with `[{"bbox_2d": [122, 539, 382, 808], "label": "..."}]`. The parser takes the first box it finds, trying JSON first and then two regular-expression patterns.

To read the numbers correctly, the script tests three interpretations on the first 32 samples of each run: coordinates on a 0–1000 grid, absolute pixels in the resized input, and absolute pixels in the original image. It keeps the one with the highest mean IoU. For Qwen3-VL the three scores are 0.854, 0.094 and 0.094; for Qwen2.5-VL the absolute-pixel interpretation wins with 0.89. Each run also records the image size the model actually received, which is how we caught one case where a resolution setting had no effect.

### 3.3 Evaluation protocol

The main metric is Acc@0.5, the fraction of expressions whose predicted box has IoU ≥ 0.5 with the ground truth, as in the paper. We also report Acc@0.75, mean IoU, the rate of unparseable answers and the time per sample. Each split is evaluated on 500 expressions sampled with a fixed seed.

With 500 samples, a single accuracy has a standard error of about 2.2 points, so small differences between two runs are hard to read on their own. Because all runs on a dataset use the same 500 samples, we compare them pairwise: we report a bootstrap 95% confidence interval for the accuracy difference, the McNemar p-value, and the number of samples that went from wrong to right (N₊) and from right to wrong (N₋). We also split every run by target size (small: under 1% of the image; medium: 1–10%; large: over 10%), by expression length, and by whether the expression contains a spatial word such as left, right, middle or below.

### 3.4 LoRA adaptation

We attach LoRA adapters (Hu et al., 2022) to the linear layers of one of three scopes: the language model only (**LLM**, 252 layers, 33 M parameters at rank 16, 0.74% of the model), the vision encoder and merger (**vision**), or both (**all**). The default setting is rank 16, α = 32, dropout 0.05, learning rate 1e-4 with cosine decay and 3% warm-up, effective batch size 8, bf16 and one epoch. The training target is the same JSON answer the model produces at inference, with the ground-truth box written in the model's own coordinate format, and the loss covers only the answer tokens. Training data come from the DIOR-RSVG training split.

### 3.5 Datasets

| Dataset | Domain | Role | Used here | Source |
|---|---|---|---|---|
| RefCOCO / RefCOCO+ / RefCOCOg | natural images (COCO) | reproduction; forgetting check | 8 splits × 500 expressions | lmms-lab Hugging Face uploads |
| DIOR-RSVG | remote sensing, 800×800, 20 classes | out-of-domain test; LoRA training | test: 500; train: 300–3,000 | Zhan et al. (2023), official splits |
| ScreenSpot | GUI screenshots | out-of-domain test | test: 500 | Cheng et al. (2024) |
| Synthetic shapes | coloured shapes on plain background | pipeline check | 106 | generated by us |

The synthetic set has one uniquely described shape per expression ("the red square"). The 4B model scores 1.00 on it, which confirms that parsing and coordinate handling work before we look at real data.

### 3.6 Implementation

PyTorch with CUDA 12.8, `transformers` 5.22 and `peft`. Zero-shot evaluation takes 0.35–0.43 s per sample with batch size 4. Training a LoRA adapter on 3,000 samples takes 13 minutes at 800² and 29 minutes at 1024². Scripts and per-run results are in the repository (Appendix A).

## 4 Experiments and Analysis

### 4.1 Reproduction (RQ1)

Table 1 lists our accuracy on each RefCOCO split. The paper only reports the average over the eight splits, which we place in the last columns. Our averages are 1.3 points (4B) and 1.0 point (8B) below the paper. The two models are almost tied in both our results and the paper's. The gap is similar for both models, so it is likely to come from a shared difference in setup, such as prompt wording or decoding settings, rather than from noise. RefCOCO+ testB is the hardest split for both models; this split forbids location words in the expressions and its targets are objects other than people.

**Table 1.** Acc@0.5 (%) on 500-expression subsets of each split. Paper numbers are from Table 4 of the Qwen3-VL report.

| Model | RC val | RC testA | RC testB | RC+ val | RC+ testA | RC+ testB | RCg val | RCg test | Avg (ours) | Avg (paper) | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B-Instruct | 92.6 | 91.6 | 88.4 | 84.2 | 89.2 | 81.6 | 88.4 | 85.8 | 87.7 | 89.0 | −1.3 |
| Qwen3-VL-8B-Instruct | 92.2 | 91.6 | 88.4 | 86.8 | 90.4 | 79.8 | 87.8 | 87.8 | 88.1 | 89.1 | −1.0 |

Unparseable answers are rare (0–0.4% per split). The 8B model is only slightly slower than 4B (0.38–0.43 vs. 0.35–0.41 s per sample), since the answers are short.

We did not run the Thinking checkpoints, which generate long reasoning chains and take 10–40 s per sample. The paper reports them slightly below the Instruct versions (88.2 vs. 89.0 for 4B, 88.2 vs. 89.1 for 8B).

### 4.2 Out-of-domain generalisation (RQ2)

**Remote sensing.** On DIOR-RSVG the 4B model reaches 53.6, 39 points below its RefCOCO val score. Table 2 shows how the errors are distributed. Accuracy falls with target size (75.0 for large, 41.7 for small) and with expression length (70.3 for three words or fewer, 42.9 for eight or more). Expressions that contain spatial words are harder than those that do not (50.5 vs. 64.0), while on RefCOCO they are slightly easier (93.7 vs. 90.7). On RefCOCO a spatial word usually separates two or three distinct objects. On DIOR-RSVG it often has to pick one airplane out of a row of identical airplanes, and many expressions are relational templates like "the vehicle that is a little smaller than the vehicle on the lower left".

**Table 2.** Qwen3-VL-4B-Instruct, zero-shot, Acc@0.5 by subgroup (number of samples in parentheses).

| Dataset | All | Small | Medium | Large | No spatial | With spatial | ≤3 w | 4–7 w | ≥8 w |
|---|---|---|---|---|---|---|---|---|---|
| RefCOCO val | 92.6 (500) | – (0) | 89.1 (137) | 93.9 (363) | 90.7 (182) | 93.7 (318) | 95.1 (304) | 89.5 (171) | 84.0 (25) |
| DIOR-RSVG test | 53.6 (500) | 41.7 (223) | 55.6 (169) | 75.0 (108) | 64.0 (114) | 50.5 (386) | 70.3 (37) | 57.0 (307) | 42.9 (156) |

**GUI screenshots and model generations.** Table 3 compares four models from three generations on all three datasets, with the same prompt, parser and samples. Qwen3.5-9B runs with thinking turned off.

**Table 3.** Acc@0.5 (%). In parentheses: paired difference to Qwen3-VL-8B with 95% CI; * marks intervals that exclude zero. The last two columns refer to ScreenSpot.

| Model | Coordinates | RefCOCO val | DIOR-RSVG test | ScreenSpot | Unparseable | s/sample |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | absolute (input) | 90.0 (−2.2 [−4.8, +0.2]) | 41.6 (−14.2 [−18.2, −10.2]*) | 26.6 (−31.2 [−36.0, −26.6]*) | 2.2% | 2.5 |
| Qwen3-VL-4B-Instruct | relative 0–1000 | 92.4 (+0.2 [−1.8, +2.2]) | 53.6 (−2.2 [−5.6, +1.2]) | 56.0 (−1.8 [−5.4, +2.0]) | 10.0% | 1.2 |
| Qwen3-VL-8B-Instruct | relative 0–1000 | 92.2 | 55.8 | 57.8 | 7.6% | 2.0 |
| Qwen3.5-9B (no thinking) | relative 0–1000 | 92.4 (+0.2 [−1.8, +2.4]) | 57.2 (+1.4 [−2.0, +4.8]) | 61.6 (+3.8 [−0.4, +8.2]) | 0.0% | 4.1 |

On RefCOCO all four models are within two points of each other. On the two new datasets the older Qwen2.5-VL falls far behind: 14 points lower than Qwen3-VL-8B on DIOR-RSVG and 31 points lower on ScreenSpot. The 4B and 8B versions of Qwen3-VL, by contrast, differ by about two points, and the difference is not significant on either dataset. Qwen3.5-9B has the highest scores on both new datasets, but its lead over Qwen3-VL-8B is within the confidence interval, and it is two to four times slower.

ScreenSpot is difficult for different reasons than DIOR-RSVG. The screenshots are large (median 2528×1184 after resizing), most targets are small icons or text labels, and few instructions contain spatial words. Qwen3-VL also returns no box for 7.6–10% of ScreenSpot instructions; Section 4.3 looks at why.

### 4.3 Hyper-parameter analysis (RQ3)

**Input resolution.** Qwen3-VL lets the user set the pixel budget per image. We resized the 800×800 DIOR-RSVG images to 448², 1024² and 1280², giving 196, 1,024 and 1,600 visual tokens instead of 625 (Table 4, Figure 1). Upscaling to 1024² improves accuracy by 3.4 points, a significant difference (31 samples fixed, 14 broken), and most of the gain is on small and large targets. At 1280² accuracy goes back down to the native level, and at 448² it drops by two points, which is within noise. The subgroup with spatial words stays between 49 and 54 across all four settings. Resolution therefore helps with small objects, but it does not address the relational expressions that cause most errors.

**Table 4.** Resolution sweep, Qwen3-VL-4B-Instruct on DIOR-RSVG test (n = 500). Δ is the paired difference to native resolution.

| Input | Tokens | Acc@0.5 | Acc@0.75 | Small | With spatial | s/sample | Δ [95% CI], p |
|---|---|---|---|---|---|---|---|
| 448² | 196 | 51.6 | 32.6 | 37.2 | 49.7 | 0.37 | −2.0 [−5.2, +1.2], 0.28 |
| 800² (native) | 625 | 53.6 | 35.4 | 41.7 | 50.5 | 0.39 | — |
| 1024² | 1024 | **57.0** | 37.4 | **44.4** | 53.9 | 0.76 | **+3.4 [+0.8, +6.0], 0.016** |
| 1280² | 1600 | 52.6 | 37.0 | 42.2 | 49.2 | 0.84 | −1.0 [−3.8, +1.8], 0.58 |

![Figure 1](figures/fig_resolution.png)
*Figure 1. Acc@0.5 against input resolution on DIOR-RSVG, for all targets, small targets and expressions with spatial words.*

**Model size.** The 8B model is not more accurate than the 4B model on any of the three datasets (Table 1 and Table 3). It produces slightly tighter boxes on RefCOCO (Acc@0.75 84.2 vs. 82.8) and fewer unanswered ScreenSpot instructions.

**Generation budget.** All runs use greedy decoding with at most 128 new tokens. We first suspected that the missing answers on ScreenSpot were cut off by this limit, so we re-ran the 4B model with 512 tokens. The results did not change at all: same accuracy, no sample changed its outcome. Reading the 50 unanswered cases explained why. Every one of them is a refusal, such as "There are none." or "There is no object matching the query in the image.", given for instructions like "file processing", "adjust the volume", "check my apple id" and "siri". These instructions name what a UI element does rather than what it looks like, and Qwen3-VL treats them as descriptions of objects that are not in the picture. Qwen2.5-VL refuses in 2.2% of cases and Qwen3.5 never does, so the behaviour comes from how each checkpoint was trained. Refusals account for roughly half of the 5.6-point gap between Qwen3-VL-4B and Qwen3.5-9B on ScreenSpot. We did not try other prompt wordings; a prompt that asks the model to always return a box would be a natural next experiment.

### 4.4 Ablation of the adaptation recipe (RQ4)

The architectural components of Qwen3-VL, such as DeepStack and MRoPE, cannot be retrained on a single GPU, so this section ablates the components of a LoRA adaptation instead: which part of the model is adapted, the rank, the amount of training data, the number of training steps and the resolution. All runs fine-tune Qwen3-VL-4B-Instruct on DIOR-RSVG training data and are evaluated on the same 500 test expressions. The zero-shot baseline here is 54.0 (re-evaluated with batch size 2; Table 2 used batch size 4 and gives 53.6).

**(a) Which part to adapt.** Table 5 compares adapters on the language model, on the vision encoder, and on both (rank 16, 3,000 samples, one epoch). The language-model adapter raises Acc@0.5 by 18.2 points; it fixes 104 test samples and breaks 13. The vision adapter adds 10.8 points. Adapting both parts gives the same Acc@0.5 as the language-model adapter alone (72.6 vs. 72.2, p = 0.90), but tighter boxes (Acc@0.75 61.0 vs. 56.8). On RefCOCO, which the adapters never saw, every adapter costs some accuracy, and adapting both parts costs twice as much as adapting one (4.4 vs. about 2.3 points). Figure 2 shows where the language-model adapter helps most: expressions with spatial words go from 50.8 to 70.7 and long expressions from 42.9 to 61.5. In other words, it mainly changes which object the model selects.

**Table 5.** LoRA scope (DIOR-RSVG test, n = 500). The last column is Acc@0.5 on RefCOCO val after adaptation (92.6 before).

| Scope | Acc@0.5 | Acc@0.75 | Small | With spatial | ≥8 words | Δ vs. zero-shot [CI] | RefCOCO val (Δ, p) |
|---|---|---|---|---|---|---|---|
| zero-shot | 54.0 | 35.8 | 41.7 | 50.8 | 42.9 | — | 92.6 |
| LLM | 72.2 | 56.8 | 54.7 | 70.7 | 61.5 | +18.2 [+14.4, +22.2] | 90.2 (−2.4, p=0.017) |
| vision | 64.8 | 54.0 | 51.6 | 64.0 | 51.9 | +10.8 [+7.0, +14.6] | 90.4 (−2.2, p=0.027) |
| all | 72.6 | **61.0** | 55.6 | 70.5 | 59.6 | +18.6 [+14.6, +22.8] | 88.2 (−4.4, p=0.0002) |

![Figure 2](figures/fig_buckets.png)
*Figure 2. Acc@0.5 by subgroup for the zero-shot model and the LLM and all-scope adapters on DIOR-RSVG.*

**(b) Rank.** With the language-model scope, ranks 4, 16 and 64 reach 67.0, 72.2 and 73.8. Going from 4 to 16 is a significant improvement (+5.2, p = 0.0001). Going from 16 to 64 adds 1.6 points, which is not significant, while the number of adapter parameters grows from 33 M to 132 M.

**(c) Training data and training steps.** Training on 300, 1,000 and 3,000 samples for one epoch gives 61.0, 61.4 and 72.4. At first this looks like more data helps, but with one epoch, three times the data also means three times the optimiser steps. To separate the two, we trained on 1,000 samples for three epochs, which gives the same 375 steps as 3,000 samples for one epoch. This run reaches 70.2, 8.8 points above one epoch on the same data (p < 0.0001) and only 2.0 points below the 3,000-sample run, a difference that is not significant (p = 0.16). Training on 3,000 samples for three epochs (1,125 steps) gives the best result in the study, 75.4. Table 6 and Figure 3 summarise these runs. In our range, more training steps help much more than more data.

**Table 6.** Training data and steps (LLM scope, rank 16, DIOR-RSVG test).

| Train samples | Epochs | Steps | Acc@0.5 | Acc@0.75 | Small | With spatial |
|---|---|---|---|---|---|---|
| 300 | 1 | 38 | 61.0 | 45.6 | 46.2 | 57.8 |
| 1,000 | 1 | 125 | 61.4 | 47.4 | 42.6 | 58.5 |
| 1,000 | 3 | 375 | 70.2 | 55.2 | 52.5 | 69.9 |
| 3,000 | 1 | 375 | 72.4 | 56.0 | 54.7 | 70.5 |
| 3,000 | 3 | 1,125 | **75.4** | **60.2** | **58.7** | **74.1** |

![Figure 3](figures/fig_steps.png)
*Figure 3. Acc@0.5 against the number of optimisation steps. At 375 steps, 1,000 and 3,000 training samples give almost the same result.*

**(d) Resolution and LoRA together.** Training and testing the language-model adapter at 1024² instead of 800² gives 74.2 instead of 72.2 and the best small-object accuracy among the one-epoch runs (57.8). The difference is in the expected direction but not significant (p = 0.19), and training takes 2.2 times as long.

**(e) Run-to-run variation.** The same configuration (LLM scope, rank 16, 3,000 samples, one epoch) was trained three times as part of different sweeps and reached 72.2, 72.2 and 72.4. Differences of under one point between configurations should therefore not be read as real.

### 4.5 Visualisation and case analysis

**Good cases.** On RefCOCO, predictions are close to the ground truth even for expressions that combine an attribute and a position, such as "white pot upper right corner" (`figures/refcoco_good.jpg`). On DIOR-RSVG the model does well when the target is a large, unique structure: bridges, running tracks, runways and tennis courts are found with IoU of 0.98 or higher (Figure 4). It also handles a relation between two large objects correctly ("the ground track field on the upper right of the baseball field on the lower left"). So the model can read remote-sensing images; its problems start when it has to choose between many similar small objects.

![Figure 4](figures/dior_good.jpg)
*Figure 4. Good cases on DIOR-RSVG (Qwen3-VL-8B, zero-shot). Green: ground truth. Red: prediction.*

**Failure cases on DIOR-RSVG.** Looking through the lowest-scoring zero-shot predictions and the before/after comparison for the LoRA adapters, three kinds of errors come up repeatedly.

1. Boxing the reference object. Many DIOR-RSVG expressions describe a target relative to another object, for example "the overpass that is similar in size to the overpass in the middle". The zero-shot model often returns the box of the object used as the reference. This is the error the language-model adapter fixes most often (Figure 5).
2. Picking the wrong instance. In a row of identical airplanes or a pair of chimneys, only the spatial phrase identifies the target, and the model often picks a neighbour.
3. Very small targets. Vehicles and wind turbines a few tens of pixels wide are often missed by every configuration. This subgroup gains the most from higher resolution and from adapting the vision encoder, and it still has the lowest accuracy (58.7 at best).

![Figure 5](figures/dior_lora_fixed_top4.jpg)
*Figure 5. DIOR-RSVG samples fixed by the LLM adapter (4 of 104). Columns: zero-shot, LLM adapter, all-scope adapter. Full grid in figures/dior_lora_fixed.jpg.*

**Samples broken by adaptation.** The language-model adapter turns 13 correct zero-shot answers into wrong ones. Most of them are nested relations such as "the chimney on the right of the chimney on the left", where the adapted model now picks the wrong side. The all-scope adapter gets six of the eight examples in Figure 6 right again, which fits its higher Acc@0.75.

![Figure 6](figures/dior_lora_broken_top4.jpg)
*Figure 6. Samples that the zero-shot model gets right and the LLM adapter gets wrong (4 of 13). The right column shows the all-scope adapter. Full grid in figures/dior_lora_broken.jpg.*

**Failure cases on ScreenSpot.** Apart from the refusals described in Section 4.3, the ScreenSpot errors of Qwen3-VL-8B fall into three groups (Figure 7). In some cases the model picks a related element: for "open account protection" it boxes the Sign in button inside the account protection panel instead of the sidebar entry, and for "take a photo to insert into the memo" it boxes the camera button that opens the menu instead of the Take Photo item. In others it picks a neighbouring icon that looks similar, such as the flip icon for "rotate left". And for very small targets it may choose the wrong object altogether, such as a desktop folder instead of the Finder icon in the dock. The first group is specific to interfaces, where an instruction can reasonably match more than one element. The other two are the same instance and size problems seen on DIOR-RSVG.

![Figure 7](figures/screenspot_bad.jpg)
*Figure 7. ScreenSpot errors of Qwen3-VL-8B. From left: related element, related element, similar neighbouring icon, very small target.*

**Comparing generations on the same samples.** Figure 8 takes the hardest DIOR-RSVG samples for Qwen3-VL-8B and shows what Qwen2.5-VL-7B and Qwen3.5-9B predict on the same images (the RefCOCO version is in `figures/generations_refcoco.jpg`). Since the samples are chosen as failures of the 8B model, the other models sometimes get them right. Overall, though, Qwen2.5-VL fixes 23 of the 8B model's DIOR-RSVG errors and makes 94 new ones. Some samples are hard for all three models: for "the bridge on the upper right of the blue expressway toll station on the left" each model boxes a different wrong region, and for a vehicle "on the upper right of the baseball field" each picks a different small vehicle.

![Figure 8](figures/generations_dior_top3.jpg)
*Figure 8. Hardest DIOR-RSVG samples for Qwen3-VL-8B (left), with the predictions of Qwen2.5-VL-7B (middle) and Qwen3.5-9B (right). Full grids in figures/generations_dior.jpg and figures/generations_refcoco.jpg.*

### 4.6 Advantages and disadvantages

**Advantages.**

- The model grounds free-form language, including attributes and relations, without a fixed list of categories.
- The same model works on natural images, remote-sensing images and screenshots without changes to the architecture.
- Adapting it to a new domain is cheap. A LoRA adapter with 0.74% of the parameters, trained for 13 minutes, adds 18 points on DIOR-RSVG, and adapters for different domains can be swapped at inference time.
- Because the answer is text, it can be read directly, which makes errors easy to inspect.

**Disadvantages.**

- Writing coordinates as text is slow compared with a detector (0.35–0.8 s per sample here), and box precision is limited: Acc@0.75 is 10 to 20 points below Acc@0.5.
- Choosing between many similar objects based on a spatial description does not transfer well to new domains.
- Small targets remain hard, and more pixels help only up to about 1024².
- Fine-tuning reduces accuracy on the original domain by 2 to 4 points, and short training runs under-fit.
- The model sometimes refuses to answer. This is correct when the object is absent, but on ScreenSpot it also happens for 10% of valid instructions, and the metric cannot tell the two situations apart.

## 5 Conclusion

We reproduced the RefCOCO results of Qwen3-VL within about one point and then tested the model on remote-sensing images and GUI screenshots. On remote-sensing images accuracy drops by 39 points, mainly because of relational expressions over many similar small objects. Higher input resolution helps small objects a little and stops helping beyond 1024². A LoRA adapter on the language model recovers most of the lost accuracy; adapting the vision encoder as well gives tighter boxes but doubles the loss on RefCOCO. Among the training settings, the number of optimisation steps has the largest effect. Across model versions, the newer generations handle the new domains much better than the older one, while the larger model of the same generation does not.

**Limitations.** Each split is evaluated on 500 samples rather than the full set. Most LoRA configurations were trained once. We did not include a detection-based model such as Grounding DINO for comparison, and we did not run the Thinking checkpoints. The error categories in Section 4.5 come from looking at a few dozen examples, not from a full annotation.

**Future work.** Reinforcement-learning fine-tuning as in EGM; training on the full 27,000-expression DIOR-RSVG training set with more steps; combining vision-encoder adaptation with higher resolution for small objects; and mixing RefCOCO data into training to reduce forgetting.

## References

- Qwen Team. *Qwen3-VL Technical Report.* arXiv:2511.21631, 2025.
- Qwen Team. *Qwen3.5.* https://qwen.ai/blog?id=qwen3.5, 2026.
- Wang et al. *Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution.* 2024.
- Kazemzadeh et al. *ReferItGame.* EMNLP 2014; Mao et al. *Generation and Comprehension of Unambiguous Object Descriptions.* CVPR 2016 (RefCOCO/+/g).
- Zhan, Xiong, Yuan. *RSVG: Exploring Data and Models for Visual Grounding on Remote Sensing Data.* IEEE TGRS 2023.
- Cheng et al. *SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents.* 2024 (ScreenSpot).
- Kamath et al. *MDETR.* ICCV 2021. Liu et al. *Grounding DINO.* ECCV 2024. Xiao et al. *Florence-2.* CVPR 2024.
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Zhan et al. *EGM: Efficient Visual Grounding Language Models.* NVIDIA, 2026.
- Radford et al. *CLIP*, 2021; Tschannen et al. *SigLIP 2*, 2025; Oquab et al. *DINOv2*, 2024; Siméoni et al. *DINOv3*, 2025.

## Appendix A: Scripts and results

| Section | Script | Outputs |
|---|---|---|
| 4.1 | `experiments/p0_reproduce.ps1` | `runs/table_p0_vs_paper.md`, `runs/summary_p0.md` |
| 4.2, 4.3 (model size), 4.5 | `experiments/p1_ood_and_generations.ps1` | `runs/summary_p1.md`, `runs/compare_p1_*.txt` |
| 4.3 (resolution) | `experiments/p2_hyperparams.ps1` | `runs/summary_dior_px.md`, `runs/compare_resolution.txt` |
| 4.3 (generation budget) | `scripts/eval_grounding.py --max-new-tokens 512` | `runs/compare_screenspot_tokens.txt` |
| 4.4 | `experiments/p3_lora_ablation.ps1` and the extra runs listed in the README | `runs/summary_p3_dior.md`, `runs/summary_p3_forgetting.md`, `runs/compare_p3_*.txt`, `runs/compare_steps.txt`, `runs/compare_rank.txt` |
| Figures | `experiments/p4_visualize.ps1`, `tools/plot_curves.py` | `docs/figures/` |

Settings kept fixed: prompt template `json`, greedy decoding, at most 128 new tokens, bf16, SDPA attention; LoRA α = 2 × rank, dropout 0.05, learning rate 1e-4, cosine schedule, 3% warm-up, effective batch size 8; evaluation subsets drawn with seed 0.

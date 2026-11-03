# Report skeleton — Visual Grounding with Qwen-VL: Reproduction, Out-of-Domain Analysis and LoRA Ablation

> 用法：每一节给出 (a) 要回答的问题、(b) 要点、(c) 表/图来源文件、(d) 已经确定的数字。
> `[P0]` / `[P1]` / `[P0b]` 处等今晚的 run 填。所有数字来自 `runs/*/metrics.json`，统计检验来自 `runs/compare_*.txt`。
> 措辞模板："X 比 Y 高 Δ 个百分点（配对 bootstrap 95% CI [a, b]，McNemar p = c；N₊ 个样本由错变对、N₋ 个由对变错）"。
> 目标 6–8 页，格式 Introduction / Method / Experiments and Analysis / Conclusion。

---

## 1. Introduction（0.5–1 页）

- 任务：指代表达理解（REC）：输入图像 + 一句自然语言，输出目标框；指标 Acc@IoU≥0.5。
- 背景：专用检测式方法（MDETR、Grounding DINO、Florence-2）vs 通用 VLM 以文本形式输出坐标（Qwen2-VL → Qwen2.5-VL → Qwen3-VL → Qwen3.5）。
- 选 Qwen3-VL 的理由：开源、2B–8B 可单卡运行、报告第 5.5 节给出 RefCOCO/+/g 基准数字（Table 4）。
- 本文贡献（4 条）：
  1. 在单张 RTX 5090 上复现 Qwen3-VL-4B/8B 的 RefCOCO 家族结果（8 个 split）`[P0]`；
  2. 两个原论文之外的域外数据集（遥感 DIOR-RSVG、GUI 截图 ScreenSpot）+ 三代模型同协议对比 `[P1]`；
  3. 超参分析（视觉 token 预算 / 分辨率、规模）与消融（Thinking 开关、LoRA 作用域 / rank / 训练数据量 / 训练步数、遗忘）；
  4. 带配对统计检验的 good/bad case 分析，归纳出三类失败模式。
- 一句话主结论（写最后）：零样本 VLM 在遥感域掉 ~39 个点；LLM 侧 LoRA 修复大部分关系推理错误（+18），优化步数比数据量 / rank / 分辨率 / 作用域更关键；分辨率只对小目标有限帮助。

## 2. Related Work / Review（1 页）

- 2.1 专用 REC 模型：MDETR（端到端调制检测）、Grounding DINO（开放词表检测 + 文本对齐）、Florence-2（统一序列到序列）。特点：框回归头、训练数据专用、推理快。
- 2.2 通用 VLM 的 grounding：坐标作为文本 token 输出；Qwen2-VL（0–1000 归一化坐标、动态分辨率 + M-RoPE）、Qwen2.5-VL（绝对像素坐标）、Qwen3-VL（回到相对坐标；interleaved-MRoPE、DeepStack 多层 ViT 特征注入、文本时间戳对齐）、Qwen3.5（统一多模态早融合、混合推理）。InternVL 一笔带过。
- 2.3 视觉编码器预训练（呼应课程 "Foundation models and feature learning"）：CLIP / SigLIP 2 的语言监督；DINOv2/v3 的自监督稠密特征；VLM 的视觉塔通常源自前者。
- 2.4 后训练提升 grounding：EGM（在 Qwen3-VL 上 SFT + GRPO，RefCOCO-avg 87.8 → 91.4），说明基座仍有提升空间；本文的 LoRA 是其轻量版。
- 2.5 遥感 / GUI grounding：DIOR-RSVG（Zhan et al., TGRS 2023）、ScreenSpot（Cheng et al., 2024）。

## 3. Method（1–1.5 页）

- 3.1 Qwen3-VL 架构：ViT（patch 16）→ 2×2 patch merger → LLM；DeepStack；MRoPE。只写与 grounding 相关的部分。
- 3.2 grounding 形式化：提示词 `Locate the "<expr>" in the image and output its bbox coordinates in JSON format.`；输出 `[{"bbox_2d":[x1,y1,x2,y2],"label":...}]`；坐标约定与**自动校准**：在前 32 个样本上对 relative_1000 / absolute_input / absolute_original 三种解释各算平均 IoU 取最优（Qwen3-VL 校准得分 0.85 vs 0.09，无歧义）。
- 3.3 评测协议：Acc@0.5（与论文一致）、Acc@0.75、mIoU、解析失败率、每样本延迟；固定种子的嵌套随机子集（limit=100 ⊂ 500），使不同 run 可配对；分桶：目标面积（<1% / 1–10% / >10% 图像面积）、表达式长度、是否含空间词。
- 3.4 统计检验：同一样本集上的配对 bootstrap 95% CI + 精确 McNemar；n=500 时单个 Acc 的标准误约 ±2.2 个点。
- 3.5 LoRA 设置：PEFT，r/α = 16/32（消融 4/8、64/128），dropout 0.05，lr 1e-4 cosine，warmup 3%，bf16，梯度检查点，有效 batch 8；作用域 llm（252 个 Linear，0.74% 参数）/ vision（视觉塔 + merger）/ all；监督目标为同一 JSON 格式、相对坐标。
- 3.6 实现：transformers 5.22 + PEFT，单张 RTX 5090（32 GB），Windows；代码仓库 github.com/Ellawjjjj/90-individual。

## 4. Experiments and Analysis（3–4 页）

### 4.1 数据集与实现细节
| 数据集 | 域 | 用途 | 规模（本文使用） | 来源 |
|---|---|---|---|---|
| RefCOCO / RefCOCO+ / RefCOCOg | 自然图像（COCO） | 复现、遗忘回测 | 8 个 split × 500 条随机子集 | lmms-lab HF 上传 |
| DIOR-RSVG | 遥感（800×800，20 类） | 域外评测 + LoRA 训练 | test 500 条；train 300–3000 条 | Zhan et al. 2023 |
| ScreenSpot | GUI 截图 | 域外评测 | test 500 条 | Cheng et al. 2024 |
| Toy（合成几何） | — | 流水线验证 | 106 条 | 自建（Acc@0.5 = 1.00） |

- 吞吐：4B-Instruct 原生分辨率 batch 4 约 0.35–0.39 s/样本；1024² 约 0.75 s；LoRA 训练 3000 条 1 epoch 13 分钟（800²）/ 29 分钟（1024²）。

### 4.2 复现（`runs/table_p0_vs_paper.md`，`runs/summary_p0.md`）`[P0]`
- 表：模型 × 8 个 split 的 Acc@0.5 + 平均，与论文 Table 4 的 RefCOCO-avg（4B 89.0 / 8B 89.1）并列。已知：4B-Instruct RefCOCO val = 92.6。
- 讨论：子集 vs 全集、val 通常高于 8 split 平均、与论文差距的来源（提示词、采样、解析）。
- 规模：4B vs 8B 配对检验（`runs/compare_scale_refcoco.txt`、`compare_scale_dior.txt`）。
- `[P0b]`（可选）Thinking vs Instruct 在同 100 个样本上：论文中 thinking 反而略低（88.2 vs 89.0）；给出我们的配对结果和推理链片段。

### 4.3 域外泛化（`runs/summary_p1.md`，`compare_p1_*.txt`）`[P1]`
- 表：{Qwen2.5-VL-7B, Qwen3-VL-4B, Qwen3-VL-8B, Qwen3.5-9B} × {RefCOCO val, DIOR test, ScreenSpot}。
- 已知：Qwen3-VL-4B RefCOCO 0.926 → DIOR 0.536（−39 个点）；DIOR 上 small 0.417 / medium 0.556 / large 0.750；带空间词 0.505 vs 不带 0.640（RefCOCO 上相反：0.937 vs 0.907）；长表达式 0.429。
- 要点：域偏移体现为"小目标 + 同类密集 + 关系型表达式"三重困难；ScreenSpot 考察另一种域（图标 / 文字元素）。

### 4.4 超参分析
**(a) 视觉 token 预算 / 分辨率**（DIOR，4B 零样本；`runs/summary_dior_px.md`，`compare_resolution.txt`）

| 输入 | token 数 | Acc@0.5 | Acc@0.75 | small | 带空间词 | s/样本 | vs 原生 |
|---|---|---|---|---|---|---|---|
| 448² | 196 | 0.516 | 0.326 | 0.372 | 0.497 | 0.37 | −2.0 [−5.2, +1.2], p=0.28 |
| 800²（原生） | 625 | 0.536 | 0.354 | 0.417 | 0.505 | 0.39 | — |
| 1024² | 1024 | **0.570** | 0.374 | **0.444** | 0.539 | 0.76 | **+3.4 [+0.8, +6.0], p=0.016** |
| 1280² | 1600 | 0.526 | 0.370 | 0.422 | 0.492 | 0.84 | −1.0 [−3.8, +1.8], p=0.58 |

要点：适度上采样显著但有限；继续上采样无收益；下采样损失在噪声内 → 模型对分辨率不敏感，瓶颈不在视觉清晰度而在关系推理（带空间词桶在所有分辨率下都在 0.49–0.54）。

**(b) 模型规模**：4B vs 8B（RefCOCO、DIOR）`[P0/P1]`。
**(c) 其他**（若时间允许）：温度、提示词模板、4-bit（`p2_hyperparams.ps1` 第 2–4 段）。没跑就在 limitation 里说明。

### 4.5 消融（DIOR test，n=500；`runs/summary_p3_dior.md`，`compare_p3_*.txt`，`compare_steps.txt`，`compare_rank.txt`）

**(a) 哪一侧在适应（LoRA 作用域，r=16，3000 条，1 epoch）**

| 作用域 | 可训练参数 | Acc@0.5 | Acc@0.75 | small | 带空间词 | 长表达式 | RefCOCO 回测 |
|---|---|---|---|---|---|---|---|
| 零样本 | — | 0.540 | 0.358 | 0.417 | 0.508 | 0.429 | 0.926 |
| LLM | 0.74% | 0.722 | 0.568 | 0.547 | 0.707 | 0.615 | 0.902 |
| 视觉 | — | 0.648 | 0.540 | 0.516 | 0.640 | 0.519 | 0.904 |
| 全部 | — | 0.726 | **0.610** | 0.556 | 0.705 | 0.596 | 0.882 |

检验：三者 vs 零样本均显著（LLM +18.2 [+14.4, +22.2]，修好 104 / 弄坏 13）；全部 vs LLM +0.4 [−2.8, +3.4], p=0.90 → Acc@0.5 持平；遗忘：LLM −2.4 [−4.2, −0.6] p=0.017，视觉 −2.2 [−4.0, −0.4] p=0.027，全部 −4.4 [−6.6, −2.2] p=0.0002。
解读：LLM 侧决定"选对哪个目标"（空间词 / 长表达式桶的提升几乎全来自它）；视觉侧决定"框得多紧"（Acc@0.75）；双侧微调框更紧但遗忘翻倍。

**(b) rank**：4 → 16 → 64 = 0.670 → 0.722 → 0.738；4→16 显著（+5.2 [+2.8, +7.8], p=0.0001），16→64 不显著（+1.6 [−1.0, +4.2], p=0.30）。

**(c) 数据量 vs 训练步数**（关键发现）

| 训练数据 | epoch | 步数 | Acc@0.5 | Acc@0.75 | small | 带空间词 |
|---|---|---|---|---|---|---|
| 300 | 1 | 38 | 0.610 | 0.456 | 0.462 | 0.578 |
| 1000 | 1 | 125 | 0.614 | 0.474 | 0.426 | 0.585 |
| 1000 | 3 | 375 | 0.702 | 0.552 | 0.525 | 0.699 |
| 3000 | 1 | 375 | 0.724 | 0.560 | 0.547 | 0.705 |
| 3000 | 3 | 1125 | **0.754** | **0.602** | **0.587** | **0.741** |

检验：步数 125→375（同 1000 条）+8.8 [+5.6, +12.2] p<0.0001；同 375 步下 3000 vs 1000 条 +2.0 [−0.6, +4.6] p=0.16（不显著）；375→1125 步 `[compare_steps.txt]`。
解读：在这个规模下优化步数主导，数据多样性的边际收益小；1 epoch 配方欠训练。

**(d) 分辨率 × LoRA 可叠加性**：LoRA-LLM @1024 = 0.742（small 0.578）vs @800 = 0.722，+2.0 [−0.6, +4.6] p=0.19（趋势一致、未达显著）；训练慢 2.2 倍。

**(e) 复现性**：同一配置三次独立训练（p3_lora_llm / p3_rank_16 / p3_data_3000）Acc@0.5 = 0.722 / 0.722 / 0.724，说明 ≥2 个点的差距可信、<1 个点不可信。

**(f) Thinking 开关**：`[P0b]`，没跑则引用论文 Table 4 并说明。

### 4.6 跨代对比 `[P1]`
- Qwen2.5-VL-7B（绝对坐标）/ Qwen3-VL-8B / Qwen3.5-9B（统一多模态、非思考模式）在三个数据集上的 Acc@0.5、解析失败率、延迟；配对检验以 Qwen3-VL-8B 为基线。
- 图：`docs/figures/generations_refcoco.jpg`、`generations_dior.jpg`（同一难例三列）。

### 4.7 可视化与 case 分析（`docs/figures/`）
- good cases：`refcoco_good.jpg`、`dior_good.jpg`。
- 三类失败模式（DIOR，来自 `dior_lora_fixed.jpg` / `dior_lora_broken.jpg` / `dior_bad_wrong_object.jpg`）：
  1. **参照物混淆**：关系型表达式（"the overpass similar in size to the overpass in the middle"）→ 模型框出参照物而非目标；LLM-LoRA 修复最多的一类；
  2. **同类实例选择**：一排相同的飞机 / 烟囱，仅靠方位词区分；
  3. **极小目标**：几十像素的车辆 / 风车，所有配置都弱，分辨率只部分缓解。
- LoRA 的净效果：零样本 → LLM-LoRA 修好 104 个、弄坏 13 个；给 2–3 个"弄坏"的例子（compare_broken.jpg），说明微调不是单向提升。
- ScreenSpot bad cases：`screenspot_bad.jpg` `[P1]`。
- `[P0b]` Thinking 推理链截取（想错在哪一步）。

### 4.8 优缺点讨论
- 优点：零样本开放词表、理解复杂语言、一套模型跨域；LoRA 极低成本（0.74% 参数、13 分钟）即可大幅适配。
- 缺点：坐标以文本输出（解析失败率虽低但非零，长输出慢）；小目标弱；关系推理在新域失效；每样本 0.4–0.8 s，远慢于专用检测器；微调有可测量的遗忘。
- 与专用模型的对比留在 related work 层面（若没跑 Grounding DINO 就明说）。

## 5. Conclusion（0.5 页）
- 3–5 条发现 + 局限（子集评测、单次训练种子、未做 Grounding DINO 对照、Thinking 子集小）+ 下一步（RL 后训练如 EGM、更大训练集、视觉侧高分辨率微调）。

## 附录（可选）
- 完整命令与超参表（README 已有）；复现性表；配对检验原始输出。

---

## 图表清单（文件 → 报告位置）
| 文件 | 报告 |
|---|---|
| runs/table_p0_vs_paper.md | 表 1（4.2） |
| runs/summary_p1.md + compare_p1_*.txt | 表 2（4.3 / 4.6） |
| runs/summary_dior_px.md + compare_resolution.txt | 表 3（4.4a） |
| runs/summary_p3_dior.md + compare_p3_*.txt | 表 4（4.5a/b） |
| runs/compare_steps.txt + summary_p3_dior.md | 表 5（4.5c） |
| runs/summary_p3_forgetting.md + compare_forgetting.txt | 表 4 最后一列 |
| docs/figures/*.jpg | 图 1–6（4.7） |
| 步数–精度折线（125/375/1125 步）、分辨率–精度折线 | 图（4.4 / 4.5，可用 summary CSV 画） |

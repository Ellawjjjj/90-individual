# 报告大纲（introduction / method / experiments and analysis / conclusion）

目标 6–8 页。每个小节后面标注了用哪个 run 目录 / 脚本产出的内容。

## 1. Introduction（0.5–1 页）
- 任务：指代表达理解（REC）= 给定图片 + 自然语言描述，输出目标框。
- 为什么用通用 VLM 做 grounding（零样本、开放词表、能理解复杂语言）以及它的代价（慢、文本坐标）。
- 本文做了什么：复现 Qwen3-VL Table 4；域外数据集；三代模型；超参与消融；case 分析。

## 2. Related Work / Review（1 页）
- 专用 REC 模型：MDETR、Grounding DINO、Florence-2（框回归 / 查询检测范式）。
- 通用 VLM：Qwen2-VL → Qwen2.5-VL → Qwen3-VL → Qwen3.5，InternVL；坐标以文本 token 输出。
- 视觉编码器与预训练（呼应课件）：CLIP / SigLIP 2 的语言对齐预训练，DINO 系列的自监督稠密特征。
- 后训练提升 grounding 的工作（EGM：SFT + RL）。

## 3. Method（1–1.5 页）
- Qwen3-VL 架构：ViT 编码器 → patch merger → LLM；DeepStack 多层视觉特征注入；interleaved-MRoPE。
- grounding 的形式化：提示词模板（prompts.py）、输出 JSON `bbox_2d`、坐标约定（0–1000 相对 vs 绝对像素）与校准（eval 的 `--coord-mode auto`）。
- 评测协议：Acc@0.5（与论文一致）、Acc@0.75、mIoU、解析失败率、每样本延迟。
- LoRA 微调设置：目标模块范围（LLM / 视觉 / 全部）、rank、训练目标格式。

## 4. Experiments and Analysis（3–4 页）
- 4.1 数据集与实现细节：各数据集规模、子集与种子、硬件、精度、软件版本。（`metrics.json` 的 `args`）
- 4.2 复现：与论文 Table 4 对照表。（`runs/summary_p0.md`）
- 4.3 域外泛化：RefCOCO vs DIOR-RSVG vs ScreenSpot vs 自建。（`runs/summary_p1.md`）
- 4.4 超参分析：`max_pixels` 曲线、温度、提示词、规模、4-bit。（`runs/summary_p2.md`，可用 summary CSV 画折线图）
- 4.5 消融：Thinking vs Instruct；LoRA 作用域 / rank / 数据量；遗忘。（`runs/summary_p2.md`、`runs/summary_p3_*.md`）
- 4.6 跨代对比：Qwen2.5-VL-7B / Qwen3-VL-8B / Qwen3.5-9B。（`runs/summary_p1.md`，`vis/compare.jpg`）
- 4.7 可视化与 case 分析：good/bad case 网格；按失败类型、目标大小、表达式长度、空间词分桶；Thinking 推理链截取。（`runs/*/vis/`）
- 4.8 优缺点讨论：综合以上数据。

## 5. Conclusion（0.5 页）
- 主要发现三到五条；局限（子集评测、单次采样、坐标解析）；下一步。

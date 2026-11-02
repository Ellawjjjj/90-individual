# Visual Grounding with Qwen-VL: 复现、拓展与分析

围绕 **Qwen3-VL Technical Report（arXiv 2511.21631）第 5.5 节的 2D grounding 评测**，
在单张 RTX 5090（Windows 原生，无需 WSL）上完成：

1. **复现**：Qwen3-VL-4B/8B（Instruct / Thinking）在 RefCOCO / RefCOCO+ / RefCOCOg 上的 Acc@0.5，对照论文 Table 4。
2. **拓展**：域外数据集（遥感 DIOR-RSVG、GUI 截图 ScreenSpot、自建照片）、三代模型同协议对比（Qwen2.5-VL → Qwen3-VL → Qwen3.5）。
3. **分析**：超参扫描（视觉 token 预算、解码、提示词、规模、量化）、LoRA 消融（适配 LLM 侧 / 视觉侧 / 全部、rank、数据量、遗忘）、good/bad case 可视化。

整套流程只依赖 `transformers` + `peft`，没有 mmdet/detectron2/flash-attn 这类 Linux-first 依赖。

---

## 1. 环境配置（Windows + RTX 5090）

5090 是 Blackwell（sm_120），必须用带 CUDA 12.8+ 的新版 PyTorch：

```powershell
conda create -n grounding python=3.11 -y
conda activate grounding
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"
# 输出里必须包含 sm_120，否则会报 "no kernel image is available"
pip install -r requirements.txt
```

模型权重很大（8B ≈ 16 GB），建议把 HF 缓存放到大盘：

```powershell
setx HF_HOME "D:\hf_cache"      # 新开终端生效
# 下载慢时可选: setx HF_ENDPOINT "https://hf-mirror.com"
```

Qwen3-VL 需要 `transformers>=4.57`；Qwen3.5 系列需要 5.x。代码对 4.57 和 5.x 都做了兼容
（训练参数按版本过滤，分辨率通过显式 `size=` 参数传入）。
`grounding/models.py` 会依次尝试 `AutoModelForImageTextToText` / `AutoModelForMultimodalLM` 等类，哪个能加载用哪个。
用 `pip show transformers` 看版本，`metrics.json` 里也会记录 `transformers_version`。

## 2. 三步验证安装

```powershell
python -m pytest -q tests                                  # 纯 Python 单元测试（解析/指标/数据）
python tools/make_toy_dataset.py --n 40 --out data/toy     # 合成数据
python scripts/eval_grounding.py --model fake --dataset data/toy/toy.jsonl --out runs/dryrun
python scripts/visualize.py --run runs/dryrun              # 不需要 GPU，检查整条流水线
```

然后用真模型跑一张图，**肉眼确认坐标系是对的**（这是最容易出错的地方）：

```powershell
python scripts/demo_single.py --model Qwen/Qwen3-VL-4B-Instruct --image data/toy/images/0000.jpg --expr "the red circle" --out runs/demo.jpg
```

脚本会打印原始输出，以及同一个框在三种坐标约定下的像素位置；打开 `runs/demo.jpg` 看红框是否框对。

分辨率参数是否生效也要验证一次（不需要 GPU，几秒钟）：

```powershell
python scripts/check_pixels.py --model Qwen/Qwen3-VL-4B-Instruct --image data/toy/images/0000.jpg --max-pixels 200704
python scripts/check_pixels.py --model Qwen/Qwen3-VL-4B-Instruct --image data/toy/images/0000.jpg --min-pixels 1638400
```

输出的 "through the full processor" 一行应分别变成约 448×448 和 1280×1280。评测日志也会打印第一个样本的 "model input (H,W)"，`metrics.json` 里有 `model_input` 统计，`summarize.py` 的表里有 `input_hw` 列——任何时候都能确认模型实际看到的分辨率。
三种约定：`relative_1000`（Qwen2-VL / Qwen3-VL / Qwen3.5：0–1000 归一化）、`absolute_input`（Qwen2.5-VL：相对于处理器缩放后的输入图的绝对像素）、`absolute_original`。
评测脚本默认 `--coord-mode auto`：在前 32 个样本上三种约定都算一遍，取平均 IoU 最高的那种，并把校准结果写进 `metrics.json`。

## 3. 数据准备

所有数据统一转成 JSONL（每行一条：图片路径、表达式、`[x1,y1,x2,y2]` 像素框），转换脚本会顺手画一张 `*_preview.jpg`，**务必打开确认绿框正确**。

```powershell
.\experiments\00_prepare_data.ps1      # RefCOCO 家族 + ScreenSpot + toy
```

| 数据集 | 来源 | 转换命令 | 说明 |
|---|---|---|---|
| RefCOCO / + / g | HF `lmms-lab/RefCOCO`, `lmms-lab/RefCOCOplus`, `lmms-lab/RefCOCOg` | `tools/convert_hf_to_jsonl.py --expr-field answer --bbox-format xywh` | 列：`answer` 是约 3 条表达式的列表，`bbox` 为 COCO 的 `[x,y,w,h]`。`--limit` 按行（区域）计数。若 404 试 `lmms-lab-encoder/` 前缀 |
| ScreenSpot | HF `rootsautomation/ScreenSpot` | `--expr-field instruction --bbox-format xyxy_norm` | GUI 截图，域外；`bbox` 为 0–1 归一化 xyxy |
| DIOR-RSVG | 从 [RSVG-pytorch](https://github.com/ZhanYang-nwpu/RSVG-pytorch) 页面的网盘链接下载 | `tools/convert_dior_rsvg.py --root data/DIOR_RSVG --out data/dior_rsvg` | 遥感，域外；也是 LoRA 微调的训练集 |
| 自建 | 用 [LabelMe](https://github.com/wkentaro/labelme) 标注矩形，label 写指代表达 | `tools/convert_labelme.py` → `tools/split_jsonl.py` | 作业鼓励项；拍 100 张校园/宿舍照片即可 |
| toy | `tools/make_toy_dataset.py` | — | 合成彩色几何形状，sanity check（正常模型应接近 100%） |

列名不确定时先看一眼：`python scripts/inspect_dataset.py --dataset hf:lmms-lab/RefCOCO --split val`

## 4. 实验

每个脚本都是独立可重跑的 PowerShell，内部就是 `python scripts/...` 命令，可以单独复制一条出来跑。
所有评测都支持 `--resume`（中断后接着跑）和 `--limit N`（固定种子的子集；**不同 N 的子集是嵌套的**，limit=200 ⊂ limit=500，方便跨 run 对同一批样本做对比）。

| 脚本 | 内容 | 对应报告章节 |
|---|---|---|
| `experiments/p0_reproduce.ps1` | Qwen3-VL-4B/8B Instruct 在 8 个 split 上的 Acc@0.5；Thinking 版在 3 个 val 上 | 4.2 复现 vs 论文 Table 4 |
| `experiments/p1_ood_and_generations.ps1` | Qwen2.5-VL-7B / Qwen3-VL-8B / Qwen3.5-9B × {RefCOCO, DIOR, ScreenSpot, 自建} | 4.3 域外泛化、4.6 跨代对比 |
| `experiments/p2_hyperparams.ps1` | `max_pixels` 扫描、温度、提示词模板、2B/4B/8B、4-bit、Thinking vs Instruct | 4.4 超参分析、4.5 消融（推理侧组件） |
| `experiments/p3_lora_ablation.ps1` | DIOR 上 LoRA：`--target-scope llm/vision/all`、rank 4/16/64、300/1000/3000 条训练数据，回测 RefCOCO 看遗忘 | 4.5 消融（微调侧组件） |
| `experiments/p4_visualize.ps1` | good/bad case 网格、按失败类型/目标大小/表达式长度/空间词分桶、三代模型同图对比 | 4.7 可视化与 case 分析 |

汇总表：`python scripts/summarize.py --glob "runs/p2_*" --out runs/summary_p2` 生成 Markdown + CSV，直接贴进报告。

配对检验：同一批样本上的多个 run 用 `python scripts/compare_runs.py --base runs/A runs/B runs/C` 比较，输出 Acc 差值的 bootstrap 95% 置信区间、翻转样本数和 McNemar p 值。n=500 时单个 Acc 的标准误约 ±0.02，相差 3–5 个点的两组必须用这个看是否可靠。

### 常用参数

```
--model          HF id / 本地路径 / "fake"
--adapter        LoRA 目录（runs/xxx/adapter）
--dtype          bf16 | fp16 | fp32 | 4bit
--max-pixels     视觉 token 预算（像素数）。Qwen 系列是动态分辨率，这个参数直接决定小目标能不能看清
--prompt         json | json_strict | plain | refcoco | cot   （grounding/prompts.py）
--temperature    0 = greedy
--enable-thinking auto|on|off   Qwen3.5 混合推理模型用它切换；*-Thinking 检查点总是思考
--batch-size     Instruct 模型 2–4，Thinking 用 1
--coord-mode     auto | relative_1000 | absolute_input | absolute_original
```

### 显存参考（5090 32 GB，bf16）

- 8B Instruct 推理：默认分辨率 batch 4 没问题；`max_pixels` 开到 1.6M 时 batch 降到 1–2。
- 8B Thinking：batch 1，`--max-new-tokens 4096`，每样本十几秒到一分钟，用 `--limit 200`。
- 4B LoRA 训练（`--max-pixels 602112`，batch 2，grad-accum 4，梯度检查点）：约 18–22 GB。8B 用 batch 1。
- OOM 先降 `--max-pixels`，再降 batch。

## 5. 输出文件

每个 run 目录：

```
raw.jsonl          每个样本的原始模型输出、模型实际看到的输入尺寸、延迟（可 --resume）
predictions.jsonl  解析后的框、IoU、是否命中、自动失败类别（parse_failure / wrong_object / loose_localisation / partial_overlap）
metrics.json       参数、坐标系校准结果、Acc@0.5 / Acc@0.75 / mIoU / 解析失败率 / 每样本秒数、按桶统计
vis/               good_cases.jpg、bad_<类别>.jpg、compare.jpg、failure_summary.md、cases/*.jpg
```

## 6. 和作业要求的对应

| 要求 | 本仓库提供 |
|---|---|
| 选题（Visual Grounding / Multi-modality reasoning） | 指代表达理解；Thinking 模型的推理链分析 |
| 至少两个数据集，鼓励子集、自建、与原论文不同 | RefCOCO 家族（原论文）+ DIOR-RSVG、ScreenSpot（域外）+ LabelMe 自建 + toy |
| 至少一个方法，鼓励多个 | Qwen3-VL 为主，同协议下 Qwen2.5-VL、Qwen3.5；可再加 Grounding DINO 作专用模型对照 |
| 自己产出结果 | 所有数字来自 `runs/*/metrics.json`；论文数字只作标注来源的参考列 |
| 超参分析 | P2：token 预算、温度、提示词、规模、精度 |
| 关键组件消融 | P2 的 Thinking 开关 + P3 的 LLM 侧 / 视觉侧 / 全部 LoRA、rank、数据量、遗忘 |
| 优缺点讨论 | 域外掉点、解析失败率、延迟（`sec_per_sample`）、小目标桶 |
| 可视化 + good/bad case | P4 |
| 报告 + 代码 | 本仓库 + 报告（introduction / method / experiments and analysis / conclusion） |

## 7. 常见问题

- **`no kernel image is available for execution on the device`**：torch 不带 sm_120，按第 1 节重装 cu128 版本。
- **解析失败率很高**：先跑 `demo_single.py` 看原始输出；Thinking 模型要给够 `--max-new-tokens`；换 `--prompt json_strict`。
- **Acc 明显偏低（<0.3）但框看起来方向对**：坐标系错了。看 `metrics.json` 里的 `calibration`，或显式指定 `--coord-mode`。
- **预览图里 GT 框不对**：`--bbox-format` 选错（xywh vs xyxy vs *_norm），用 `inspect_dataset.py` 看原始值。
- **Qwen3.5 加载失败**：`pip install -U transformers`；仍失败时在 issue 里贴 `grounding/models.py` 打出的 "Could not load model" 错误列表。
- **LoRA 时 `--target-scope` 没匹配到模块**：`python scripts/train_lora.py ... --list-modules` 打印所有 Linear 层名，然后用 `--target-scope custom --target-regex "..."`。
- **Windows 上 DataLoader 报错**：训练脚本已固定 `dataloader_num_workers=0`。

## 8. 目录结构

```
grounding/   parse.py（框解析/坐标转换）metrics.py（IoU、分桶）data.py（JSONL/HF 加载）
             prompts.py（提示词与训练目标）models.py（模型加载、批量生成、fake 模型）vis.py（画图）
scripts/     eval_grounding.py  train_lora.py  visualize.py  summarize.py  inspect_dataset.py  demo_single.py
tools/       convert_hf_to_jsonl.py  convert_dior_rsvg.py  convert_labelme.py  split_jsonl.py  make_toy_dataset.py
experiments/ 00_prepare_data.ps1  p0_reproduce.ps1  p1_ood_and_generations.ps1  p2_hyperparams.ps1
             p3_lora_ablation.ps1  p4_visualize.ps1
tests/       单元测试（不需要 GPU）
```

## 参考

- Qwen3-VL Technical Report, arXiv:2511.21631 — 5.5 节 2D/3D grounding，Table 4 小模型 RefCOCO-avg。
- Qwen3.5 blog: https://qwen.ai/blog?id=qwen3.5 ；模型 `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`。
- RefCOCO/+/g: Kazemzadeh et al. 2014; Mao et al. 2016. DIOR-RSVG: Zhan et al. 2023. ScreenSpot: Cheng et al. 2024.
- NVIDIA EGM（在 Qwen3-VL 基础上用 SFT+RL 提升 grounding）可作相关工作与讨论的参照。

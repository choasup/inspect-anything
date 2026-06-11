# 01 · 多模态架构调研与优势总结

> 调研时间：2026-06。目标：拆解当前最强的开源多模态架构，提炼出对 inspect-anything 有直接借鉴价值的设计决策。

## 1. 主流架构总览

当前 VLM 的主流范式已经收敛为三段式：**视觉编码器 → 连接器（projector/resampler）→ LLM 解码器**。差异主要体现在四个维度：视觉编码器如何处理高分辨率、视觉 token 如何压缩、LLM 是否 MoE、是否支持原生 grounding（输出坐标）。

### 1.1 Qwen3-VL（Alibaba）— 当前开源天花板

- **架构**：原生动态分辨率 ViT + MRoPE（多维旋转位置编码，时间/高/宽分维编码）+ MoE LLM（235B-A22B / 30B-A3B 两档，均有 Instruct 与 Thinking 变体及官方 FP8 版本）。
- **优势**：
  - 动态分辨率不切 tile，图像按原始长宽比直接进 ViT，避免 tiling 造成的跨块语义割裂——对货架横幅长图、产线宽幅相机图至关重要。
  - 原生 2D/3D grounding：直接以文本形式输出归一化坐标，检测/定位任务不需要外挂检测头也能做，是教师自动标注的关键能力。
  - 32 语种 OCR + 文档理解顶级，价签核对、仪表读数、包装文字审核可直接零样本可用。
  - MoE 的 A3B（激活 3B）档位证明了「大容量、小激活」路线在多模态上成立，旗舰版对标 Gemini-2.5-Pro / GPT-5。
- **对我们的启示**：**教师层首选**。30B-A3B 自托管做自动标注性价比最高；235B 按需调用做仲裁。

### 1.2 InternVL3（Shanghai AI Lab）— 大视觉编码器路线

- **架构**：InternViT-6B（独立预训练的 60 亿参数视觉编码器，448px）+ Qwen2.5 系列 LLM，旗舰 78B；训练采用 V3 起的原生多模态预训练 + MPO（混合偏好优化）。
- **优势**：
  - 证明了**视觉编码器的容量值得单独投资**：InternViT-6B 对细粒度视觉差异（缺陷纹理、SKU 包装微差）的表征明显强于 0.3-0.4B 级 CLIP-ViT。
  - Pixel unshuffle 把视觉 token 压缩 4 倍（448px → 256 token），是最简单有效的 token 压缩手段。
  - MPO 训练法对推理类 benchmark 提升 4+ 点，说明偏好优化在多模态上同样有效。
- **对我们的启示**：学生模型的视觉塔要尽量大、LLM 可以小（详见 02 文档的「视觉重、语言轻」原则）；pixel unshuffle 直接采用。

### 1.3 FastVLM（Apple, CVPR 2025）— 速度路线的标杆

- **架构**：FastViTHD 混合卷积-Transformer 层级视觉编码器，多尺度特征融合，专为高分辨率输入优化。
- **优势**：
  - 直击 VLM 推理的真正瓶颈：**高分辨率下的视觉编码延迟 + 视觉 token 数量**（prefill 时间随 token 数线性甚至超线性增长）。
  - 层级卷积前端在高分辨率下比纯 ViT 快数倍，且 token 数随分辨率增长更缓。
  - 证明「准确率-延迟」帕累托前沿可以靠编码器架构推进，不必靠砍分辨率。
- **对我们的启示**：学生模型视觉编码器采用混合层级架构（而非纯 ViT），这是端侧 <100ms 目标的核心保障。

### 1.4 SmolVLM（HuggingFace）— 小模型工程极限

- **架构**：256M-2.2B 全系列，SigLIP 视觉塔 + 激进 token 压缩（单图最低 64 token）+ 小 LLM。
- **优势**：
  - 256M 版本推理 <1GB 显存，却超过 300 倍大的 Idefics-80B——证明**数据配方比参数量重要**。
  - 全套训练数据、配方、工具 Apache 2.0 开源，是小模型训练 pipeline 的最佳参考实现。
- **对我们的启示**：1-2B 学生模型在垂直场景完全够用，前提是训练数据质量（这正是飞轮提供的）；其训练配方可直接复用为我们 SFT pipeline 的骨架。

### 1.5 Moondream — 端侧实用主义

- **优势**：<5GB 内存可跑（树莓派级硬件），原生支持 point/detect/count 结构化输出而非纯自由文本。
- **对我们的启示**：**结构化输出（坐标、计数、JSON）应是一等公民**，训练时就作为目标格式，而不是事后用正则从自由文本里抠。

### 1.6 开放词汇检测系（Grounding DINO / YOLO-World / Florence-2 / SAM2）

VLM 之外，检测专用架构仍不可替代：

- **Grounding DINO**：文本查询驱动的开放词汇检测，精度最高，是自动标注流水线（如 Grounded-SAM-2）的事实标准。
- **YOLO-World**：prompt-then-detect 范式——文本词表离线编码后融合进 YOLO 检测头，推理时零文本编码开销，实时帧率。注意 Ultralytics 商用许可问题，自研需走 Grounding DINO 蒸馏路线。
- **SAM2**：可提示分割 + 视频跟踪，配合 grounding 模型构成「检测→分割→跟踪」全自动标注流水线。
- **Florence-2 / DINO-X**：单模型统一检测/分割/描述，序列到序列接口。
- **对我们的启示**：纯 VLM 做密集小目标（货架上百个 SKU）的检测/计数仍偏弱且慢，**学生侧需要 VLM + 实时开放词汇检测头双轨**；Grounded-SAM-2 流水线直接用于飞轮的自动预标注。

### 1.7 异常检测范式（工业质检特有）

- AnomalyGPT 等工作证明 VLM 可以做异常「有无 + 数量 + 位置」的归因描述。
- 但工业落地主力仍是**无监督异常检测**（PatchCore / EfficientAD 系：只用正常样本建模，毫秒级，无需缺陷标注），VLM 负责对检出异常做语义归因和报告生成。
- **对我们的启示**：质检线采用「无监督 AD 模型初筛 + VLM 归因」两级架构，解决缺陷样本稀缺的冷启动问题。

## 2. 架构优势总结（可直接采用的设计结论）

| # | 设计结论 | 来源 | 用在哪 |
|---|----------|------|--------|
| 1 | 原生动态分辨率 > 固定分辨率切 tile | Qwen3-VL | 学生模型输入设计 |
| 2 | 视觉编码器容量值得单独做大，「视觉重、语言轻」 | InternVL3 | 学生模型参数分配 |
| 3 | 混合卷积-ViT 层级编码器是高分辨率低延迟的关键 | FastVLM | 学生模型视觉塔 |
| 4 | Pixel unshuffle / token merging 压缩视觉 token 4-9 倍，精度损失极小 | InternVL3 / SmolVLM | 连接器设计 |
| 5 | MoE「大容量小激活」适合教师层自托管 | Qwen3-VL 30B-A3B | 教师选型 |
| 6 | 坐标/计数/JSON 结构化输出要进训练目标 | Qwen3-VL / Moondream | 学生 SFT 格式 |
| 7 | 数据配方 > 参数量，小模型上限由数据决定 | SmolVLM | 飞轮的价值依据 |
| 8 | VLM + 专用检测头双轨，密集检测不硬上 VLM | YOLO-World 系 | 学生侧部署形态 |
| 9 | Grounding+SAM2 流水线 = 自动标注基础设施 | Grounded-SAM-2 | 飞轮预标注 |
| 10 | 偏好优化（MPO/DPO）对多模态推理有稳定增益 | InternVL3 | 训练第三阶段 |
| 11 | 无监督 AD 初筛 + VLM 归因，解决缺陷样本稀缺 | AnomalyGPT/PatchCore | 质检产品线 |
| 12 | FP8/量化版本随模型一起发布是工程标配 | Qwen3-VL | 发布流程 |

## 参考来源

- [BentoML: Multimodal AI — Best Open-Source VLMs in 2026](https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models)
- [DataCamp: Top 10 Vision Language Models in 2026](https://www.datacamp.com/blog/top-vision-language-models)
- [Labellerr: Best Open-Source Vision Language Models of 2026](https://www.labellerr.com/blog/top-open-source-vision-language-models/)
- [SmolVLM: Redefining small and efficient multimodal models (arXiv)](https://arxiv.org/html/2504.05299v1)
- [HuggingFace: SmolVLM blog](https://huggingface.co/blog/smolvlm)
- [LearnOpenCV: VLM on Edge Devices](https://learnopencv.com/vlm-on-edge-devices/)
- [IDEA-Research: Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)
- [PyImageSearch: Grounded SAM 2 — Open-Set Detection to Segmentation and Tracking](https://pyimagesearch.com/2026/01/19/grounded-sam-2-from-open-set-detection-to-segmentation-and-tracking/)
- [YOLO-World: Real-Time Open-Vocabulary Object Detection (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.pdf)
- [TildAlice: YOLO vs SAM vs Grounding DINO 选型指南](https://tildalice.io/yolo-sam-grounding-dino-detection-guide/)

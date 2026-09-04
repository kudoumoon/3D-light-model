# Language Preference

- 默认使用中文回复。
- 除非用户明确要求英文，否则解释、分析、计划、总结使用中文。
- 代码、变量名、函数名、命令、日志、报错信息保持原始语言。
- 专业术语优先使用“中文 + English original”的形式。
- 回答直接、清晰，先给结论，再解释原因。

# GPU Usage

- 绝对不能抢占别人的 GPU 卡；执行任何训练、评估、推理任务前，必须先检查 GPU 占用和进程情况。
- 每次执行任务都要在不影响他人任务的前提下，尽可能利用所有空闲 GPU，提高实验吞吐和项目质量。
- 空卡判断必须保守：优先选择低显存、低利用率、进程表无占用且连续稳定空闲的 GPU。

# Experiment Code Quality

- 实验代码优先保持路径清晰、假设明确、错误显式暴露；不要堆叠大量 `if/else` 兜底逻辑来掩盖数据、配置或模型问题。
- 必要的兼容分支必须写清楚触发条件和实验影响，避免让兜底机制改变指标含义或影响结果可信度。

# Document Writing Quality

- 凡是撰写、改写或润色文档内容，包括报告、论文段落、组会材料、README、方法正文和实验总结，必须先调用并遵循 `humanizer-zh` skill。
- 文档写作要保留事实、证据边界和指标口径；不要为了显得结果更强而夸大实验结论。
- 涉及论文或学术写作时，优先配合 `nature-writing` / `nature-polishing` 控制结构、论证链和可审稿性。
# M1 Research Core and Active To-Do

## 核心创新主体（当前论文主线）

M1 的主体不是复现或蒸馏 MoGe-3，而是 Geometry-Aligned Latent 3D：从冻结 world-model VAE 的 latent grid 直接预测与 cell 对齐的 depth、validity 和 confidence，并用预测几何支持 latent-to-latent transport。MoGe-3 仅用于教师监督、初始化和可控对照；论文的主要贡献应落在 latent-grid geometry alignment、可传输几何表示，以及面向运动的概率置信度路由。

当前必须保持的输出契约：

- `latent_depth [B,1,44,80]`
- `latent_points [B,3,44,80]`
- `latent_valid [B,1,44,80]`
- `latent_confidence [B,1,44,80]`
- `intrinsics [B,3,3]` 和 geometry metadata

论文中不得把“低延迟 M1 head”写成“完整系统已加速”，除非 VAE、renderer、现有 M2 和完整 DiT pipeline 已在同一硬件、同一输入协议下统一测量。

## 当前阶段 To-Do

1. **扩大 student 泛化域**：在保持真实 pose / intrinsics / depth 监督的前提下，引入 TUM、Bonn 及至少一个不同室内外分布的数据域；采用 dataset-disjoint、camera-disjoint 或 cross-domain test，分别报告静态、动态、薄结构、反光和大运动场景。
2. **coverage 修复**：将 `46.5%` coverage 作为明确瓶颈，比较 latent-cell-aware pooling、subcell splatting、boundary-preserving geometry 和显式 hole/occlusion 状态；不能用 confidence gating 掩盖 geometry 空洞。
3. **概率 confidence**：报告 calibration curve、AUC、ECE、Brier score、risk-coverage、precision@fixed-keep-ratio 和跨域校准；confidence label 必须只使用训练期 target latent / pose，测试时不能读 target。
4. **统一效率证据**：测量 Frozen VAE encode、M1 geometry、confidence、latent warp renderer、现有 M2、完整 DiT pipeline；覆盖 batch size、latent spatial resolution、chunk length，并统一 warm-up、dtype、GPU、同步方式和显存统计。
5. **baseline / benchmark**：baseline 至少包括 RGB-space M1 v7、MoGe-3 teacher、naive depth resize、latent geometry head、latent geometry + TVOD、latent geometry + transport loss、Copy transport；benchmark 包括 depth/point、projection、coverage、latent consistency、decoded quality、confidence calibration、latency、throughput、peak VRAM 和参数量。
6. **现有 M2 闭环**：只接入和评估现有 M2，不修改 M2 代码；比较 Copy、v7、当前 latent M1、confidence routing 的统一质量与速度。
7. **评审循环**：每轮实验保存 config、commit、checkpoint、metrics、latency、GPU 信息、失败样例和 split manifest；评审从创新性、泛化性、可行性、有效性、效率和可复现性评分，只有证据达到 95/100 才结束。


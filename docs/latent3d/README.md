# M1：Geometry-Aligned Latent 3D 审阅入口

M1 的论文主线已经从“RGB 预测深度，再把结果缩放到 latent”调整为“在冻结 world-model VAE 的原生网格上预测几何，并用 latent transport 评价几何是否有用”。MoGe-3 保留为离线伪标签来源和上界参照，不再承担方法定义。

## 一句话方法

给定冻结 VAE 的空间 latent `z_t ∈ R^(B×16×44×80)`，轻量几何头输出与同一网格对齐的 depth、point map、valid mask、motion-conditioned confidence 和相机元数据；训练目标不仅拟合离线几何，还要求预测几何在虚拟或真实相机运动下把 source latent 投影到正确的 target latent。

## 审阅顺序

1. [实验基线锁定](../../experiments/latent3d/BASELINE_LOCK.json)：v7/v10、VAE 和早期证据的不可覆盖记录。
2. [当前结果总表](../../LATENT3D_RESULTS.md)：只按 Fact、Negative Result、Hypothesis 写结论。
3. [P13/P14 升级报告](../M1_LATENT3D_UPGRADE_P13_P14_2026_09_02.md)：latent reprojection 与虚拟 6DoF projective behavior distillation。
4. [真实数据路线](REAL_DATASETS.md)：哪些数据可直接用于 GT depth/pose/K，哪些只能作为稀疏或估计监督。
5. [M1→M2 只读桥接报告](../M1_LATENT3D_M2_INTEGRATION_GROUP_REPORT_ZH.md)：不修改 M2 的接口和已有闭环证据。
6. [中文 Method 草稿](../M1_METHOD_DRAFT_ZH.md)：论文正文的当前版本。

## 代码分类

| 层级 | 主要文件 | 职责 |
|---|---|---|
| 输出接口 | `latent_geometry_head.py` | latent → depth/points/valid/confidence/K/metadata |
| RGB 几何对齐 | `latent_geometry_alignment.py`、`latent_surface_alignment.py` | 固定 pooling 与可学习 surface selection |
| 重投影 | `latent_reprojection_loss.py` | 软 z-buffer forward splat、Copy 对照、warp loss |
| 时间语义 | `latent_chunk_geometry.py` | Wan causal temporal group 的 anchor 归属；不改变冻结 shape |
| 置信度 | `latent_motion_confidence.py` | geometry 冻结后的 motion-conditioned confidence |
| 系统桥接 | `latent_m2_bridge.py` | 将固定输出交给 M2；本仓库不修改 M2 |
| 真实数据硬门 | `tools/run_tum_gt_latent3d_feasibility.py` | TUM sensor depth + mocap pose + K，不经过 MoGe |
| 训练与消融 | `tools/train_latent_geometry_head.py`、`tools/train_latent_reprojection_head.py`、`tools/train_virtual_projective_distillation.py` | scene split、跨 seed、transport-aware 目标 |

## 固定输出契约

- `latent_depth: [B,1,H_l,W_l]`
- `latent_points: [B,3,H_l,W_l]`
- `latent_valid: [B,1,H_l,W_l]`
- `latent_confidence: [B,1,H_l,W_l]`
- `intrinsics: [B,3,3]`
- 元数据：`spatial_downsample`、`temporal_downsample`、`coordinate_convention`

对 Wan/Matrix-Game 当前配置，`H_l×W_l = 44×80`，空间压缩率为 8，latent channel 为 16。单帧 M1 对每个 causal latent slice 独立运行；chunk 扩展不改变上述 shape。

## 证据纪律

实验目录不上传大 checkpoint、VAE 权重或数据集。GitHub 只保存协议、配置、机器可读指标、必要论文图和失败说明。任何结果都必须区分传感器真值、估计位姿、MoGe 伪标签和受控合成几何；四者不能混写为“GT”。

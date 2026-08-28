# 可交付几何模型与重投影友好型设计

日期：2026-08-28

## 结论

当前可交付版本建议采用 `MoGe-3 ViT-L, refine_steps=0` 作为高质量泛化几何模型，输出统一的 `geometry.npz` point map 契约用于后续重投影。它比 MoGe-2 Small 慢约 1.65x，但在困难场景 Temple Run 上有效 mask 从 49.67% 提升到 70.72%，对应 yaw5 CUDA 重投影覆盖率从 46.13% 提升到 63.40%。这类低纹理/大前景/游戏视角场景正是论文里容易暴露重投影失败的位置。

若目标是速度收益最大化，线上路径不应每帧跑 MoGe-3。推荐分层交付：

- Teacher/reference：MoGe-3 ViT-L `refine_steps=0`，生成强泛化 point map、depth、mask、intrinsics、normal。
- Realtime baseline：MoGe-2 Small 或后续蒸馏学生模型，承担高频 geometry refresh。
- 论文创新：训练一个重投影友好型几何头，使输出不只追求深度像素误差，而是直接优化可重投影性、遮挡边界、hole/conflict 预测和时序复用收益。

## 实验环境

- GPU：NVIDIA H100 80GB HBM3，实际使用 GPU 4/5；GPU 6 当时被其他任务占用约 72GB。
- MoGe upstream commit：`74fbce054ebed49800de42d0ad0e83495065719a`
- MoGe-3 checkpoint：`checkpoints/moge-3-vitl/model.pt`
- MoGe-3 checkpoint size：`1,481,333,394` bytes
- MoGe-3 checkpoint SHA256：`9b41b7b9f65ad80aab7ad686f5e9cc0d1fd33f1964022618dfbcd52fc1fb7925`
- 输入图像：
  - `third_party/Matrix-Game/Matrix-Game-2/demo_images/gta_drive/0000.png`
  - `third_party/Matrix-Game/Matrix-Game-2/demo_images/temple_run/0000.png`
  - `third_party/Matrix-Game/Matrix-Game-2/demo_images/universal/0003.png`

## 几何推理结果

单位为 ms，均为 resident RGB/model 的 `infer()` 微基准；不含下载、解码、H2D 和导出。

| Scene | Model | Steps | p50 | p95 | Valid mask |
|---|---:|---:|---:|---:|---:|
| GTA | MoGe-2 Small | 0 | 13.97 | 17.12 | 96.31% |
| Temple | MoGe-2 Small | 0 | 13.69 | 13.76 | 49.67% |
| Universal | MoGe-2 Small | 0 | 13.77 | 14.46 | 79.56% |
| GTA | MoGe-3 ViT-L | 0 | 23.33 | 23.40 | 96.36% |
| GTA | MoGe-3 ViT-L | 1 | 36.86 | 39.64 | 96.36% |
| GTA | MoGe-3 ViT-L | 3 | 57.79 | 60.21 | 96.36% |
| Temple | MoGe-3 ViT-L | 0 | 23.64 | 23.75 | 70.72% |
| Temple | MoGe-3 ViT-L | 1 | 35.74 | 37.69 | 70.72% |
| Temple | MoGe-3 ViT-L | 3 | 54.07 | 57.39 | 70.72% |
| Universal | MoGe-3 ViT-L | 0 | 22.81 | 22.86 | 79.73% |
| Universal | MoGe-3 ViT-L | 1 | 33.21 | 33.74 | 79.73% |
| Universal | MoGe-3 ViT-L | 3 | 53.46 | 53.69 | 79.73% |

## CUDA 重投影结果

Pose perturbation：yaw 5 deg，forward 0.10，splat radius 1。

| Scene | Geometry | Coverage | Resident GPU p50 | Upload+GPU p50 |
|---|---:|---:|---:|---:|
| GTA | MoGe-2 Small | 88.98% | 1.214 | 1.597 |
| Temple | MoGe-2 Small | 46.13% | 1.021 | 1.474 |
| Universal | MoGe-2 Small | 70.90% | 1.106 | 1.483 |
| GTA | MoGe-3 ViT-L step0 | 88.73% | 1.130 | 1.503 |
| Temple | MoGe-3 ViT-L step0 | 63.40% | 1.104 | 1.532 |
| Universal | MoGe-3 ViT-L step0 | 72.07% | 1.105 | 1.494 |

## 重投影友好型几何模型设计

核心观点：论文创新不应该只说“用了更好的深度模型”，而是定义 `Reprojection-Friendly Geometry`，让几何模型显式服务于加速生成/补帧/视角更新。

建议训练目标：

1. Point map teacher distillation：以 MoGe-3 ViT-L step0/1 为 teacher，蒸馏到更轻的学生模型。监督 `points/depth/normal/intrinsics/mask`，保留 MoGe-3 在困难场景上的泛化优势。
2. Projection loss：随机采样小 pose perturbation，将 point map forward splat 到目标视角，优化 source-target photometric / perceptual consistency。重点不是深度绝对误差，而是投影后像素落点和可见性。
3. Occlusion-aware mask loss：单独预测 `valid_to_warp`、`disocclusion_hole`、`depth_conflict`。这能直接服务 tile/router，避免把遮挡边界误当作可复用区域。
4. Boundary sharpness loss：在物体轮廓、深度突变和 motion boundary 处提高 mask/normal/depth 的一致性，减少 forward splat 的边界拖影。
5. Confidence/routing head：输出每个 tile 的 warp confidence、hole ratio、expected reprojection error，用于决定 copy/warp/active-DiT/full-DiT 的路由。
6. Temporal scale consistency：相邻帧或相邻 anchor 的 depth scale/intrinsics 要稳定，否则 point map 可看但不可连续复用。

建议模型输出契约：

- `points`: `H x W x 3`, OpenCV camera coordinate, x right, y down, z forward
- `depth`: `H x W`
- `mask`: geometric valid mask
- `intrinsics`: normalized `3 x 3`
- `normal`: optional but recommended
- `warp_confidence`: `H x W` or tile-level confidence
- `disocclusion_risk`: `H x W` or tile-level risk
- `depth_conflict_risk`: `H x W` or tile-level risk

## 推荐论文实验路线

第一阶段交付泛化几何：用 MoGe-3 ViT-L step0 生成 point map，作为高质量 reference。当前 `runs/ab_*_moge3_vitl/step_0/geometry.npz` 已可直接用于重投影。

第二阶段证明速度收益：不要每帧跑几何。采用 anchor interval K，比如 K=4/8/16；anchor 帧跑几何，中间帧走 CUDA warp + hole/risk tile 的 active DiT。已有 DiT proxy 表明只有当几何复用跨多个帧摊销时，速度收益才成立。

第三阶段加入创新：训练 reprojection-friendly student，使其在同等或更低几何耗时下，提高 coverage、降低 hole/conflict、提升 tile router 的正确性。这是论文相对“直接接 MoGe-3”的主要贡献点。

## 当前产物位置

- MoGe-2 baseline：`runs/ab_gta_moge2_s`, `runs/ab_temple_moge2_s`, `runs/ab_universal_moge2_s`
- MoGe-3 deliverable：`runs/ab_gta_moge3_vitl`, `runs/ab_temple_moge3_vitl`, `runs/ab_universal_moge3_vitl`
- Clean MoGe-3 warp：`runs/ab_*_moge3_vitl/step_0/warp_yaw5_clean/reprojection_cuda.json`


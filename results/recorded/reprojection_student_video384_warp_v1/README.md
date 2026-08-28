# Reprojection Student Video384 Warp v1 实验记录

日期：2026-08-28  
目标：在 MoGe-3 teacher point map 基础上训练一个更快、便于重投影的轻量几何模型。

## 结论

本轮已经得到一个可运行、可交付的 v3 几何 student 原型：

- 输入：单张 RGB，最长边 384。
- 输出：`points` point map、`depth`、`mask`、`normal`、`source_confidence`、`warp_confidence`。
- 训练数据：从 Matrix-Game 视频裁剪出的 188 帧，由 MoGe-3 ViT-L 生成 teacher geometry。
- best checkpoint：epoch 75，训练阶段 best val loss = 0.277859。
- 正式验证集评估：38 个 holdout 视频帧，平均 student inference = 2.63ms。
- 对比 MoGe-3 teacher 几何生成耗时约 22.6–27.3ms，本模型几何预测约 8.6x–10.4x 更快。

当前模型已经适合作为论文系统里的“快速几何分支（fast geometry branch）”和重投影实验载体，但还不是最终泛化模型。主要短板是训练数据仍只来自一个视频的两个 crop，泛化性需要更大 teacher 数据集继续提升。

## 创新点：projected-valid / warp-confidence 监督

上一版 student 只监督 point map、mask、normal 和投影坐标一致性。v3 新增一个 `warp_confidence` head：

1. 对 teacher point map 施加随机小视角运动：
   - yaw：±2.5°、±5°、±7.5°
   - forward：0.05、0.10、0.15
2. 将 teacher 3D 点投影到目标视角。
3. 如果点在目标相机前方并且投影仍在图像内，则该 pixel 的 projected-valid target = 1，否则为 0。
4. 使用 BCE 训练 `warp_confidence`。

这让模型不只是预测几何，还显式学习“哪些源像素更适合被 forward splat 到新视角”。这更贴近论文里的重投影目标。

## 数据集

`teacher_dataset_summary.json`

| 指标 | 数值 |
|---|---:|
| teacher samples | 188 |
| max size | 384 |
| MoGe-3 tokens | 1200 |
| refine steps | 0 |
| valid fraction mean | 0.9227 |
| valid fraction min | 0.8577 |
| valid fraction max | 0.9863 |
| teacher ms mean | 27.33ms |

数据划分按 scene 做 temporal holdout：每个 scene 最后 20% 帧为验证集，因此是“视频内未来帧验证”，不是跨场景泛化验证。

## 训练配置

`train/config.json`

| 项 | 数值 |
|---|---:|
| model width | 48 |
| epochs | 80 |
| learning rate | 0.0002 |
| train samples | 150 |
| val samples | 38 |
| best epoch | 75 |
| best val loss | 0.277859 |

loss weights：

| loss | weight |
|---|---:|
| point | 1.00 |
| mask | 0.25 |
| normal | 0.10 |
| edge | 0.20 |
| projection | 2.00 |
| warp | 0.50 |

## Best checkpoint 验证结果

`eval_best/summary.json`

| 指标 | 数值 |
|---|---:|
| val samples | 38 |
| loss | 0.298247 |
| point | 0.088940 |
| mask | 0.026074 |
| normal | 0.019566 |
| edge | 0.029262 |
| projection | 0.036274 |
| warp | 0.244861 |
| inference median mean | 2.634ms |
| inference p95 mean | 2.642ms |

训练阶段 best val loss 与正式评估 loss 不完全一致，是因为 projected-valid/projection loss 在每次计算时会随机采样 yaw/forward。checkpoint 选择仍以训练记录中的 best val loss 为准。

## 重投影覆盖率测试

测试设置：

- CUDA forward splat
- yaw = 5°
- forward = 0.10
- splat radius = 1
- selected validation frames = 6

`reprojection_selected/summary.json`

| sample | teacher coverage | student coverage | gap |
|---|---:|---:|---:|
| straight 0075 | 0.7861 | 0.7103 | -0.0758 |
| straight 0084 | 0.7853 | 0.7197 | -0.0657 |
| straight 0093 | 0.7797 | 0.6672 | -0.1125 |
| turning 0075 | 0.8804 | 0.8665 | -0.0139 |
| turning 0084 | 0.8782 | 0.8572 | -0.0209 |
| turning 0093 | 0.8906 | 0.8409 | -0.0498 |

均值：

| 指标 | teacher | student |
|---|---:|---:|
| coverage mean | 0.8334 | 0.7770 |
| warp median mean | 1.1198ms | 1.0735ms |

解释：

- student 的重投影覆盖率尚未追平 MoGe-3 teacher，平均低 5.64 个百分点。
- 转弯 crop 表现明显更接近 teacher，说明模型在局部视频分布内能学到可用几何。
- 直行 crop 后段帧退化更明显，说明当前数据量和模型容量不足以保证所有局部场景都高覆盖。

## 可交付物位置

- 新训练脚本：`tools/train_reprojection_student_warp.py`
- 新评估脚本：`tools/evaluate_reprojection_student_warp.py`
- 训练记录：`train/`
- best checkpoint 评估：`eval_best/`
- selected 重投影 JSON 与图像：`reprojection_selected/`

checkpoint 本身保存在本地 `runs/reprojection_student/student_video384_warp_v1_width48/checkpoints/best.pt`，按仓库策略不提交到 GitHub。

## 下一步建议

是否需要继续训练自己的模型：需要，但不要只在当前 188 帧上继续加 epoch。

推荐下一步：

1. 扩大 teacher 数据：至少 1k–5k 帧，覆盖更多游戏场景、室内/室外、直线/转弯、近景/远景、动态遮挡。
2. 加入 multi-motion warp 评估：yaw ±5/±10、forward 0.05/0.10/0.20，覆盖率、hole ratio、边缘撕裂指标一起记录。
3. 把 `warp_confidence` 用进实际 forward splat：低 confidence 点降低 splat 权重或过滤，减少错误覆盖。
4. 训练更强 student：width 64 或 tiny encoder + decoder，并加入轻量 depth/normal 边缘保真。
5. 设计论文对照组：MoGe-3 ViT-L teacher、MoGe-2 small、student v2、student v3-warp，同一套重投影覆盖率和速度表。


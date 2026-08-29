# M1 3D Geometry Module：TVOD 规模化训练与重投影评估报告

日期：2026-08-29  
目标：构建一个基于 MoGe-3 teacher 的轻量 3D 几何模型，输出 point map / valid mask / warp confidence，用于后续重投影模块，并验证相对直接跑 MoGe-3 是否存在速度收益。

## 当前结论

当前最佳可交付模型建议使用：

`runs/reprojection_student/student_video384_tvod_v7_2k_occ075_width64_lr15/checkpoints/best.pt`

该模型不是几何 loss 最低的模型，但在速度、覆盖率保持、hard scene 稳定性之间最均衡。它适合作为论文 M1 的主交付模型；`occ05_width80` 可作为平均覆盖率更高但更慢、更不稳的 Pareto 变体。

严格按 A 会 / ICLR oral 水平评估，当前 M1 仍不能声称达到 90/100。主要短板是训练域仍偏 Matrix-Game 视频场景，缺少跨数据集泛化验证、真实 downstream DiT 加速闭环，以及 hard-case 覆盖率缺口仍明显。当前更合理评分约 78-82/100。

## 方法创新：TVOD

本轮采用的核心创新点是 Target-View Occupancy Distillation（TVOD）。

传统蒸馏只在 source view 上匹配 teacher 的 point map、mask、normal 或 depth-like geometry。这对“看起来几何误差低”有帮助，但不保证下游 forward reprojection 时目标视角的像素覆盖率好。

TVOD 的训练目标改为显式考虑虚拟相机运动后的 target view：

1. 对 student 与 teacher 的 predicted flow / projected coordinates 做 coarse target-view splatting。
2. 用 differentiable occupancy grid 近似目标视角覆盖分布。
3. 对 student occupancy 和 teacher occupancy 做 smooth L1 distillation。
4. 同时保留 warp-confidence head，用 projected-valid mask 监督哪些点重投影后可信。

代码入口：

- `tools/train_reprojection_student_warp.py`
  - `target_view_occupancy(...)`
  - `--occupancy-weight`
- `tools/evaluate_reprojection_student_warp.py`
  - 记录 `occupancy`
- `tools/compare_reprojection_models.py`
  - 多 motion 下比较 teacher/student 重投影覆盖率、worst case、by scene、by motion。

## 训练规模

本轮新增 teacher 数据：

- teacher root：`runs/teacher_moge3_video384_v6_2k`
- 样本数：1880 frames
- 来源：Matrix-Game-2 / Matrix-Game-3 视频，多 crop 抽帧
- val：375 samples
- teacher MoGe-3 平均推理耗时：
  - mean：25.710 ms
  - mean excluding first sample：25.263 ms

抽帧工具：

`tools/extract_video_frames_for_teacher.py`

## 最终候选对比

收益比采用：

`SUP = geometry speedup × coverage utility ratio`

其中：

- `geometry speedup = teacher_moge3_ms / student_inference_ms`
- `coverage utility ratio = student_coverage_mean / teacher_coverage_mean`
- teacher MoGe-3 采用 v6_2k mean-skip-first：25.263 ms

| 模型 | eval loss | point | projection | inference ms | coverage gap | yaw10 gap | worst gap | speedup | coverage ratio | SUP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| occ075_width80 | 0.1497 | 0.00440 | 0.00901 | 10.47 | -0.0713 | -0.0650 | -0.1975 | 2.41x | 0.914 | 2.21 |
| occ05_width80 | 0.1521 | 0.00468 | 0.00942 | 10.47 | -0.0549 | -0.0486 | -0.2137 | 2.41x | 0.934 | 2.25 |
| occ075_width64_lr15 | 0.1622 | 0.00528 | 0.01042 | 7.59 | -0.0577 | -0.0515 | -0.2001 | 3.33x | 0.931 | 3.10 |

主模型选择：`occ075_width64_lr15`。

理由：

- 速度收益最高：3.33x geometry speedup。
- SUP 最高：3.10。
- 平均 coverage gap 只比 `occ05_width80` 低约 0.0028，但速度明显更快。
- 在部分 hard scene 上比 `occ05_width80` 更稳，例如 `game3_left`、`game3_mid_left`、`game3_right`。

`occ05_width80` 可作为覆盖率优先版本，但 worst-case 更差，不建议作为默认交付。

## 已发现 hard cases

主要失败模式：

- `game2_mid_left__frames__frame_000300`
  - 三个候选均有明显覆盖率缺口。
  - width64 主模型 worst gap：-0.2001。
- `game3_left__frames__frame_000448`
  - `occ05_width80` worst gap 达 -0.2137。
  - width64 主模型在该场景更稳，但仍有 -0.1316 的缺口。

这些 hard cases 说明：单纯扩大训练样本或降低几何 loss 不一定改善重投影可用性。下一轮应做 hard-case mining / motion-aware sampling，而不是只继续堆 epoch。

## 当前模块作用

M1 当前提供：

1. 快速 3D point map 预测，用于替代每帧 MoGe-3 teacher 推理。
2. valid mask / warp-confidence，用于下游重投影时过滤不可靠区域。
3. target-view occupancy-aware 训练，使模型目标从“source-view 几何拟合”转向“reprojection utility”。
4. 对多 yaw + forward motion 的重投影覆盖率评估接口，可直接给后续同学选择模型或做 router。

适用场景：

- 视频生成 / 世界模型中的连续帧几何缓存。
- 中小视角变化的 forward reprojection。
- 游戏、合成视频、动态相机但主体结构相对连续的场景。
- 需要用轻量 geometry prior 替代高成本 DiT 或高成本 3D foundation model 重复推理的 pipeline。

暂不应过度声称适用：

- 大 baseline / 大视角跳变。
- 严重非刚体、透明/反光、大遮挡变化。
- 与训练域差异很大的真实室外/室内数据，除非补做 cross-dataset eval。

## 距离 90/100 还缺什么

必须补齐：

1. 跨数据集验证：真实视频、室内/室外、动态物体、低纹理、强遮挡。
2. hard-case mining：针对 `game2_mid_left/frame_300` 与 `game3_left/frame_448` 这类场景做 oversampling / loss reweighting。
3. downstream 闭环：证明 M1 + reprojection 能减少 DiT steps 或 token/latency，并保持图像质量。
4. ablation：
   - no occupancy
   - occupancy weight sweep
   - no warp-confidence
   - width64 vs width80
   - train scale 639 vs 1880 vs 更大规模
5. reliability calibration：warp-confidence 需要 ECE / AUC / rejection curve，证明 mask 能筛掉错误几何。

## 可复现实验记录

关键 summary 已归档到：

`results/recorded/reprojection_student_tvod_v7_m1/summaries/`

其中包含：

- `occ075_width80_eval_summary.json`
- `occ075_width80_reprojection_summary.json`
- `occ05_width80_eval_summary.json`
- `occ05_width80_reprojection_summary.json`
- `occ075_width64_lr15_eval_summary.json`
- `occ075_width64_lr15_reprojection_summary.json`


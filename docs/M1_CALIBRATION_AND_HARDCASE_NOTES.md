# M1 Reliability Calibration and Hard-Case Mining Notes

日期：2026-08-29

## 当前结论

本轮继续推进了两个审稿人会关注的问题：

1. warp-confidence 是否真的能筛掉不可重投影区域；
2. hard-case mining 是否能直接提升当前主模型。

结果是：warp-confidence 有明确筛选价值；但简单 scene-level hard-case weighting 的 fine-tune 没有改善主模型，暂不作为最终模型。

## Warp-confidence calibration

评估工具：

`tools/evaluate_warp_confidence_calibration.py`

评估对象：

`runs/reprojection_student/student_video384_tvod_v7_2k_occ075_width64_lr15_eval_best_final`

评估设置：

- teacher：`runs/teacher_moge3_video384_v6_2k`
- samples：30
- motions：4
- total cases：120
- pixel labels：teacher point map 在目标视角下的 projected-valid mask

关键结果：

| metric | value |
|---|---:|
| num_pixels | 17,523,628 |
| positive_rate | 0.7817 |
| confidence_mean | 0.8367 |
| AUC global | 0.8139 |
| AUC mean | 0.8289 |
| ECE | 0.0624 |

Threshold curve：

| threshold | kept ratio | projected-valid rate |
|---:|---:|---:|
| 0.5 | 0.8970 | 0.8225 |
| 0.6 | 0.8075 | 0.8576 |
| 0.7 | 0.7576 | 0.8757 |
| 0.8 | 0.6707 | 0.9017 |
| 0.9 | 0.5891 | 0.9224 |

解释：

warp-confidence 不是完美校准概率，但排序能力有效。阈值升高会牺牲覆盖面积，换取更高 projected-valid rate。因此它适合在后续重投影模块中作为 reliability gate / rejection mask 使用。

## Hard-case scene weighting 负结果

新增训练能力：

- `--hardcase-summary`
- `--hardcase-strength`
- `--hardcase-cap`
- `--init-checkpoint`

训练思路：

使用 v7 主模型重投影 summary 的 `by_scene.coverage_gap_mean` 构造 scene-level weight，只加权 train split 的同 scene 样本，不直接训练 val hard frame，避免验证集泄漏。

尝试模型：

`student_video384_tvod_v8_hardscene_occ075_width64_lr5_p5w025`

初始化：

`student_video384_tvod_v7_2k_occ075_width64_lr15/checkpoints/best.pt`

训练权重与原 checkpoint 保持一致：

- projection：5.0
- warp：0.25
- occupancy：0.75

前 6 epoch 结果：

| epoch | val loss | point | projection | warp | occupancy |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.1617 | 0.00508 | 0.00998 | 0.3299 | 0.02194 |
| 2 | 0.1621 | 0.00496 | 0.00981 | 0.3355 | 0.02205 |
| 3 | 0.1656 | 0.00532 | 0.01034 | 0.3404 | 0.02106 |
| 4 | 0.1632 | 0.00511 | 0.01004 | 0.3336 | 0.02138 |
| 5 | 0.1650 | 0.00506 | 0.00990 | 0.3427 | 0.02223 |
| 6 | 0.1672 | 0.00532 | 0.01037 | 0.3413 | 0.02213 |

原 v7 主模型 checkpoint val loss：0.15925。

因此该策略没有达到改进目的，已停止，作为负结果保留。

## 对下一步的技术判断

不要继续用简单 scene-level reweighting 堆训练。原因：hard frames 的问题可能不是 scene 频率不足，而是局部遮挡、几何深度边界、目标视角可见性变化、或 teacher/student mask 分布偏差。

下一步更值得做：

1. pixel-level / patch-level hard mining：对 coverage gap 对应区域采样，而不是整 scene 加权。
2. motion-conditioned confidence：当前 warp-confidence 不输入 motion，难以同时适配 yaw ±5 / ±10。
3. occlusion-aware target occupancy：把 z-buffer / depth conflict 纳入 TVOD，而不是只看 coarse occupancy。
4. confidence calibration loss：加入 Brier / ECE proxy，让 confidence 更接近 projected-valid probability。
5. downstream rejection curve：给重投影同学一个阈值表，例如 threshold=0.8/0.9 时覆盖-可靠性 tradeoff。

## 归档文件

- calibration summary：`results/recorded/reprojection_student_tvod_v7_m1/calibration/occ075_width64_lr15_calibration_summary.json`
- hard-scene negative metrics：`results/recorded/reprojection_student_tvod_v8_hardscene_negative/metrics_first6.jsonl`
- hard-scene weights：`results/recorded/reprojection_student_tvod_v8_hardscene_negative/scene_weights.json`

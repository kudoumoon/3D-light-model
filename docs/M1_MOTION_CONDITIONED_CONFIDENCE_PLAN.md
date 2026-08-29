# M1 Motion-Conditioned Warp Confidence Plan

日期：2026-08-29

## 为什么需要这个改动

上一轮 calibration 证明：当前主模型的 `warp_confidence` 有筛选价值，但它是单一无条件 confidence map。实际重投影时，目标相机运动不同，某个 source pixel 是否仍可投影会明显变化。例如 yaw +10 和 yaw -10 的可见区域不同，单一 confidence 只能学到平均可见性。

这限制了两个方面：

1. confidence calibration 不够精确；
2. 后续 reprojection 模块无法根据真实 motion 调整 rejection mask。

## 新增方法

新增模型：`MotionConditionedReprojectionStudent`

入口：

- `tools/train_motion_conditioned_reprojection_student.py`
- `tools/evaluate_motion_conditioned_calibration.py`

结构：

1. RGB image 经过共享 encoder/decoder，输出 feature map。
2. geometry head 输出：
   - `points_scaled`
   - `mask_logits`
   - `normal`
3. motion-conditioned warp head 输入：
   - shared feature
   - motion code `[yaw / 10, forward / 0.10]`
4. warp head 输出当前 motion 对应的 `warp_logits`。

## 与 TVOD 的关系

这个方法不是替代 TVOD，而是增强 TVOD：

- TVOD 让 geometry 学到 target-view occupancy；
- motion-conditioned confidence 让 reliability mask 依赖目标视角运动；
- 下游重投影可以针对真实 camera motion 查询对应 confidence map。

## 预期收益

如果训练有效，应该主要提升：

1. AUC global / AUC mean；
2. ECE；
3. threshold curve：同等 kept ratio 下 projected-valid rate 更高；
4. hard motion，尤其 yaw ±10 的 rejection quality。

它不一定直接提升 point loss 或 raw coverage，因为 geometry head 仍然相同；主要贡献是让后续重投影更可控、更可靠。

## 训练建议

推荐配置：

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> .venv/bin/python tools/train_motion_conditioned_reprojection_student.py   --teacher runs/teacher_moge3_video384_v6_2k   --output runs/reprojection_student   --name student_video384_tvod_v9_motionconf_width64   --epochs 80   --lr 1.5e-4   --width 64   --projection-weight 5.0   --warp-weight 0.25   --occupancy-weight 0.75   --device cuda
```

评估：

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> .venv/bin/python tools/evaluate_motion_conditioned_calibration.py   --teacher runs/teacher_moge3_video384_v6_2k   --checkpoint runs/reprojection_student/student_video384_tvod_v9_motionconf_width64/checkpoints/best.pt   --output runs/reprojection_student/student_video384_tvod_v9_motionconf_width64_calibration   --per-scene 3   --device cuda
```

## 成功标准

以 v7 主模型 baseline 为对照：

- baseline AUC global：0.8139
- baseline AUC mean：0.8289
- baseline ECE：0.0624
- baseline threshold=0.8 projected-valid rate：0.9017
- baseline threshold=0.9 projected-valid rate：0.9224

可接受改进：

- AUC global >= 0.84，或
- ECE <= 0.05，或
- threshold=0.8 projected-valid rate 提升至少 1.5 points，同时 kept ratio 不大幅下降。

如果达不到这些标准，该 idea 仍可作为方法路线，但不能作为主贡献实验。

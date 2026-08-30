# M1 Method Draft: Reprojection-Friendly Geometry Module

## 论文叙事定位

M1 turns monocular geometry from passive depth prediction into an action-conditioned compute-allocation signal for real-time world generation. 论文写作中应把它定位为“面向重投影复用的几何路由模块”，而不是普通 monocular depth estimator。

## 模块目标

M1 的目标不是复刻 MoGe-3 的完整能力，而是提供一个更适合下游重投影加速的轻量几何模块。给定单帧 RGB，M1 输出：

- point map：每个像素对应的 camera-space 3D point；
- depth / valid mask / normal：供 forward splatting、遮挡判断和几何一致性检查使用；
- motion-conditioned warp confidence：给定候选相机运动，预测该像素在目标视角中是否可安全重投影。

下游使用方式是：高置信几何区域直接 warp/reuse，低置信区域交给更重的 DiT/refiner。这样 M1 的评价应围绕 geometry usability 和 downstream acceleration，而不是单独追求 depth MSE。

## Teacher-student 几何蒸馏

Teacher 采用 MoGe-3，离线导出每帧的 point map、depth、mask、normal 和 intrinsics。Student 采用轻量 CNN encoder-decoder，在 384 resolution 下训练，输出与 teacher 同 contract 的 geometry.npz。

基础监督包括：

1. Point loss：对 valid teacher pixels 做 smooth L1 point-map regression。
2. Mask loss：预测 source-view valid geometry mask。
3. Normal loss：保持局部 surface orientation。
4. Edge loss：约束 depth gradient，减少几何边缘被过度平滑。

## Reprojection-friendly supervision

仅匹配 source-view point map 不足以保证 target-view 可重投影。因此训练中采样虚拟相机运动，包括 yaw 和 forward translation，把 teacher/student point map 投影到目标视角，并加入两类任务相关监督：

1. Projection loss：约束 student point 在目标视角的 projected coordinate 与 teacher 一致。
2. Target-view occupancy loss：在 coarse target grid 上比较 teacher/student 的 splatted occupancy，减少 target-view holes。
3. Warp-valid confidence：预测该 source pixel 在给定 motion 下是否仍位于 target view 内且位于相机前方。

这使得 M1 直接优化下游 forward reprojection 所需的几何条件。

## Motion-conditioned confidence

v10 采用冻结几何 + 单独训练 motion-conditioned confidence head 的两段式设计：

- 冻结 v7 base geometry，避免 joint fine-tuning 破坏已得到的 point-map 质量；
- 输入目标 motion encoding，输出 motion-specific warp confidence；
- 用 projected-valid label 训练，使 confidence 可以作为下游 tile/pixel gating 信号。

当前结果显示 v10 confidence AUC_global 0.9525、ECE 0.0221，threshold 0.8 时 kept_ratio 0.7489、projected-valid rate 0.9597。

## Hard-case coverage repair

本轮 v11 增加两个训练目标，解决最差 coverage gap：

1. Coverage-deficit loss：只惩罚 `student target occupancy < teacher target occupancy` 的区域，避免模型通过过度保守 mask 获得好看的平均误差，却在 target view 留洞。
2. Depth-edge-aware point loss：在 teacher depth discontinuity 高的像素加权，强化遮挡边界、薄结构、几何突变区域。

这两个目标的创新点是把训练关注点从 source-view geometry fidelity 转向 target-view reusable coverage，符合论文的加速目标。

## 下游闭环接口

M1 给下游提供三类信息：

- dense point map：用于 camera transform + forward splat；
- source/motion confidence：用于决定哪些区域可直接复用；
- valid/normal/depth：用于过滤遮挡、边界和不稳定区域。

下游可实现为 gated pipeline：

1. 估计目标运动或使用控制动作；
2. 用 M1 得到 point map 与 motion confidence；
3. 高置信区域重投影并复用；
4. 低置信区域进入 DiT/refiner；
5. 用 active token ratio 评估速度收益，用 target-frame quality 评估质量收益。

## 泛化性预期

适用场景：

- 单目 RGB 可推断几何结构的 3D/游戏/室内/街景类场景；
- camera motion 主导的交互式世界模型；
- 短时序或 streaming 场景中相邻视角变化可由重投影解释的区域；
- 需要用 geometry cache 减少 DiT 计算的长视频/世界模型系统。

风险场景：

- 大量非刚体动态物体；
- 透明/反射/低纹理区域；
- 极端新视角导致大面积 disocclusion；
- teacher 自身几何不可靠的 out-of-domain 图像。

这些风险需要通过 confidence gating 暴露给下游，而不是用代码兜底掩盖。

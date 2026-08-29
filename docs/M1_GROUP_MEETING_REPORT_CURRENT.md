# M1 组会汇报草案：3D 几何模型与重投影友好扩展

## 一句话结论

我们当前 M1 已经形成可交付方案：**v7 提供稳定快速 point map，v10 在冻结几何的基础上提供 motion-conditioned warp confidence**。它比单纯 MoGe-3 蒸馏更有论文点，但还未达到我认为的 A 会 oral 级，需要补跨域和下游闭环。

## 主要结果

| 指标 | v7 baseline | v10 frozen geometry + motion confidence |
|---|---:|---:|
| point loss | 0.005281 | 0.005281 |
| projection loss | 0.010421 | 0.010543 |
| mean coverage gap | -0.057653 | -0.057653 |
| worst coverage gap | -0.200120 | -0.200120 |
| confidence AUC | 0.8139 | 0.9525 |
| ECE | 0.0624 | 0.0221 |
| 推理速度 | 7.59ms | 10.56ms |

## 方法路线

1. 用 MoGe-3 导出 teacher 几何。
2. 用 TVOD 训练轻量几何 student：不仅拟合 point map，还通过 target-view occupancy loss 让几何更贴近重投影目标。
3. 加入 motion-conditioned confidence：输入目标视角运动，输出每个像素在该运动下是否适合重投影。
4. 发现联合微调 v9 会损伤几何 coverage，因此改成 v10：冻结 v7 几何，只训练 motion confidence head。

## 论文可讲的创新点

- **Target-View Occupancy Distillation (TVOD)**：几何训练目标从 source-view point regression 扩展到 target-view occupancy，对重投影更直接。
- **Motion-conditioned Warp Confidence**：不是输出单一静态置信度，而是对不同目标相机运动输出不同可靠性 map。
- **Frozen-Geometry Reliability Adaptation**：把“几何精度”和“重投影可用性预测”解耦，避免 confidence 训练破坏 point map。

## 当前不足

- hard-case coverage gap 仍到 -0.20。
- 数据域还偏游戏视频，缺少真实场景泛化。
- 尚未与 downstream DiT/reprojection 模块做闭环收益。
- v10 confidence head 有额外延迟，需要工程优化。

## 下一步

1. 扩展真实数据 teacher export。
2. 做 hard-case mining：边界、快速运动、大遮挡区域单独加权或 patch loss。
3. 与重投影同学对接 confidence gating API，实测生成质量/速度收益。
4. 做 v10 head batch-motion 优化，把多 motion confidence 一次性输出。

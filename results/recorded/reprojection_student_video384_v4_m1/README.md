# M1 几何模型：Video384 v4/v4b 扩大训练实验

日期：2026-08-29  
目标：训练一个以 MoGe-3 为 teacher 的快速几何模型，输出可用于后续重投影模块的 3D 信息（point map / depth / mask / normal / confidence）。

## 结论

本轮已完成 M1 的扩大训练版本。当前推荐作为 M1 交付模型的是：

`student_video384_warp_v4b_proj5_width64`

它相比上一轮 v3 的核心改进：

- teacher 数据从 188 帧扩大到 639 帧；
- scene 从 2 个 crop 扩大到 6 个 crop；
- 模型宽度从 48 提升到 64；
- projection loss 权重从 2.0 提升到 5.0；
- warp BCE 权重从 0.50 降到 0.25，减少随机 projected-valid 分类项对几何拟合的干扰；
- 仍保留 `warp_confidence` 作为后续重投影模块的可用置信度接口。

当前模型不是 MoGe-3 teacher 的完全替代上限，但已经可以作为论文系统里的 M1 快速几何模块：以明显更低延迟输出 dense point map，让后续同学的重投影模块直接消费。

## M1 模型具备的作用

M1 的定位是“快速 3D 几何生成器（fast geometry generator）”，输入一张 RGB 帧，输出后续重投影需要的中间几何状态：

| 输出 | 作用 |
|---|---|
| `points` | 每个像素对应的 3D point map，是 forward splat / reprojection 的核心输入 |
| `depth` | 从 point map 的 z 轴得到，可用于遮挡排序、深度可视化和 z-buffer |
| `mask` | 源图中几何有效区域 |
| `normal` | 表面方向，可用于几何边缘、遮挡边界、后续 refinement |
| `source_confidence` | 源几何可信度 |
| `warp_confidence` | 预测该像素在小视角运动下是否适合被重投影 |

和直接走 MoGe-3 / DiT 类重模型相比，M1 的价值在于把“每帧重几何估计”变成“快速 student 前向 + CUDA reprojection”。这符合论文的加速方向。

## 泛化性与适用场景

当前 M1 适用范围：

- 游戏/世界模型视频帧（game / world-model frames）；
- 前向移动、轻微转向、道路/建筑/开放场景；
- 单帧 384 尺度输入；
- 小视角重投影，例如 yaw 约 ±5°、小幅 forward translation；
- 与 Matrix-Game 风格接近的场景分布。

当前 M1 尚不应过度宣称适用于：

- 大幅跨域真实世界照片；
- 大视角旋转或大 baseline novel view synthesis；
- 极近景快速运动、强动态物体、透明/反光区域；
- 需要 metric-scale 绝对几何精度的任务；
- 完全替代 MoGe-3 teacher 的高精度几何估计。

严谨表述应为：M1 是一个以 MoGe-3 为 teacher 蒸馏得到的快速 dense geometry predictor，在 Matrix-Game / world-model 视频分布内具备较好的局部泛化，能为后续重投影模块提供低延迟 point map 和置信度信号。

## 数据集

teacher 数据来自两个 Matrix-Game demo 视频，多 crop 抽帧：

- Matrix-Game-2：left / mid / right，各 100 帧；
- Matrix-Game-3：left / mid / right，各 113 帧；
- 总计 639 帧；
- 训练集 510；
- 验证集 129；
- 按每个 scene 的时间尾部 20% 做 validation holdout。

`teacher/summary.json`

| 指标 | 数值 |
|---|---:|
| teacher samples | 639 |
| scenes | 6 |
| max size | 384 |
| MoGe-3 tokens | 1200 |
| refine steps | 0 |
| valid fraction mean | 0.9846 |
| valid fraction min | 0.8577 |
| valid fraction max | 1.0000 |
| teacher ms mean | 23.40ms |

## 训练结果

### v4：width64 baseline

`v4/train/summary.json`, `v4/eval_best/summary.json`

| 指标 | 数值 |
|---|---:|
| epochs | 100 |
| best val loss | 0.19749 |
| eval loss | 0.20348 |
| eval point | 0.00649 |
| eval projection | 0.00926 |
| eval warp | 0.32987 |
| inference median mean | 8.07ms |

### v4b：projection-focused M1

`v4b/train/summary.json`, `v4b/eval_best/summary.json`

| 指标 | 数值 |
|---|---:|
| epochs | 70 |
| best val loss | 0.14759 |
| best epoch | 61 |
| eval loss | 0.15046 |
| eval point | 0.00689 |
| eval projection | 0.00950 |
| eval warp | 0.32981 |
| inference median mean | 4.83ms |
| inference p95 mean | 5.46ms |

说明：v4b 的 loss 权重和 v4 不同，因此 loss 绝对值不能只按数值比较；更关键的是 v4b 在 selected forward-warp coverage 上比 v4 更好。

## 重投影覆盖率

测试设置：

- selected validation samples = 12；
- yaw = 5°；
- forward = 0.10；
- splat radius = 1；
- CUDA forward splat；
- teacher geometry 和 student geometry 使用同一 reprojection 程序。

### 汇总

`comparison_summary.json`

| 模型 | selected student coverage | teacher coverage | gap |
|---|---:|---:|---:|
| v4 | 0.7588 | 0.9359 | -0.1771 |
| v4b | 0.8084 | 0.9359 | -0.1275 |

v4b 相比 v4 的 selected coverage 绝对提升：

`+0.0496`，约 +4.96 个百分点。

这说明提高 projection 权重是有效方向，但 student 与 MoGe-3 teacher 的 coverage 仍有差距。后续重投影模块如果使用 `warp_confidence` 做过滤/加权，可能进一步改善视觉质量，但这属于后续模块同学的工作范围。

## 速度收益

当前 MoGe-3 teacher 在 384 尺度、1200 tokens、0 refine step 下 teacher 几何生成均值约 23.40ms。

M1 v4b 几何预测均值约 4.83ms：

- 相对 MoGe-3 teacher：约 4.85x 几何预测加速；
- 若使用 v3/v4 小模型，则速度可到 2–8ms 区间，但质量/coverage 有差异；
- CUDA reprojection 本身通常约 1ms 左右，但少数 benchmark 有 28ms outlier，需要后续单独稳定计时流程。

论文里建议拆开报告：

1. geometry estimation time；
2. reprojection time；
3. total M1 + reprojection pipeline time；
4. 对比 MoGe-3/DiT-heavy baseline 的端到端耗时。

## 当前可交付模型

本地 checkpoint：

`runs/reprojection_student/student_video384_warp_v4b_proj5_width64/checkpoints/best.pt`

GitHub 不提交 `.pt` checkpoint，原因是仓库策略忽略大权重文件。GitHub 提交的是可复现实验代码、配置、metrics、summary 和代表性重投影图。

后续如需把 checkpoint 给其他同学，应通过服务器本地路径、共享盘或 release artifact 单独分发。

## 代码与记录

训练脚本：

`tools/train_reprojection_student_warp.py`

评估脚本：

`tools/evaluate_reprojection_student_warp.py`

本轮记录：

- `teacher/summary.json`
- `v4/train/`
- `v4/eval_best/`
- `v4/reprojection_selected/`
- `v4b/train/`
- `v4b/eval_best/`
- `v4b/reprojection_selected/`
- `comparison_summary.json`

## 下一步建议

如果继续提升 M1，优先级如下：

1. 数据继续扩到 2k–5k teacher frames，增加真实跨场景验证，而不是只做同视频尾帧验证。
2. 加入直接 coverage-aware loss：不仅监督 projected-valid BCE，还要约束 projected point distribution 对目标画幅的覆盖。
3. 做 `warp_confidence` 加权 splatting 实验：让后续重投影模块使用 confidence 过滤坏点。
4. 增加多 motion 验证集：yaw ±5/±10，forward 0.05/0.10/0.20。
5. 把 M1 输出格式固化成一个轻量 API，避免后续同学对接时误用 scale / coordinate convention。


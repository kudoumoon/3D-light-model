# P15b：TUM RGB-D 纯旋转 Latent Warp 验证

TUM `freiburg1_rpy` 提供固定位置、绕三个主轴旋转的真实 Kinect RGB-D 与 motion-capture trajectory。沿用 P15 完全相同的冻结 Wan VAE、640×352 crop、官方 K、四种 latent-grid pooling 和 Copy 对照。扩展实验为每个运动档 8 个 pair，共 24 个 pair。

| Motion bin | Warp latent L1 ↓ | Copy L1 ↓ | Warp 胜率 ↑ | Coverage ↑ | Composite PSNR ↑ | Copy PSNR ↑ | Composite SSIM ↑ | Copy SSIM ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-cell | 0.13856 | 0.16000 | 87.5% | 89.10% | 20.18 | 18.77 | 0.8128 | 0.8661 |
| 1–4 cells | 0.14472 | 0.22058 | 100% | 87.78% | 18.99 | 14.40 | 0.8047 | 0.7571 |
| ≥4 cells | 0.18083 | 0.36409 | 100% | 69.79% | 13.33 | 9.19 | 0.3963 | 0.0886 |

原始指标：[12-pair metrics.json](../../results/latent3d/p15_tum_freiburg1_rpy_gt_v1_standard_metrics/metrics.json)；[24-pair metrics.json](../../results/latent3d/p15_tum_freiburg1_rpy_gt_v2_8pairs/metrics.json)。

## 结论边界

- 纯旋转在中、大运动上稳定优于 Copy，支持显式 3D transport 的必要性。
- sub-cell 的 decoded SSIM 反而低于 Copy，说明小运动时 latent feature 对齐与 decoded 纹理结构不是同一个指标；正式论文必须同时报告 latent、coverage 和 decoded 指标。
- 这仍是单一室内序列。它扩展了 motion 类型，不等价于跨场景泛化。

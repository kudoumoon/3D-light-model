# P15b：TUM RGB-D 纯旋转 Latent Warp 验证

TUM `freiburg1_rpy` 提供固定位置、绕三个主轴旋转的真实 Kinect RGB-D 与 motion-capture trajectory。沿用 P15 完全相同的冻结 Wan VAE、640×352 crop、官方 K、四种 latent-grid pooling 和 Copy 对照。

| Motion bin | Warp latent L1 ↓ | Copy L1 ↓ | Warp 胜率 ↑ | Coverage ↑ | Composite PSNR ↑ | Copy PSNR ↑ | Composite SSIM ↑ | Copy SSIM ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-cell | 0.14528 | 0.17123 | 81.25% | 91.28% | 19.09 | 17.79 | 0.6750 | 0.7872 |
| 1–4 cells | 0.16238 | 0.24015 | 100% | 87.68% | 17.21 | 13.54 | 0.7393 | 0.6451 |
| ≥4 cells | 0.18390 | 0.34087 | 100% | 73.62% | 14.43 | 9.48 | 0.4301 | 0.1455 |

原始指标：[metrics.json](../../results/latent3d/p15_tum_freiburg1_rpy_gt_v1_standard_metrics/metrics.json)。

## 结论边界

- 纯旋转在中、大运动上稳定优于 Copy，支持显式 3D transport 的必要性。
- sub-cell 的 decoded SSIM 反而低于 Copy，说明小运动时 latent feature 对齐与 decoded 纹理结构不是同一个指标；正式论文必须同时报告 latent、coverage 和 decoded 指标。
- 这仍是单一室内序列。它扩展了 motion 类型，不等价于跨场景泛化。

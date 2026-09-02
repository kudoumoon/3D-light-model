# P15：TUM RGB-D 的 MoGe-free Latent Warp 硬门

## 结论

在 TUM RGB-D `freiburg1_xyz` 的真实 Kinect RGB-D、motion-capture 位姿和官方相机内参上，冻结 Wan VAE 的 source latent 可以通过显式 3D 投影得到有意义的 target latent。扩展实验使用 24 个 pair，覆盖 sub-cell、1–4 latent cells 和 ≥4 cells 三档运动，四种 pooling 共 96 次比较。中等运动与 hard-motion 的 warp latent L1 胜率均为 100%；因此 feasibility gate 通过，可以继续训练 latent geometry Student。

这条实验不调用 MoGe。它证明的是“高质量真实几何 + 冻结 VAE latent + 显式 warp”成立，不证明当前 Student 已经具备同等几何质量。

## 实验设置

- 数据：TUM RGB-D `freiburg1_xyz`，796 个完成 RGB/depth/pose 同步的 frame；最终扩展实验每个运动档 8 个 pair。
- 几何：Kinect 注册深度，深度比例 5000。
- 位姿：motion-capture camera-to-world；运行时转换为 source-camera→target-camera。
- 相机：官方 Freiburg 1 内参，执行 640×480→640×352 center crop 后同步更新 K。
- VAE：冻结 Wan2.1 VAE，RGB `[1,3,1,352,640]`，latent `[1,16,44,80]`。
- 对照：Copy；Warp 与 Copy 只在完全相同的 projected-valid support 上比较。
- 空洞：decoded composite 只用 source latent 填补，绝不读取 target latent。
- pooling：center、average、median、minimum。

原始指标：[12-pair metrics.json](../../results/latent3d/p15_tum_freiburg1_xyz_gt_v4_standard_metrics/metrics.json)；[24-pair metrics.json](../../results/latent3d/p15_tum_freiburg1_xyz_gt_v5_8pairs/metrics.json)。以下主表采用 24-pair 扩展结果。

## 主要结果

| Alignment | Warp latent L1 ↓ | Copy L1 ↓ | Warp 胜率 ↑ | Warp cosine ↑ | Coverage ↑ | Composite PSNR ↑ | Copy PSNR ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| center | 0.15826 | 0.24234 | 91.7% | 0.93184 | 79.51% | 17.65 | 14.34 |
| average | 0.15555 | 0.24214 | 100% | 0.93430 | 81.80% | 18.08 | 14.34 |
| median | 0.15546 | 0.24216 | 100% | 0.93452 | 81.76% | 18.09 | 14.34 |
| minimum | **0.15554** | 0.24227 | **100%** | **0.93449** | 81.67% | **18.13** | 14.34 |

minimum 在这一序列上略优，但与 average/median 的差距很小。现阶段可把 minimum 视为 R0 最优固定条件，不能据此宣布它对所有数据域最优；边界和飞点更多的数据集可能改变排序。

## 按运动量分层

| Motion bin | Warp latent L1 ↓ | Copy L1 ↓ | Warp 胜率 ↑ | Coverage ↑ | Composite PSNR ↑ | Copy PSNR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| sub-cell | 0.13393 | 0.15862 | 93.75% | 89.16% | 20.31 | 18.50 |
| 1–4 cells | 0.14709 | 0.22150 | 100% | 86.20% | 19.00 | 14.45 |
| ≥4 cells | 0.18759 | 0.34656 | 100% | 68.20% | 14.64 | 10.08 |

运动越大，Warp 相对 Copy 的优势越明显，但 coverage 从 89.2% 下降到 68.2%。这正好限定了 M1/M2 分工：M1 负责可见 cell 的几何复用和 confidence，M2 负责大 disocclusion 的 repair/regenerate。

## 审计与限制

1. 目前只有一个真实室内序列，不是跨场景泛化证据。
2. pair 通过 GT-motion proxy 分层，选择发生在读取 target latent 之前，没有按结果挑样本。
3. 当前 SSIM 是全局统计版本，LPIPS 未安装；正式表格需要补标准 windowed SSIM 和 LPIPS。
4. Freiburg 1 使用官方 pinhole K，尚未显式建模残余畸变。
5. decoded PSNR 在 `[-1,1]` 归一化张量上使用统一口径；当前只作同实验内的相对比较。
6. 失败的 v1/v2 分别暴露 Wan encode/decode 返回 CPU 的接口行为，均未产生可引用指标；v3 才是有效结果。

## 决策

- A. latent warp 是否成立：在这一真实 GT 序列上成立。
- B. 当前最优 alignment：minimum，但优势很小，需要跨数据集复核。
- C. 是否继续 Student：可以；下一阶段使用真实 sensor depth target，MoGe 只补无 GT 区域。
- D. 是否已经达到泛化结论：没有。至少还需要 TUM rotation、Bonn dynamic 和一个 scene-disjoint 大规模室内集合。

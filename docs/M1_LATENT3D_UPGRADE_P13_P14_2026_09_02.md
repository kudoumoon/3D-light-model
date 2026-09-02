# M1 Geometry-Aligned Latent 3D：P13–P14 升级实验记录

更新日期：2026-09-02

## 结论

本轮没有替换当前 `64×3` M1 checkpoint。

直接使用估计位姿做 target-latent 重投影微调没有通过验证集，三种设置都回退到 epoch 0。虚拟 6DoF projective behavior distillation 在一个随机种子上有效，但三种子结果不稳定：全 pair warp L1 平均下降 0.57%，hard-motion warp L1 平均变化接近零，valid IoU 三个种子均未改善。因此，P14 目前是值得保留的研究假设，不是已经成立的主方法。

冻结的 M1 输出接口没有变化：

- `latent_depth`: `[B, 1, H_l, W_l]`
- `latent_points`: `[B, 3, H_l, W_l]`
- `latent_valid`: `[B, 1, H_l, W_l]`
- `latent_confidence`: `[B, 1, H_l, W_l]`
- `intrinsics`: `[B, 3, 3]`

## 实验前提

训练数据仍来自 Matrix-Game 2/3 官方展示视频，共 1880 个抽帧样本、10 个 crop scene。现有 source-target pair 的相对位姿由 MoGe point map 辅助 SIFT + PnP-RANSAC 估计，不是 GT pose。

114 个 pair 通过预注册可靠性门槛：`inliers ≥ 200`、`inlier_ratio ≥ 0.6`、`median reprojection error ≤ 1.5 px`。按场景严格拆分后，训练、验证、测试分别为 69、25、20 个 pair。运动幅度至少 1 RGB pixel 的 pair 只有 32 个，其中测试集 6 个。

这些数据能用于机制筛查，不能支持真实场景泛化、GT-pose 或跨 VAE 结论。由于本轮已经查看测试场景结果，后续方法选择也不能继续把这 10 个场景当作未见测试集。

## P13：直接 target-latent reprojection

三组实验共享相同初始化、场景拆分和训练轮数：

1. 仅几何微调，控制额外训练步数的影响；
2. 几何监督 + target latent L1；
3. 几何监督 + target latent L1 + cosine。

三组实验的验证集最优 checkpoint 都是 epoch 0。最终测试指标完全一致：all-pair warp L1 为 0.18028，hard-motion warp L1 为 0.21501。

这个结果说明，当前估计位姿 pair 不适合直接优化 M1。训练 pair 中多数运动小于一个像素；位姿误差、动态内容和曝光变化会共同进入 target latent loss。继续增大 loss 权重只会放大噪声，不能解决监督定义问题。

状态：**Negative Result**。

## P14：虚拟 6DoF projective behavior distillation

P14 不再把另一个真实帧的 latent 当作训练目标。对每个 source latent 采样已知的虚拟 yaw 和 translation，分别使用 teacher geometry 与 student geometry 搬运同一个 frozen VAE latent，再比较：

- target-grid projected coordinates；
- warp 后的 latent feature；
- coverage gap。

这样可以去除 PnP 位姿误差和 source-target 外观变化，只检查 student 是否学会 teacher 的投影行为。

三种子聚合结果如下。负值表示误差下降；valid IoU 正值才表示改善。

| 指标 | 平均相对变化 | 改善种子数 |
|---|---:|---:|
| virtual feature L1 | +0.29% | 1/3 |
| virtual coordinate L1 | -1.23% | 1/3 |
| virtual coverage gap | -5.84% | 2/3 |
| geometry AbsRel | -2.51% | 1/3 |
| geometry projection L1 | -1.64% | 1/3 |
| valid IoU | -0.49% | 0/3 |
| estimated-pose all-pair warp L1 | -0.57% | 2/3 |
| estimated-pose hard-motion warp L1 | -0.01% | 1/3 |

seed 7 的结果较好：geometry AbsRel 下降 9.29%，projection L1 下降 6.78%，hard-motion warp L1 下降 1.02%，但 valid IoU 下降 0.27 个百分点。seed 11 出现 geometry 退化，seed 23 则由验证集回退到初始化。

状态：**Hypothesis**。当前证据只说明 projective behavior 是可能有效的训练目标，尚未证明它能稳定升级 M1。

完整机器可读汇总位于：

`results/latent3d/p14_upgrade_cross_seed_summary_v1/metrics.json`

## 如何降低对 MoGe 的依赖

现阶段的准确表述是：MoGe-3 是离线伪标签来源，不是 M1 架构，也不参与推理。Student 直接读取 frozen world-model VAE latent，参数量和推理路径均独立于 MoGe。尽管如此，1880 个训练目标全部来自 MoGe，审稿人仍可以把当前方案概括为“MoGe 蒸馏到 VAE latent”。仅强调推理时不用 MoGe，无法消除这个问题。

更稳妥的路线是把方法定义为 **teacher-agnostic projective distillation**：

1. 统一监督契约，只接收 `depth / valid / K / coordinate convention`，不把任何 MoGe 内部模块写入 M1；
2. 训练源允许来自 GT RGB-D、LiDAR、stereo/SfM、多视图重建或单目基础模型；
3. 几何监督负责 warm start，虚拟相机 projective behavior 负责学习可搬运性；
4. 在 GT pose 多视图数据上加入真实 target-latent reprojection，作为最终下游目标；
5. 几何收敛后冻结主干，再训练 motion-conditioned confidence，输出可复用区域与需要 repair 的区域。

必须补齐以下 teacher-source ablation：MoGe-only、GT/RGB-D-only、stereo/SfM-only、mixed-source、MoGe warm-start + projective objective，以及无单目 teacher 的 multi-view latent objective。只有 GT/mixed/no-MoGe 设置仍能工作，才能有证据支持 teacher-agnostic claim。

## 与最新 latent 3D 工作的边界

不能再把“把 VAE token 提升到 3D 并直接 latent warp”写成我们的独有创新。LSM-World 已经使用外部 RGB depth reconstructor，把 latent cell 反投影到 3D cache，并在 latent grid 上做 readout。其默认深度来自 Depth Anything 3，同时还比较了其他深度模型。[Latent Spatial Memory for Video World Models](https://arxiv.org/abs/2606.09828)

我们的可区分问题应收敛为：

- 直接从 frozen VAE latent 预测 latent-grid geometry，推理时不运行 RGB depth reconstructor；
- 显式定义 Video VAE temporal compression 后每个 temporal latent 的几何归属；
- 用 projective behavior 与真实 target-latent consistency 训练“适合搬运”的几何，而不只拟合 depth；
- 输出校准的 motion-conditioned confidence，让 M2 区分 reuse 与 repair；
- 保持轻量 2D slice head 和冻结接口，不把 M1 改造成大规模时序网络。

MoGe 本身通过 affine-invariant point map 和混合数据监督提供强几何先验，因此把它作为 cold-start teacher 是合理的；问题在于论文是否能证明方法不依赖某个特定 teacher。[MoGe](https://arxiv.org/abs/2410.19115)、[MoGe-2](https://arxiv.org/abs/2507.02546)

## 下一轮实验门槛

下一轮不应继续在当前 10 个 crop scene 上调参。需要新数据满足以下最低条件：

- 真实视频或真实多视图图像，不是 world-model 推理输出；
- GT/可靠 camera pose；
- train/validation/test 按原始场景隔离；
- test 在方法冻结前不可见；
- 至少覆盖室内、室外、驾驶、动态物体和大视差；
- 同时记录 geometry、latent warp、decode、coverage、confidence、latency 和 VRAM。

在新数据上，P14 至少需要三种子满足：hard-motion warp L1 稳定下降、valid IoU 不显著下降、geometry 指标不退化。否则继续保留现有 `64×3` baseline。

## 当前审稿判断

以 A 会 oral 标准看，当前 M1 约为 74/100：系统接口、VAE 对齐、renderer 验证、强运动 Warp>Copy 和效率证据较完整；主要缺口是最新工作的创新重叠、真实 GT-pose 数据、跨 VAE/跨域泛化，以及 projective objective 的跨种子稳定性。

本轮没有通过扩大模型容量或堆叠 loss 掩盖这些缺口。下一阶段最有价值的投入不是继续增加 H100 训练时长，而是获得新的 GT-pose 多视图证据，并完成 teacher-source 与跨 VAE ablation。

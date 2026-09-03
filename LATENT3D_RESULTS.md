# Geometry-Aligned Latent 3D：当前结果与证据边界

更新时间：2026-09-02。这里是 M1 主线的机器可查阅结论，不替代各实验目录中的原始 `metrics.json`。

## Fact

- Matrix-Game 使用的冻结 Wan VAE 将 `[B,3,T,352,640]` 压缩为 `[B,16,F,44,80]`，空间压缩率 8、时间压缩率 4。单帧 round-trip 为 PSNR 30.3144 dB、SSIM 0.99743。
- latent renderer 已通过 identity、整数 cell 平移、局部 z-buffer、尺度等变性和梯度检查。
- 在受控 teacher geometry 实验中，16/16 有明显运动的样本均优于 Copy；在估计位姿的 hard-motion 筛选中，9/9 优于 Copy。25 个以小运动为主的样本中，warp 胜率只有 28%–32%，说明运动分层是必要条件。
- `64×3` latent geometry head 有 231,746 个参数，在 H100 上约 1.0–1.2 ms。扩大到 `128×3` 没有稳定收益，因此当前不采用更大的 head。
- 只读 M1→M2 桥接的 6 个受控样本中，M1 warp latent L1 为 0.09799，Copy 为 0.23470。它证明接口和机制能工作，不等于真实世界模型端到端收益。
- P14 的 virtual 6DoF projective behavior distillation 在 seed 7 上改善 virtual feature L1 1.64%、coordinate L1 6.20%、held-out AbsRel 9.29%；但跨 seed 的 all-pair warp L1 平均仅改善 0.57%，hard-motion 均值接近零，valid IoU 为 0/3 seeds 获胜。因此该方案尚未晋升为默认配置。
- P15 在真实 TUM `freiburg1_xyz` 上完成了 12 个 GT pose/depth pair；minimum pooling 的 warp latent L1 为 0.14997，Copy 为 0.24350，decoded composite PSNR 为 17.93 dB，Copy 为 14.24 dB；这是 MoGe-free feasibility gate 的正结果，不是泛化结论。
- P15b 在真实 TUM `freiburg1_rpy` 纯旋转序列上，中等和 hard-motion 的 warp latent L1 胜率均为 100%，但 sub-cell decoded SSIM 低于 Copy，说明必须同时报告 latent 与 decoded 指标。
- P15 扩展到 24 个 pair 后，`freiburg1_xyz` 的 minimum pooling warp latent L1 为 0.15554、Copy 为 0.24227，胜率 100%；hard-motion 的 warp L1 为 0.18759、Copy 为 0.34656，coverage 为 68.20%。
- P15b 扩展到 24 个 pair 后，`freiburg1_rpy` 的 minimum pooling warp latent L1 为 0.15425、Copy 为 0.24823，胜率 100%；hard-motion coverage 为 69.79%.
- P16 三 seed 的 `Projection + TVOD` Student 已完成，但测试 AbsRel 均值约 0.3072；对应的 P17 基础几何损失对照均值约 0.2689，projection L1 约 0.002326 vs 0.002409。当前证据不支持把现有高权重 transport loss 作为默认训练配置。
- P18 低权重消融（projection=10, tvod=0.1, seed=7）已完成：AbsRel 0.26144、projection L1 0.002303、valid IoU 0.99028、延迟 1.44 ms。相对同 seed 基础组仅为轻微改善，支持继续做低权重多 seed 验证，但尚不足以晋升默认配置。
- P15c 使用不同的 `candidate_stride=3` 独立采样，在 `freiburg1_xyz` 的 24 个 pair 上仍得到正收益：minimum pooling warp latent L1 0.15654 vs Copy 0.26142，胜率 91.7%，decoded PSNR 17.37 vs 13.54 dB；该结果不与主表合并，用于检验采样敏感性。

- P20 权重扫描（seed=7）显示 projection=3、TVOD=0.03 是当前折中候选：AbsRel 0.2682、projection L1 0.002124、occupancy L1 0.006873；但相较基础组的深度优势并不成立，下一步必须进行多 seed 复验。
- P21 候选配置 projection=3、TVOD=0.03 的 seed=11 复验完成：AbsRel 0.27615、projection L1 0.002363、occupancy L1 0.007628、valid IoU 0.98830、延迟 1.12 ms。与 seed=7 的结果接近，但 projection 指标存在波动，仍需 seed=23 后再决定默认配置。
## Negative Result

- 把 target latent 直接用于训练的 P13 L1 / L1+cos 条件均在 epoch 0 最优，没有证明相对 control 的稳定收益。
- 单纯提高模型宽度和深度不能替代 supervision quality；`128×3` 的跨 seed 表现不稳定。
- 小运动样本上 Copy 很强。若不按 latent-cell motion 分层，平均指标会掩盖几何只在可测运动下产生增量价值。
- 目前 1880 帧主要训练数据来自 MoGe-3 伪标签和 Matrix-Game 展示视频，不能支持跨真实场景泛化结论。
- P14 的 projective behavior distillation 目前是机制证据，不是已完成的 A 会主结果。

## Hypothesis

- 以 sensor/laser depth + calibrated pose 为主监督、MoGe 仅补充无 GT 区域，能降低 teacher bias，并让 novelty 落在 VAE-native latent geometry 和 transport-aware objective 上。
- geometry pooling 应由 downstream latent transport 选择，而不是只比较 depth error；边界区域可能需要 surface-aware selection，但必须在真实数据上胜过 median/minimum 才能保留。
- confidence 的价值主要在 disocclusion、depth conflict、动态区域和大运动，而不是为所有 cell 给一个统一置信分数。

## 当前判断

1. Bonn dynamic 与更多真实场景上的 GT pose/depth latent warp 验证；TUM 的 feasibility gate 已通过。

1. MoGe-free 的真实 GT pose/depth latent warp 硬门；
2. 至少两个真实数据域上的 scene-disjoint 验证；
3. transport-aware loss 的跨 seed 稳定正收益；
4. 动态和边界 hard cases 的 confidence 闭环；
5. 与外部 RGB-depth lifting 方法在参数、延迟和质量上的公平对比。

在这些证据补齐前，论文可把方法定义为“Geometry-Aligned Latent 3D”，但泛化与端到端加速只能写为待验证目标。

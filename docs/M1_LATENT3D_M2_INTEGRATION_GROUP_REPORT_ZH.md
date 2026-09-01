# M1 组会汇报：Geometry-Aligned Latent 3D

日期：2026-09-02  
范围：M1 latent 3D 升级、Teacher feasibility、M1→M2 只读联调。M2 源码未修改。

## 一句话结论

我们已经证明，Frozen Wan VAE 的 spatial latent 可以和显式 3D 几何对齐，并在有明显相机运动的样本上改善重投影结果。当前证据足以支持继续训练 latent geometry head，但还不足以声称模型已经具备跨场景泛化能力，也不足以宣称 M1 已达到最终交付标准。

## 1. 技术路线

### M1 当前升级路线

```text
上一帧 RGB
   ↓ Frozen Wan VAE Encoder
z_t ∈ R^[B,16,44,80]
   ↓ Lightweight Latent Geometry Head
latent depth + valid logits
   ↓ analytic back-projection with K
latent point map (X,Y,Z)
   ↓ frozen-geometry motion confidence head
latent valid / motion-conditioned confidence
   ↓ bilinear forward splat + target-cell local z-buffer
warped latent ẑ_{t→t+1}
```

M1 预测 depth 和 valid support，point map 由 depth 与归一化内参 `K` 解析得到。这样避免自由预测 XYZ 带来的尺度和投影不一致。输出 contract 为：

```text
latent_depth       [B,1,44,80]
latent_points      [B,3,44,80]
latent_valid       [B,1,44,80]
latent_confidence  [B,1,44,80]，完成 motion head 后提供
intrinsics         [B,3,3]
spatial_downsample 8
temporal_downsample 4，当前只记录，不处理 temporal latent
```

几何 head 与 motion confidence head 分开训练。几何冻结后再训练 motion confidence，避免 confidence loss 改坏 point map。

### M1→M2 接口

M2 使用三 latent chunk：`[B,16,3,44,80]`，后续 chunk 对应 RGB horizon `(4,8,12)`。M1 提供单个 source latent frame 的 latent geometry，M2 根据 action-conditioned pose 生成三个 horizon 的 candidate。

本次没有把 M2 仓库复制进 M1，也没有修改 M2。M1 侧新增 `latent_m2_bridge.py`，将 M1 输出转成 M2 现有 `geometry_pose_candidate` 所需的 `points / depth / mask / intrinsics` 字典，并保留 confidence 和 provenance。联调使用的 M2 commit 是：

```text
49fc1bb804a9c166170dc5fda126e2ac7377870d
```

联调前后 M2 工作树均为 clean，说明本轮没有改动 M2。

## 2. 已完成实验

| 实验 | 设置 | 结果 | 结论边界 |
|---|---|---|---|
| Renderer 单元与解析验证 | identity、整格平移、local z-buffer、梯度 | 8/8 checks 通过 | M1 renderer 实现正确，不代表真实场景质量 |
| Controlled Frozen-VAE | 4 个跨场景样本，8/16/32/64 px 横向平移 | 16/16 Warp 胜 Copy；decoded PSNR 平均 +13.60 dB，95% CI [+12.35,+14.82] dB | 证明 spatial latent 可被显式 3D warp |
| 25 对 source-K 重估 pose screen | MoGe point + SIFT/PnP，22 对通过可靠性门槛 | `native_point_average` 的 decoded PSNR 平均 +0.066 dB，95% CI [−0.061,+0.187] dB | 近 identity 样本过多，不能作为明显运动结论 |
| Hard-motion Teacher screen | 9 对，全部为 12-frame gap，median projected displacement ≥4 px | 最佳 `minimum`：8/9 PSNR 胜、9/9 valid-L1 胜，平均 +0.620 dB，95% CI [+0.261,+0.936] dB | 真实视频 hard-motion 子集通过 Student-training feasibility gate；N=9，不能代表泛化 |
| M1→M2 只读联调 | 2 个样本，整格和半格横/纵向平移 | 整格时 M1/M2 等价；半格横移 M1 +8.56 dB、M2 +6.41 dB；半格纵移 M1 +7.10 dB、M2 +5.79 dB，相对 Copy | 接口打通；M2 的 nearest-splat 对亚格点运动有可测损失 |
| Head latency | H100 BF16，44×80 latent grid | geometry + confidence median 约 0.982 ms，合计 263,187 参数 | 这是 head 增量延迟，不是完整 VAE→M1→M2 端到端延迟 |

## 3. 关键结果解读

### 3.1 latent 是否真的可以重投影？

答案是：在几何和 pose 正确时，可以。

受控实验把同一 source RGB 做成已知平面相机平移，source 与 target 分别经过同一个 Frozen Wan VAE。对于 8–64 px 的位移，latent warp 在所有 16 组实验上都优于 latent Copy；位移越大，Copy 误差越明显，Warp 的相对收益仍保持为正。

真实视频实验中，25 对短间隔样本大多接近 identity，因此总体收益不显著。重新构造 12-frame hard-motion 集合后，RGB Teacher warp 在 9/9 样本上胜过 Copy，latent Teacher screen 的最佳 alignment 也达到正的 bootstrap 区间。这两个实验共同说明：此前的负结果主要受到样本运动不足和伪 pose 质量影响，不能直接推出 VAE latent 不适合显式 3D。

### 3.2 哪种 geometry alignment 最好？

在 hard-motion 9 对实验上，`minimum` 的 decoded composite PSNR 均值最高（+0.620 dB），但 `average`、`median`、`native_point_average` 的差距很小，且都通过同一 feasibility gate。当前不应过度解读“minimum 一定最好”；更稳妥的工程选择是保留 `native_point_average` 或 `median` 作为默认候选，再用更大 GT pose 集合确认。

P3 的 alignment 结论暂时是：RGB point map 不能简单 bilinear resize depth；应在 latent cell 上做显式 pooling，并记录有效 support。confidence-weighted pooling 和 learned pooling 仍未完成公平比较。

### 3.3 M2 联调暴露出的下游问题

M2 当前 `geometry_pose_candidate` 会：

- 将 geometry resize 到 44×80；
- 对每个 horizon 使用 `scale_pose`；
- 使用整数 nearest splat；
- 将 candidate visible mask 聚合到 2×2 token mask；
- 当前没有消费 M1 的逐 cell motion confidence。

在整格移动上，M2 与 M1 renderer 结果相同。在半格移动上，M2 的 nearest splat 明显落后于 M1 的 bilinear local-z renderer。这是 M2 renderer 的离散化问题，不能通过修改 M1 的 depth head 解决。M2 后续若要利用亚 cell 几何，需要队友单独评估其 renderer；本仓库不修改 M2。

另外，M2 已有的 full-refresh、causal action、KV state 和长时序实验有自己的失败门槛。那些失败涉及 action→pose 预测、exact clean-context commit、persistent KV 污染和 temporal latent 状态，不能归因给 M1 的单帧 geometry 输出。

## 4. 事实、负结果与假设

### Fact

- latent grid 的形状已由实际 Wan VAE 路径验证为 `16×44×80` 的单 spatial latent frame。
- `forward_splat_latent` 已修正 binary occupancy、target-cell local z-buffer 和 autograd 冲突；CPU 解析测试与梯度测试通过。
- Controlled Frozen-VAE 平移实验 16/16 胜 Copy。
- 9 对 hard-motion 真实视频 Teacher screen 的最佳 alignment 通过预设 Student-training gate。
- M1 bridge 已消费 M2 原生 `geometry_pose_candidate`，M2 commit `49fc1bb` 前后工作树均 clean。
- H100 上 latent geometry + motion confidence head 的增量参数量为 263,187，combined median latency 约 0.982 ms。

### Negative result

- 25 对普通短间隔 pose screen 的整体 Decode PSNR 没有形成显著正收益，说明小运动和伪 pose 不能支持 Student 训练。
- 大间隔 72/96/144 帧候选中，严格可靠 pose 样本极少，说明长间隔特征匹配容易失效；降低门槛会污染证据。
- 当前 hard-motion 集合只有 9 对，不能证明跨游戏、跨域或动态物体泛化。
- Latent geometry head 目前只有结构和接口测试，尚未用真实 latent teacher target 完成正式训练。
- LPIPS 尚未在本轮 latent screen 中取得有效结果，SSIM 仍是 global proxy，不应与论文标准 windowed SSIM 混用。

### Hypothesis

- 在有 GT pose/depth 的多场景数据上，latent-native geometry head 可能保留 teacher warp 的主要收益，并显著低于 RGB→MoGe 的延迟。
- 训练时加入 latent reprojection loss，可能比只拟合 MoGe depth 更适合 M2 的 downstream warp；这一点必须用 held-out pose 和 decoded quality 验证。
- 如果 M2 保留 nearest splat，M1 的亚格点优势会被部分吞掉；M1 应继续输出连续 depth/confidence，但 renderer 归因需要在 M2 侧单独解决。

## 5. 当前成熟度评估

按审稿人视角，当前 M1 latent 3D 升级约为 78–84/100 的 prototype 阶段，不应写成 A 会 oral 已完成。分项看：

- 技术路线与接口：通过。输出定义清楚，M1/M2 边界清楚。
- 可行性：通过受控实验和 hard-motion Teacher screen，但仍受 N=9 限制。
- 泛化性：未通过。缺少 GT pose/depth、多域数据、动态/反光/透明 hard cases。
- 创新性：有亮点。重点是 geometry directly aligned to VAE latent grid，并用 latent warp consistency 约束 M1；但正式创新性需要与现有 latent 3D cache / geometry-aware world model 工作比较。
- 系统收益：只完成 M1 head latency 和受控 warp quality。没有合法的 M1→M2 causal long-rollout speed/quality/state 闭环收益。
- 可复现性：较好。新实验均保存 config、pose manifest、VAE hash、GPU、commit 和 metrics；M2 只读联调有前后状态检查。

## 6. 接下来必须补的实验

1. 使用带 GT pose/depth 的真实数据，至少覆盖室内、室外、静态场景、动态物体和大视角运动。当前服务器没有可直接使用的这类数据，且磁盘剩余空间有限，不应盲目下载完整数据集。
2. 在 GT pose 下比较 Copy、center、average、median、minimum、native point pooling 和 learned pooling；统计 latent L1/cosine、decoded PSNR/SSIM/LPIPS、coverage、hole ratio 和 boundary error。
3. 用真实 VAE latent 训练第一版 geometry head，先只加入 depth、edge、projection、latent occupancy loss；确认收敛后再加入 latent reprojection loss。
4. 对 depth discontinuity、thin structure、foreground/background boundary、large yaw、translation、disocclusion、dynamic/reflective/transparent 区域单独汇报。
5. 在不改 M2 的前提下，把 M1 bridge 接入 M2 的正式 external payload；当前缺少 M2 的 P1 payload/checkpoint 联调数据，因此尚未进行合法 causal long-rollout。
6. 最后再做完整收益比：VAE encode、M1 head、pose、warp、exact context commit、decode 全链路 p50/p95，以及质量、action fidelity、KV state safety 的配对结果。

## 7. 可复核证据位置

- M1→M2 bridge：[latent_m2_bridge.py](../latent_m2_bridge.py)
- Bridge contract tests：[test_latent_m2_bridge.py](../tests/test_latent_m2_bridge.py)
- 受控 latent feasibility：[run_controlled_planar_latent_feasibility.py](../tools/run_controlled_planar_latent_feasibility.py)
- 只读 M2 联调：[run_m1_m2_readonly_bridge.py](../tools/run_m1_m2_readonly_bridge.py)
- Renderer 验证：[p2_renderer_validation_v2/metrics.json](../results/latent3d/p2_renderer_validation_v2/metrics.json)
- Controlled VAE 结果：[p2_controlled_planar_latent_v1/metrics.json](../results/latent3d/p2_controlled_planar_latent_v1/metrics.json)
- Hard-motion latent 结果：[p2p3_hardmotion_ge4px_teacher_v1/data_quality_audit.json](../results/latent3d/p2p3_hardmotion_ge4px_teacher_v1/data_quality_audit.json)
- Motion sufficiency audit：[p2p3_hardmotion_ge4px_teacher_v1/motion_sufficiency_audit.json](../results/latent3d/p2p3_hardmotion_ge4px_teacher_v1/motion_sufficiency_audit.json)
- M2 bridge 结果：[m1_m2_readonly_bridge_v1/metrics.json](../results/latent3d/m1_m2_readonly_bridge_v1/metrics.json)
- Hard-motion RGB panels：[p2_rgb_pose_geometry_hardmotion_ge4px_v1/visuals](../results/latent3d/p2_rgb_pose_geometry_hardmotion_ge4px_v1/visuals)

本报告只把上述结果作为当前阶段证据。未完成的 GT 验证、latent Student 正式训练和 M2 causal 闭环，不在本轮结论中提前承诺。

# M1 论文内容草稿：重投影友好的 3D 几何模块

本文档用于后续论文写作，不是最终投稿稿。内容以当前仓库证据为边界，重点覆盖 M1 的方法（Method）、相关工作定位（Related Work）、实验叙述骨架和局限性说明。当前不生成论文图。

## 写作边界

核心论点：在交互式世界模型中，M1 将单目几何估计从静态深度/点云预测转化为重投影可复用性预测；它以 MoGe-3 作为离线教师，训练轻量学生网络输出 point map、几何有效性和 motion-conditioned warp confidence，用于下游重投影与局部 DiT/refiner 计算分配。

当前证据边界：M1-v7 已给出稳定 point map 与重投影友好的基础几何，M1-v10 在冻结 v7 几何后训练 motion-conditioned confidence head。已有结果支持速度收益、confidence calibration 和高运动场景下的闭环 proxy 收益；跨域验证和 hard-case coverage repair 仍需要扩大实验规模后才能写成强结论。

建议论文贡献表述：

1. 提出一种重投影导向的几何蒸馏目标，使学生几何不仅拟合 source-view point map，也对 target-view projection 和 occupancy 负责。
2. 提出 motion-conditioned warp confidence，将几何模块输出从单一 3D 表示扩展为面向目标相机运动的可复用性信号。
3. 设计 hard-case coverage repair 训练项，针对遮挡边界、薄结构和 target-view hole 进行显式约束。该部分当前已实现，最终是否作为正式贡献取决于后续 v11 实验结果。

## 术语表

| 中文术语 | 英文原词 / 代码名 | 本文用法 |
|---|---|---|
| 3D 几何模块 | M1 geometry module | 本文的主模块，给下游视频/世界模型提供几何信息 |
| 教师模型 | MoGe-3 teacher | 离线导出几何伪标签，不作为在线推理路径 |
| 学生模型 | reprojection student | 轻量 CNN encoder-decoder，在线输出几何与置信度 |
| 点图 | point map | 每个像素对应的 camera-space 3D point |
| 有效掩码 | valid mask | source-view 中几何可信区域 |
| 重投影置信度 | warp confidence | 给定相机运动后，像素可安全重投影的概率 |
| 运动条件置信度 | motion-conditioned confidence | 以 yaw / forward 等目标运动作为条件的置信度预测 |
| 目标视角覆盖率 | target-view occupancy / coverage | forward reprojection 后目标视角中被覆盖的区域比例 |
| 覆盖缺口 | coverage gap | student coverage minus teacher coverage |
| 下游闭环 proxy | closed-loop proxy | 基于 safe tile 与 active-DiT microbenchmark 的速度收益近似，不等同于完整端到端结果 |

## 可直接进入论文的 Method 草稿

### 方法概述

给定单帧 RGB 图像 $I \in \mathbb{R}^{H \times W \times 3}$ 和一个目标相机运动 $a$，M1 输出 source-view 几何 $G$ 和 motion-conditioned warp confidence $C_a$。几何 $G$ 包含每个像素的 camera-space point map $P \in \mathbb{R}^{H \times W \times 3}$、深度 $D$、有效掩码 $M$ 和法向 $N$。置信度 $C_a \in [0,1]^{H \times W}$ 表示在运动 $a$ 下，当前像素对应的 3D 点能否被安全重投影到目标视角。下游模块使用 $P$ 做 camera transform 和 forward splatting，使用 $C_a$ 决定哪些区域直接复用，哪些区域交给 DiT/refiner 重新生成。

M1 的训练分为两阶段。第一阶段用 MoGe-3 离线导出的几何作为教师信号，训练轻量学生网络学习 point map、valid mask、normal 和重投影相关目标。第二阶段冻结第一阶段得到的基础几何，只训练 motion-conditioned confidence head，使置信度学习不破坏已有 point map 质量。这个设计把在线推理路径限制在轻量学生网络内，同时保留 MoGe-3 的几何先验。

### 教师几何导出

我们使用 MoGe-3 作为离线教师模型。对每个训练样本，教师导出 RGB、point map、depth、valid mask、normal 和 camera intrinsics，并统一保存为 `geometry.npz`。后续训练、评估和重投影脚本都只读取这一数据契约，避免不同实验各自定义几何格式。教师几何只用于监督学生模型；在线推理时不调用 MoGe-3。

这个设置的目的不是复现 MoGe-3 的完整能力，而是把其几何知识压缩到一个更适合交互式重投影的学生模型中。与直接在线运行教师模型相比，学生网络需要满足两个额外要求：一是推理延迟低，二是输出必须能被下游重投影接口直接消费。因此，训练目标不能只看 source-view 几何误差，还需要约束目标视角中的投影行为。

### 轻量学生几何网络

学生模型采用轻量 CNN encoder-decoder。网络输入为 RGB 图像，输出 point residual、source valid logits、normal 和 warp logits。point map 在训练时与教师 point map 对齐，valid logits 用于预测 source-view 几何是否可靠，normal 用于保留局部表面方向。该结构牺牲了大型视觉 backbone 的一部分表达能力，但换来了更低延迟和更稳定的系统集成成本。

基础几何损失包括 point loss、mask loss、normal loss 和 depth-gradient loss。设教师 point map 为 $P^T$，学生预测为 $P^S$，教师有效掩码为 $M^T$，基础点图损失可写为

$$
\mathcal{L}_{point}=\frac{1}{|M^T|}\sum_{u \in M^T} \rho(P^S_u-P^T_u),
$$

其中 $\rho$ 表示 smooth L1 loss。mask loss 使用 binary cross entropy 预测 $M^T$。normal loss 约束学生法向与教师法向的余弦相似度。depth-gradient loss 约束深度局部梯度，减少过平滑 point map 在遮挡边界产生的投影错误。

### 重投影导向监督

只拟合 source-view point map 不足以保证下游可用。一个小的 3D 误差在 source view 中可能不明显，但经过 camera transform 后会在 target view 产生 holes、错位或错误遮挡。因此，我们在训练中采样虚拟相机运动 $a$，把教师和学生的 point map 投影到同一个目标视角，并比较它们的投影坐标和覆盖行为。

给定相机运动 $a$，投影函数 $\Pi_a(\cdot)$ 将 source-view point map 转换为目标视角的 projected coordinates。projection loss 写为

$$
\mathcal{L}_{proj}=\frac{1}{|M^T|}\sum_{u \in M^T}\rho(\Pi_a(P^S_u)-\Pi_a(P^T_u)).
$$

进一步地，我们将 projected coordinates splat 到目标视角，得到学生和教师的 target-view occupancy，分别记为 $O^S_a$ 和 $O^T_a$。occupancy loss 约束两者一致：

$$
\mathcal{L}_{occ}=\rho(O^S_a, O^T_a).
$$

这一项直接对应下游需求。对于重投影系统，重要的不只是每个 source pixel 的 3D 坐标是否精确，还包括这些 3D 点能否在目标视角形成足够连续、可复用的覆盖区域。

### Motion-conditioned warp confidence

M1-v10 引入 motion-conditioned confidence head。给定目标运动 $a$，该 head 输出 $C_a$，表示每个 source pixel 在目标视角中是否 projected-valid。训练标签来自教师几何经过同一运动后的 projected-valid mask。置信度损失采用 binary cross entropy：

$$
\mathcal{L}_{conf}=\mathrm{BCE}(C_a, V^T_a),
$$

其中 $V^T_a$ 是教师几何在运动 $a$ 下得到的 projected-valid 标签。

我们没有继续使用 joint fine-tuning 作为当前主版本，因为 joint fine-tuning 可能改善 confidence，却破坏已经收敛的基础 point map 和 coverage。当前采用两阶段方案：先训练 M1-v7 得到稳定基础几何，再冻结该几何，只更新 confidence head。这种解耦让 M1 的在线接口更清晰：point map 提供几何基座，motion confidence 提供下游路由信号。

### Hard-case coverage repair

已有评估显示，M1-v10 的平均 coverage gap 可接受，但最差场景仍存在明显缺口。为此，代码中已加入两个针对 hard case 的训练项。

第一项是 coverage-deficit loss。它只惩罚学生目标视角覆盖低于教师覆盖的区域：

$$
\mathcal{L}_{deficit}=\mathrm{ReLU}(O^T_a-O^S_a).
$$

该项避免把所有 occupancy 差异等价处理，而是直接瞄准下游最敏感的问题：target-view holes。第二项是 depth-edge-aware point loss。它根据教师深度不连续程度提高遮挡边界、薄结构和几何突变区域的点图损失权重。形式上，设边界权重为 $w(D^T)$，则

$$
\mathcal{L}_{edge-point}=\frac{1}{|M^T|}\sum_{u \in M^T} w(D^T_u)\rho(P^S_u-P^T_u).
$$

这两个目标不依赖后处理兜底，也不改变评估定义。它们把 hard case 修复放在训练目标中完成，适合作为后续论文中的可消融模块。当前需要通过 v11 训练确认它们是否稳定改善 worst coverage gap。

### 总体训练目标

综合上述项，第一阶段基础几何训练目标为

$$
\mathcal{L}_{M1}=\lambda_p\mathcal{L}_{point}+\lambda_m\mathcal{L}_{mask}+\lambda_n\mathcal{L}_{normal}+\lambda_g\mathcal{L}_{grad}+\lambda_{proj}\mathcal{L}_{proj}+\lambda_{occ}\mathcal{L}_{occ}+\lambda_d\mathcal{L}_{deficit}+\lambda_e\mathcal{L}_{edge-point}.
$$

第二阶段固定基础几何参数，只优化 $\mathcal{L}_{conf}$。在论文中，$\mathcal{L}_{deficit}$ 和 $\mathcal{L}_{edge-point}$ 应作为 hard-case repair 变体报告。如果 v11 实验没有带来正收益，则正文只保留其动机和负结果，避免把未验证设计写成主要贡献。

### 下游接口

M1 的输出不是最终视频帧，而是供下游世界模型使用的几何控制信号。下游可采用 gated reprojection pipeline：先用 $P$ 做相机变换和 forward splatting，再用 $C_a$ 选择高置信区域直接复用；低置信区域、disocclusion、遮挡边界和疑似动态区域交给 DiT/refiner 处理。系统收益应同时报告质量指标和计算指标。质量侧关注 target-frame PSNR/LPIPS、warp-better-than-copy fraction、holes 和 false accept；计算侧关注 active token ratio、geometry latency、refiner latency 和端到端 latency。

## Related Work 写作草稿

### 单目几何估计与几何教师

单目深度和 3D 几何估计为图像到 3D 表示提供了基础能力。近年来的大模型式单目几何方法可以在开放场景中输出密集深度、法向或 point map，使单帧 RGB 具备可用于视角变换的几何信息。[CITATION: MoGe-3] 这类方法的优势在于泛化性强，适合用作离线教师；限制在于在线推理成本较高，并且训练目标通常以 source-view 几何一致性为主，不直接优化 target-view 重投影可复用性。M1 继承教师模型的开放场景几何先验，但把学习目标转向 projection、occupancy 和 warp confidence，使几何输出更贴近视频生成中的计算复用需求。

### 几何引导的视频生成与世界模型

视频扩散模型和世界模型通常依赖强生成 backbone 处理时序一致性、运动和新视角内容。Matrix 系列、WorldWarp 和 MiniWorld 等工作显示，几何、运动或缓存机制可以帮助生成系统减少重复计算，并改善交互式场景中的时序稳定性。[CITATION: Matrix-1.0/2.0/3.0] [CITATION: WorldWarp] [CITATION: MiniWorld] 这些系统的核心问题不是单帧几何本身，而是如何把几何信息转化为可控、可复用、低延迟的生成路径。M1 的定位正是在这一接口处：它不替代下游生成模型，而是为下游提供 dense point map 与 motion-conditioned reuse confidence，使生成模型可以把算力集中在几何不可靠或目标视角新出现的区域。

### 重投影、遮挡和可复用区域选择

重投影方法通过已知或估计的几何关系把 source view 内容映射到 target view。它在相机运动主导的场景中效率高，但对深度误差、遮挡边界和 disocclusion 敏感。传统做法常依赖后处理 mask、hole filling 或固定阈值来处理无效区域，这些规则在复杂视频和开放场景中容易影响指标含义。M1 将 projected-valid 和 target-view occupancy 纳入训练，使模型学习哪些区域在给定运动下可以复用。motion-conditioned confidence 进一步把可复用性从静态几何属性转化为与目标运动相关的路由信号。

### 与本文方法的区别

现有单目几何模型回答的是“这张图的 3D 结构是什么”。几何引导生成方法回答的是“如何把几何接入生成模型”。M1 关注二者之间的中间问题：在给定目标运动后，哪些 3D 信息值得被重投影复用，哪些区域应交给更重的生成模型处理。这个问题决定了几何模块能否真正带来速度收益，也决定了错误重投影是否会污染下游结果。

## Experiments 写作骨架

### 实验问题

实验应围绕四个问题组织，而不是按脚本运行顺序罗列。

1. M1 是否比在线 MoGe-3 更快，同时保持可用的 point map 和 projection quality？
2. motion-conditioned confidence 是否能可靠预测 projected-valid 区域？
3. hard-case coverage repair 是否减少 target-view holes，尤其是 worst-case coverage gap？
4. 在下游 gated reprojection 设置中，M1 是否带来正的速度收益，并且质量不低于合理基线？

### 当前可写入正文的结果

当前最稳的基础几何版本是 M1-v7，checkpoint 为 `runs/reprojection_student/student_video384_tvod_v7_2k_occ075_width64_lr15/checkpoints/best.pt`。该版本在当前记录中达到 point loss 0.00528、projection loss 0.01042，geometry latency 为 7.594 ms。相对于 MoGe-3 teacher 的 25.263 ms，几何推理约为 3.33x 加速。

当前最稳的置信度版本是 M1-v10，checkpoint 为 `runs/reprojection_student/student_video384_tvod_v10_frozengeom_motionconf_width64_lr1e4/checkpoints/best.pt`。该版本冻结 M1-v7 的基础几何，只训练 motion-conditioned confidence head。记录结果显示，v10 confidence AUC_global 为 0.9525，ECE 为 0.0221。在 threshold 0.8 下，kept ratio 为 0.7489，projected-valid rate 为 0.9597。这说明置信度可以筛出大约四分之三的可复用区域，并且其中绝大多数区域在教师定义下确实 projected-valid。

高运动闭环 proxy 显示，safe tile fraction 在 0.4428 到 0.7757 之间，estimated active ratio 在 0.2243 到 0.5572 之间。结合 active-DiT microbenchmark，nearest active-DiT speedup 为 1.96x 到 3.69x。warp better than copy fraction 在 0.3365 到 0.5136 之间。该结果说明在转向和较大相机运动场景中，几何重投影有潜在计算收益；但它仍是 proxy，不能替代完整 gated-DiT 端到端实验。

### 当前必须保留的负结果和边界

hard-case 评估覆盖 120 个 case。M1-v10 的 mean coverage gap 为 -0.0577，worst coverage gap 为 -0.2001。最差场景为 `game2_mid_left`，student coverage mean 为 0.7191，teacher coverage mean 为 0.8296，mean coverage gap 为 -0.1104。这说明当前模型在平均意义上可用，但遮挡边界、薄结构和较大视角变化下仍会产生 target-view holes。该结果正是 v11 hard-case repair 的实验动机。

跨域验证目前不足。当前 evidence pack 只有 1 个域、5 个样本；论文级泛化结论至少需要 3 个域、每域 30 个以上样本。当前可以写成“初步验证”或“sanity check”，不能写成强泛化结论。后续应补齐 Matrix-Game-2、Matrix-Game-3、MoGe example images 以及更多真实室内/街景/游戏域数据。

## Discussion 写作草稿

M1 的主要价值在于把几何估计变成下游生成系统可操作的计算路由信号。单独的 point map 可以支持重投影，但不能告诉系统何时应该相信重投影。motion-conditioned confidence 补上了这个接口，使下游能够根据目标运动选择复用区域，并把不可靠区域交给更强的生成模型处理。这个设计让 M1 的速度收益不是来自削弱下游模型，而是来自减少下游模型需要处理的空间范围。

当前证据支持两个谨慎结论。第一，M1-v7/v10 相比在线 MoGe-3 有明确延迟优势，并能输出可用于重投影的几何与置信度。第二，在高运动 proxy 设置中，safe tile 与 active token ratio 显示了正的计算收益空间。当前证据还不足以支持三个更强结论：模型已经跨域泛化、hard case 已被完全修复、完整 Matrix-Game 或 DiT pipeline 已获得端到端加速。这三点应作为后续实验补齐，而不是在正文中提前放大。

M1 的适用范围是 camera motion 主导、场景几何相对稳定、相邻视角变化可由重投影解释的交互式生成任务。典型场景包括游戏环境、驾驶视角、室内和建筑类视频。风险场景包括大量非刚体运动、透明/反射材料、低纹理区域、极端新视角和教师模型本身失效的 out-of-domain 图像。在这些场景中，错误的高置信重投影比保守拒绝更危险，因此后续实验应重点报告 false accept、coverage gap 和 gated-DiT 的质量退化。

## 图表与证据规划

当前不生成新图，但论文图可以按以下结构准备。

| 图/表 | 内容 | 当前状态 |
|---|---|---|
| Figure 1 | M1 pipeline：MoGe-3 teacher export、student geometry、motion confidence、gated reprojection | 待最终方案敲定后生成 |
| Figure 2 | 重投影监督示意：source point map 到 target occupancy / projected-valid | 待生成 |
| Figure 3 | qualitative cases：RGB、teacher/student warp、confidence、holes | 已有部分 warp 可视化，可整理 |
| Table 1 | latency、point/projection loss、confidence AUC/ECE | 当前可写 |
| Table 2 | ablation：无 projection、无 occupancy、无 motion confidence、hard-case repair | 需要补齐训练 |
| Table 3 | 跨域泛化：每域样本数、loss、coverage gap、latency | 当前样本不足 |
| Figure 4 | downstream proxy / end-to-end speed-quality tradeoff | proxy 已有，end-to-end 待补 |

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| M1 比在线 MoGe-3 更快 | teacher 25.263 ms，M1-v7 7.594 ms，约 3.33x | supported |
| M1-v10 confidence 可作为重投影路由信号 | AUC_global 0.9525，ECE 0.0221，threshold 0.8 下 projected-valid rate 0.9597 | supported |
| 重投影导向训练比普通 point distillation 更适合下游 | 代码和训练目标已实现 projection/occupancy/projected-valid；仍需要完整 ablation | partially supported |
| hard-case repair 能修复 worst coverage gap | coverage-deficit 和 depth-edge loss 已实现，但 v11 训练未完成 | needs evidence |
| M1 已跨域泛化 | 当前只有 1 域 5 样本 | not supported yet |
| M1 带来端到端 DiT 加速 | 当前只有 active-DiT proxy 1.96x 到 3.69x | needs end-to-end evidence |

## 后续写论文前必须补齐的实验

1. hard-case repair：跑完 v11 conservative/mid/aggressive，报告 mean coverage gap、worst coverage gap、point/projection loss 和 latency。若 coverage 改善但 point/projection 退化，需要给出 speed-quality tradeoff。
2. 跨域验证：至少 3 个域，每域 30 个以上样本。每个域报告 point loss、projection loss、coverage gap、confidence AUC/ECE 和 latency。
3. 下游闭环：实现 gated reprojection + DiT/refiner 的近端到端实验，报告 quality、active token ratio、latency 和 failure cases。
4. 消融实验：分别移除 projection loss、occupancy loss、motion-conditioned confidence、coverage-deficit loss 和 depth-edge-aware point loss，确认每个模块的作用。
5. 失败案例：保留透明/反射、动态物体、低纹理、极端视角等 case，不用后处理兜底隐藏问题。

## 可放入论文的精简贡献版本

本文提出 M1，一个面向交互式世界模型的重投影友好 3D 几何模块。与直接在线运行通用单目几何模型不同，M1 使用 MoGe-3 离线生成密集几何监督，并训练轻量学生网络输出 point map、valid mask、normal 和 motion-conditioned warp confidence。训练目标显式包含 source-view geometry、target-view projection、target-view occupancy 和 projected-valid confidence，使几何表示直接服务于下游重投影复用。当前模型在记录实验中相对于 MoGe-3 teacher 达到约 3.33x 几何推理加速，motion confidence AUC_global 为 0.9525，ECE 为 0.0221；在高运动闭环 proxy 中，active-DiT speedup 估计为 1.96x 到 3.69x。后续实验将重点验证 hard-case coverage repair、跨域泛化和完整下游闭环收益。

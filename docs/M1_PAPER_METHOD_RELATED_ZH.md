# M1 论文内容草稿：重投影友好的 3D 几何模块

本文档服务于 M1 论文写作。当前版本以仓库中已有证据为边界，整理 Method、Related Work、Experiments 与 Discussion 的中文草稿。论文图暂不生成，等最终方法和实验边界确定后再单独设计。

## 写作边界

M1 处理的是交互式世界模型中的几何接口问题。它把单目 3D 估计从静态 point map 预测，改写为面向重投影的可复用性估计：模型输出 dense point map，也输出与目标相机运动相关的 warp confidence。下游系统据此决定哪些区域直接重投影，哪些区域交给 DiT/refiner 重新生成。

当前证据支持 M1-v7 作为基础几何模型，M1-v10 作为冻结几何后的 motion-conditioned confidence 版本。已有结果可以支撑速度、置信度校准和高运动 proxy 收益。跨域泛化、hard-case coverage repair 和完整 gated-DiT 端到端收益还没有达到强结论标准，正文中必须保留这个边界。

可采用的贡献表述如下。

1. 重投影导向的几何蒸馏：训练目标同时约束 source-view point map、target-view projection 和 target-view occupancy。
2. Motion-conditioned warp confidence：模型在给定目标运动后预测像素级可复用性，为下游 active token / tile routing 提供信号。
3. Hard-case coverage repair：通过 coverage-deficit loss 和 depth-edge-aware point loss 处理遮挡边界、薄结构和 target-view holes。该部分已经接入代码，是否作为正式贡献取决于 v11 训练结果。

## 术语表

| 中文术语 | 英文原词 / 代码名 | 本文用法 |
|---|---|---|
| 3D 几何模块 | M1 geometry module | 给下游视频/世界模型提供几何信息的模块 |
| 教师模型 | MoGe-3 teacher | 离线导出几何伪标签，不进入在线推理路径 |
| 学生模型 | reprojection student | 轻量 CNN encoder-decoder，在线输出几何和置信度 |
| 点图 | point map | 每个像素对应的 camera-space 3D point |
| 有效掩码 | valid mask | source-view 中几何可信区域 |
| 重投影置信度 | warp confidence | 给定相机运动后，像素可安全重投影的概率 |
| 运动条件置信度 | motion-conditioned confidence | 以 yaw / forward 等目标运动作为条件的置信度预测 |
| 目标视角覆盖率 | target-view occupancy / coverage | forward reprojection 后目标视角中被覆盖的区域比例 |
| 覆盖缺口 | coverage gap | student coverage minus teacher coverage |
| 下游闭环 proxy | closed-loop proxy | 基于 safe tile 和 active-DiT microbenchmark 的速度收益近似，不能等同于完整端到端结果 |

## Method 草稿

### 方法概述

给定单帧 RGB 图像 $I \in \mathbb{R}^{H \times W \times 3}$ 和目标相机运动 $a$，M1 输出 source-view 几何 $G$ 与 motion-conditioned warp confidence $C_a$。$G$ 包含 camera-space point map $P \in \mathbb{R}^{H \times W \times 3}$、深度 $D$、有效掩码 $M$ 和法向 $N$。$C_a \in [0,1]^{H \times W}$ 表示像素对应的 3D 点在运动 $a$ 下能否安全投影到目标视角。下游系统用 $P$ 做 camera transform 和 forward splatting，用 $C_a$ 选择可复用区域。

训练分两步。第一步以 MoGe-3 离线几何为教师信号，训练学生网络预测 point map、valid mask、normal 和重投影相关目标。第二步冻结第一步得到的几何参数，只训练 motion-conditioned confidence head。这样做的原因很直接：joint fine-tuning 可能提高 confidence，但也可能破坏已经稳定的 point map 和 coverage；冻结几何可以把“几何基座”和“路由置信度”分开优化。

### 教师几何导出

MoGe-3 只作为离线教师使用。对每个训练样本，教师导出 RGB、point map、depth、valid mask、normal 和 camera intrinsics，并保存为统一的 `geometry.npz`。训练、评估和重投影脚本都读取同一数据契约，避免不同实验使用不兼容的几何格式。在线推理阶段不调用 MoGe-3。

这个设计不是为了复刻 MoGe-3 的全部能力，而是为了把教师的开放场景几何先验压缩成一个低延迟接口。学生模型必须同时满足两个条件：推理足够快，输出能直接进入下游重投影流程。因此，训练目标不能停留在 source-view point map 误差上，还要约束目标视角中的投影坐标和覆盖率。

### 学生几何网络

学生网络是轻量 CNN encoder-decoder。输入为 RGB 图像，输出 point residual、source valid logits、normal 和 warp logits。point map 与教师 point map 对齐；valid logits 预测 source-view 几何是否可信；normal 保留局部表面方向；warp logits 用于后续 projected-valid supervision。与更重的视觉 backbone 相比，这一结构的表达能力较低，但延迟和部署成本更适合交互式系统。

基础几何损失包括 point loss、mask loss、normal loss 和 depth-gradient loss。设教师 point map 为 $P^T$，学生预测为 $P^S$，教师有效掩码为 $M^T$，point loss 为

$$
\mathcal{L}_{point}=\frac{1}{|M^T|}\sum_{u \in M^T} \rho(P^S_u-P^T_u),
$$

其中 $\rho$ 为 smooth L1 loss。mask loss 使用 binary cross entropy 预测 $M^T$。normal loss 约束学生法向与教师法向的余弦相似度。depth-gradient loss 约束深度局部梯度，用来缓解边界过平滑带来的投影错位。

### 重投影导向监督

Source-view 几何误差不能完整反映重投影质量。某些 3D 误差在原图中很小，但经过相机变换后会在 target view 中形成 holes、错位或错误遮挡。为此，训练时采样虚拟相机运动 $a$，把教师和学生的 point map 投影到同一目标视角，再比较二者的投影坐标和覆盖行为。

令 $\Pi_a(\cdot)$ 表示由运动 $a$ 定义的投影函数，projection loss 为

$$
\mathcal{L}_{proj}=\frac{1}{|M^T|}\sum_{u \in M^T}\rho(\Pi_a(P^S_u)-\Pi_a(P^T_u)).
$$

随后将 projected coordinates splat 到目标视角，得到学生 occupancy $O^S_a$ 和教师 occupancy $O^T_a$。occupancy loss 为

$$
\mathcal{L}_{occ}=\rho(O^S_a, O^T_a).
$$

这部分监督直接对应下游需求。重投影系统关心的不只是单个像素的 3D 坐标是否接近教师，还关心这些 3D 点在目标视角中能否形成连续、可复用的覆盖区域。

### Motion-conditioned warp confidence

M1-v10 在冻结基础几何后训练 motion-conditioned confidence head。给定目标运动 $a$，该 head 输出 $C_a$，预测每个 source pixel 是否 projected-valid。监督标签来自教师几何在同一运动下得到的 projected-valid mask $V^T_a$，损失为

$$
\mathcal{L}_{conf}=\mathrm{BCE}(C_a, V^T_a).
$$

该设计把 confidence 从静态几何质量判断改为运动相关判断。同一个像素在小幅平移下可能可以复用，在大 yaw 或前向运动下则可能落到遮挡边界或离开视野。下游使用 $C_a$ 时不需要重新解释几何误差，只需要根据目标运动读取当前可复用区域。

### Hard-case coverage repair

M1-v10 的平均 coverage gap 已经可用，但最差样本仍有明显 target-view holes。代码中已接入两个 hard-case repair 训练项，用来把问题放回训练目标，而不是依赖后处理兜底。

第一项是 coverage-deficit loss。它只惩罚学生覆盖率低于教师覆盖率的区域：

$$
\mathcal{L}_{deficit}=\mathrm{ReLU}(O^T_a-O^S_a).
$$

这一项针对的是下游最敏感的错误，即学生没有覆盖教师可覆盖的目标视角区域。第二项是 depth-edge-aware point loss。它根据教师深度不连续程度提高边界像素权重：

$$
\mathcal{L}_{edge-point}=\frac{1}{|M^T|}\sum_{u \in M^T} w(D^T_u)\rho(P^S_u-P^T_u).
$$

其中 $w(D^T_u)$ 由教师深度边缘计算得到。这个权重让模型更重视遮挡边界、薄结构和深度突变区域，因为这些区域的小误差最容易在 target view 中变成 holes 或错误遮挡。该设计当前还需要 v11 训练和消融验证；如果实验收益不足，正文应把它写成负结果或补充实验，而不是主贡献。

### 总体训练目标

第一阶段的基础几何训练目标为

$$
\mathcal{L}_{M1}=\lambda_p\mathcal{L}_{point}+\lambda_m\mathcal{L}_{mask}+\lambda_n\mathcal{L}_{normal}+\lambda_g\mathcal{L}_{grad}+\lambda_{proj}\mathcal{L}_{proj}+\lambda_{occ}\mathcal{L}_{occ}+\lambda_d\mathcal{L}_{deficit}+\lambda_e\mathcal{L}_{edge-point}.
$$

第二阶段固定基础几何，只优化 $\mathcal{L}_{conf}$。正式论文中，$\mathcal{L}_{deficit}$ 和 $\mathcal{L}_{edge-point}$ 应作为 hard-case repair 变体报告，并通过 ablation 说明它们对 coverage gap、point/projection loss 和 latency 的影响。

### 下游接口

M1 输出的是几何控制信号，不是最终视频帧。下游 pipeline 可以先用 $P$ 做相机变换和 forward splatting，再用 $C_a$ 接受高置信区域；低置信区域、disocclusion、遮挡边界和疑似动态区域交给 DiT/refiner。质量指标应覆盖 target-frame PSNR/LPIPS、warp-better-than-copy fraction、holes 和 false accept。计算指标应覆盖 active token ratio、geometry latency、refiner latency 和端到端 latency。

## Related Work 草稿

### 单目几何估计与几何教师

单目深度和 3D 几何估计把单帧 RGB 转换为可用于视角变换的几何表示。近期的大模型式单目几何方法可以在开放场景中输出密集深度、法向或 point map。[CITATION: MoGe-3] 这类模型适合作为离线教师，但在线推理成本较高；它们的训练目标通常也更关注 source-view 几何一致性，而不是 target-view 重投影可复用性。M1 使用 MoGe-3 提供伪标签，但把学生模型的目标改为 projection、occupancy 和 warp confidence，从而服务于下游计算复用。

### 几何引导的视频生成与世界模型

视频扩散模型和世界模型需要同时处理时序一致性、运动和新视角内容。Matrix 系列、WorldWarp 和 MiniWorld 等工作说明，几何、运动建模和缓存机制可以减少重复计算，并改善交互式生成的延迟。[CITATION: Matrix-1.0/2.0/3.0] [CITATION: WorldWarp] [CITATION: MiniWorld] 这些方法的关键接口不是单帧几何本身，而是几何如何进入生成路径。M1 位于这个接口处：它不替代生成模型，而是提供 dense point map 与 motion-conditioned reuse confidence，让生成模型把算力集中到几何不可靠或目标视角新出现的区域。

### 重投影、遮挡和可复用区域选择

重投影把 source view 内容按照几何关系映射到 target view。在相机运动主导的场景中，它比重新生成整帧更便宜；代价是对深度误差、遮挡边界和 disocclusion 很敏感。固定阈值、hole filling 和后处理 mask 可以缓解部分问题，但也容易改变指标含义。M1 把 projected-valid 和 target-view occupancy 写入训练目标，让模型直接学习给定运动下的可复用区域。motion-conditioned confidence 则把这个判断从静态几何属性变为目标运动相关的路由信号。

### 与本文方法的区别

现有单目几何模型主要回答“图像中的 3D 结构是什么”。几何引导生成方法主要回答“如何把几何接入生成系统”。M1 关注二者之间的接口：给定目标运动后，哪些 3D 信息值得重投影复用，哪些区域应该交给更重的生成模型处理。这个接口决定速度收益，也决定错误重投影是否会污染后续生成。

## Experiments 草稿

### 实验问题

实验部分应围绕四个问题组织。

1. M1 是否比在线 MoGe-3 更快，同时保持可用的 point map 和 projection quality？
2. Motion-conditioned confidence 是否能预测 projected-valid 区域？
3. Hard-case coverage repair 是否降低 target-view holes，尤其是 worst-case coverage gap？
4. 在 gated reprojection 设置中，M1 是否带来正的速度收益，并且不引入不可接受的质量退化？

### 当前可写入正文的结果

当前基础几何模型为 M1-v7，checkpoint 为 `runs/reprojection_student/student_video384_tvod_v7_2k_occ075_width64_lr15/checkpoints/best.pt`。记录结果中，M1-v7 的 point loss 为 0.00528，projection loss 为 0.01042，geometry latency 为 7.594 ms。MoGe-3 teacher 的对应 latency 为 25.263 ms，因此 M1-v7 的几何推理约为 3.33x 加速。

当前置信度模型为 M1-v10，checkpoint 为 `runs/reprojection_student/student_video384_tvod_v10_frozengeom_motionconf_width64_lr1e4/checkpoints/best.pt`。该版本冻结 M1-v7 几何，只训练 motion-conditioned confidence head。记录结果中，v10 confidence 的 AUC_global 为 0.9525，ECE 为 0.0221。在 threshold 0.8 下，kept ratio 为 0.7489，projected-valid rate 为 0.9597。这个结果说明，置信度可以筛出约 75% 的候选复用区域，其中约 96% 在教师定义下 projected-valid。

高运动 closed-loop proxy 给出了初步计算收益。safe tile fraction 为 0.4428 到 0.7757，estimated active ratio 为 0.2243 到 0.5572。结合 active-DiT microbenchmark，nearest active-DiT speedup 为 1.96x 到 3.69x。warp better than copy fraction 为 0.3365 到 0.5136。这个结果只能说明几何重投影在较大运动场景中有正收益空间，不能替代完整 gated-DiT 端到端实验。

### 当前必须保留的负结果

Hard-case 评估包含 120 个 case。M1-v10 的 mean coverage gap 为 -0.0577，worst coverage gap 为 -0.2001。最差场景是 `game2_mid_left`，student coverage mean 为 0.7191，teacher coverage mean 为 0.8296，mean coverage gap 为 -0.1104。这说明当前模型在平均意义上可用，但在遮挡边界、薄结构和较大视角变化下仍会产生 target-view holes。

跨域验证仍不足。当前 evidence pack 只有 1 个域、5 个样本；论文级泛化结论至少需要 3 个域，并且每域不少于 30 个样本。现有跨域结果只能作为 sanity check。后续应补齐 Matrix-Game-2、Matrix-Game-3、MoGe example images，以及更多真实室内、街景和游戏域数据。

## Discussion 草稿

M1 的价值在于把几何估计变成下游生成系统可以直接使用的路由信号。Point map 支持重投影，但它本身不告诉系统该相信哪些投影结果。Motion-conditioned confidence 给出了这个判断，使下游可以按目标运动选择复用区域，并把不可靠区域留给 DiT/refiner。速度收益来自减少重模型处理的空间范围，而不是削弱重模型本身。

当前结果支持两个结论。第一，M1-v7/v10 相比在线 MoGe-3 有明确延迟优势，并能输出重投影所需的几何与置信度。第二，高运动 proxy 显示 active token ratio 可以下降，计算收益为正。三个更强的结论还不能写死：跨域泛化尚未充分验证，hard-case coverage repair 尚未完成训练，完整 Matrix-Game 或 DiT pipeline 的端到端加速尚未给出。

M1 更适合 camera motion 主导、场景几何相对稳定、相邻视角变化可由重投影解释的交互式生成任务。典型场景包括游戏环境、驾驶视角、室内和建筑类视频。风险场景包括非刚体运动较多的画面、透明或反射材料、低纹理区域、极端新视角和教师模型失效的 out-of-domain 图像。在这些场景中，错误接受高置信重投影比保守拒绝更危险，因此后续实验应重点报告 false accept、coverage gap 和 gated-DiT 的质量退化。

## 图表与证据规划

当前不生成新图。后续论文图可以按下表准备。

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

1. Hard-case repair：跑完 v11 conservative/mid/aggressive，报告 mean coverage gap、worst coverage gap、point/projection loss 和 latency。若 coverage 改善但 point/projection 退化，需要给出 speed-quality tradeoff。
2. 跨域验证：至少 3 个域，每域不少于 30 个样本。每个域报告 point loss、projection loss、coverage gap、confidence AUC/ECE 和 latency。
3. 下游闭环：实现 gated reprojection + DiT/refiner 的近端到端实验，报告 quality、active token ratio、latency 和 failure cases。
4. 消融实验：分别移除 projection loss、occupancy loss、motion-conditioned confidence、coverage-deficit loss 和 depth-edge-aware point loss，确认每个模块的作用。
5. 失败案例：保留透明/反射、动态物体、低纹理和极端视角等 case，不用后处理兜底隐藏问题。

## 可放入论文的精简贡献版本

本文提出 M1，一个面向交互式世界模型的重投影友好 3D 几何模块。M1 使用 MoGe-3 离线生成密集几何监督，并训练轻量学生网络输出 point map、valid mask、normal 和 motion-conditioned warp confidence。训练目标同时约束 source-view geometry、target-view projection、target-view occupancy 和 projected-valid confidence，使几何表示直接服务于下游重投影复用。当前记录中，M1-v7 相比 MoGe-3 teacher 达到约 3.33x 几何推理加速；M1-v10 的 confidence AUC_global 为 0.9525，ECE 为 0.0221；高运动 closed-loop proxy 中，active-DiT speedup 估计为 1.96x 到 3.69x。后续实验需要补齐 hard-case coverage repair、跨域泛化和完整下游闭环收益。

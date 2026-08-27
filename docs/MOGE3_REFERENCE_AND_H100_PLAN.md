# MoGe-3 参照选择与 H100 实验报告

日期：2026-08-28

## 一句话结论

建议论文叙事以 **MoGe-3 作为几何参照和高质量教师**，而不是继续只围绕 MoGe-2 展开；但系统实现上仍保留 **MoGe-2 Small / Copy / Homography** 作为必须击败的实时强基线。

换句话说：MoGe-3 更适合做“我们应该追求什么几何质量、什么区域值得花几何预算”的参照，不应在没有 H100 端到端实验前被写成“已经更快的实时模块”。真正要证明的是：MoGe-3 带来的更好重投影，是否能让 AR-DiT 少跑足够多的步数或 token，从而覆盖新增几何成本。

## 为什么参照对象应切到 MoGe-3

MoGe-3 的机制更贴近我们的论文问题。它不是简单给一张深度图，而是在单帧 RGB  geometry 前端之后加入 self-guided sparse volumetric refinement。这个机制天然提供一个几何预算轴：`K_g=0/1/3`。这和我们希望做的 `Reuse / Repair / Regenerate` 或 `0/1/2/4 DiT steps` 路由非常搭，可以形成统一叙事：

**不是所有帧都需要同样精细的几何，也不是所有区域都需要完整 DiT 生成。**

官方论文中，MoGe-3 ViT-L 的局部点准确率从 MoGe-2 ViT-L 的 46.6% 提升到 MoGe-3 `K_g=0` 的 50.3%，再到 `K_g=3` 的 55.9%。这说明 MoGe-3 的确能提供更强的局部几何和细节边界信号。

但也必须保留反面事实：同一表中 metric point Rel 从 MoGe-2 ViT-L 的 8.65 变为 MoGe-3 `K_g=0` 的 8.88、`K_g=3` 的 9.61；也就是说，局部更细不等于全局投影一定更准。论文运行时表也显示，在 A100 FP16 batch=1、700² 条件下，MoGe-2 ViT-L 为 39 ms，MoGe-3 ViT-L 三步为 121 ms。MoGe-3 不是天然轻量模型。

因此本文建议的定位是：

| 模块 | 在我们论文中的角色 |
| --- | --- |
| Copy | 必须击败的短时序强基线 |
| Rotation Homography | 纯鼠标视角旋转的低成本基线 |
| MoGe-2 Small | 当前轻量部署基线 |
| MoGe-2 Large | 隔离“骨干变大”带来的收益 |
| MoGe-3 Large `K_g=0/1/3` | 主参照、质量上界、预算轴、离线教师 |
| 轻量蒸馏几何头 | 如果 MoGe-3 太慢，最终可能部署的模块 |

## 我们论文的核心问题

我们的目标不是做一个新的单目深度模型，也不是复现 WorldWarp 的 3DGS 世界。我们的目标是：

**在 Matrix-Game 类交互式视频世界模型中，利用几何重投影和玩家行为预测，减少 AR-DiT 的真实计算，同时保持画质、交互响应和长期自回归一致性。**

可写成论文问题：

给定历史生成帧 `I_t`、历史几何 `G_t`、玩家动作 `a_{t:t+k}`、AR-DiT 历史状态 `H_t`，系统需要生成未来帧 `I_{t+k}`。完整 AR-DiT 生成质量高但慢；几何重投影快但会在遮挡、动态物体、反射、近景边界和错误位姿下失败。我们要学习一个控制器，在每个时间或区域上选择：

- `Reuse`：直接 Copy 或旋转单应；
- `Warp`：使用几何深度/点图重投影；
- `Repair`：少步 DiT 或轻量修复；
- `Regenerate`：完整 DiT 生成可信关键帧。

论文的核心不是“MoGe-3 比 MoGe-2 强”，而是：

**MoGe-3 作为高质量几何参照，可以帮助我们建立动作感知的几何预算和生成预算调度，从而在质量约束下减少 AR-DiT 计算。**

## 技术骨架

推荐论文方法名暂定为：

**Action-Guided Geometry Reuse for Real-Time AR-DiT Game Generation**

系统由五个部分组成。

### 1. 可信神经关键帧与几何锚点

每隔 `R` 帧，或当风险过高时，运行完整 AR-DiT 得到可信关键帧。对可信关键帧运行 MoGe-3，得到：

- RGB/HDR；
- point map 或 depth；
- camera intrinsics `K`；
- valid mask；
- normal；
- geometry uncertainty features；
- anchor timestamp；
- AR-DiT committed KV/state id。

这里的 3D 信息不是在线建模成完整世界，而是 2.5D 可见表面缓存。第一篇论文建议不要把目标设成“从生成视频实时建 3DGS”，那会把问题放大太多。我们只需要能把最近可信帧投到目标视角的几何表征。

### 2. 玩家动作到目标视角

玩家输入不能直接等价于真实相机矩阵。建议分三档处理：

| 设定 | 做法 | 难度 |
| --- | --- | --- |
| 有引擎或模拟器状态 | 直接读相机位姿、FOV、角色速度、碰撞 | 最适合作为上界 |
| 有训练轨迹但部署 RGB-only | 学动作到相机增量 `a -> ΔT, ΔK` 的小模型 | 推荐主线 |
| 完全无状态 | 从历史 RGB、光流、动作估计相机运动 | 难，作为扩展 |

重投影公式为：

```text
p' = π( K_target · ( R · ( D · K_source^-1 p ) + t ) )
```

其中纯光心旋转时深度会消掉：

```text
p' ~ K_target · R · K_source^-1 · p
```

这给我们一个重要策略：鼠标视角小旋转优先使用 Homography 或 Copy，不要每次都花 MoGe-3；WASD 平移、近景遮挡和第三人称绕角色运动才更需要深度。

### 3. MoGe-3 参照几何与自适应几何预算

MoGe-3 提供 `K_g=0/1/3` 的几何预算轴。我们可以把它转成研究问题：

- 静止或纯旋转：`K_g=0` 或不跑几何；
- 小平移、远景：轻量几何或 MoGe-3 `K_g=0`；
- 近景侧移、遮挡边界、薄结构：MoGe-3 `K_g=1/3`；
- 高动态、反射、水面、粒子、交互事件：不要盲目 Warp，直接 Repair 或 Regenerate。

如果 H100 实验显示 MoGe-3 三步太贵，但确实能显著提高可复用区域，就把 MoGe-3 作为离线教师，蒸馏一个轻量几何头。蒸馏损失不应只用深度误差，还要加入重投影相关目标：

- 给定动作扰动后的投影坐标误差；
- 前后景排序；
- 边界带一致性；
- 空洞和冲突区域预测；
- 多帧尺度稳定性；
- 对 Repair/Regenerate 路由的下游收益。

### 4. 重投影质量感知与路由器

运行时不能看到真实目标帧，所以路由器不能使用 target RGB 或 target depth。可用特征应该来自历史和预测：

- motion magnitude：动作幅度、预测平移、预测旋转；
- depth risk：深度梯度、近景比例、MoGe valid mask、`K_g=0` 与 `K_g=1/3` 差异；
- visibility risk：forward/backward support、空洞率、深度冲突、disocclusion；
- semantic risk：角色、HUD、水面、镜面、透明、粒子；
- temporal risk：anchor age、连续复用次数、多锚点不一致；
- model risk：Repair 后是否仍违反边界或运动一致性。

路由目标不是“画面看起来还行”，而是控制错误复用：

```text
FalseAccept = 被系统接受复用但实际不达标的区域 / 被系统接受复用的区域
SafeReuse = 被系统接受且实际达标的区域 / 可评价区域
```

初始阈值可以扫 1% 和 5% FalseAccept 工作点。正式论文中必须在独立校准集上选阈值，再在测试集上报告结果。

### 5. Provisional Display 与 Committed Memory 分离

这是 AR-DiT 游戏生成里非常关键的一点。Warp 或 Repair 的帧可以先显示，以降低玩家感知延迟；但不能随意写入长期 AR-DiT KV 或世界记忆。否则一次错误投影会污染后续所有自回归状态。

推荐双轨机制：

- **Provisional path**：低延迟显示，允许 Copy/Warp/Repair；
- **Committed path**：只由完整 AR-DiT 或高置信结果更新 KV/geometry anchor；
- **Rollback/refresh**：风险升高时丢弃 provisional 状态，刷新可信关键帧。

这个设计可以成为论文方法亮点：我们不是单纯跳步，而是在交互系统里区分“先给玩家看到”和“写入长期世界状态”。

## H100 实验路线

### P0：先验证几何上界，不急着接 DiT

目标：判断问题到底在深度、位姿、投影器，还是动态渲染变化。

用有真值的游戏/仿真序列做四组：

| 深度 | 位姿 | 目的 |
| --- | --- | --- |
| GT depth | GT pose | 几何重投影上界 |
| Pred depth | GT pose | 单独测深度模型 |
| GT depth | Pred pose | 单独测动作到位姿 |
| Pred depth | Pred pose | 真实部署近似 |

同时比较：

- Copy；
- Rotation Homography；
- MoGe-2 Small；
- MoGe-2 Large；
- MoGe-3 Large `K_g=0/1/3`。

如果 GT depth + GT pose 都打不过 Copy，说明当前场景中重投影不适合作为主加速；如果 GT pose 有效但 Pred pose 无效，先做动作到位姿；如果 GT depth 有效但 MoGe 输出无效，才是 MoGe-3/蒸馏几何头的问题。

### P1：MoGe-3 与 MoGe-2 的同硬件 A/B

固定同一批输入、裁剪、`num_tokens`、精度、投影器、mask 规则和质量阈值。建议矩阵：

```text
Copy
Homography
MoGe-2 Small Normal
MoGe-2 Large Normal
MoGe-3 Large K_g=0
MoGe-3 Large K_g=1
MoGe-3 Large K_g=3
```

必须记录：

- model load time；
- first call latency；
- warmup 后 p50/p95/p99；
- peak allocated/reserved memory；
- `geometry.npz`；
- projected RGB；
- hole/conflict mask；
- common support metrics；
- full-frame metrics；
- per-scene paired difference。

H100 单图几何脚本入口已放入仓库：

```bash
python tools/benchmark_geometry_versions.py \
  --repo /path/to/MoGe \
  --version v3 \
  --checkpoint Ruicheng/moge-3-vitl \
  --revision a96e58bad16a94c9a3c193a5d4cd75b4b6906c94 \
  --image /path/to/input.png \
  --output /path/to/results/moge3_l_kg0_1_3 \
  --steps 0,1,3 \
  --num-tokens 1200 \
  --max-size 640 \
  --warmup 10 \
  --repeat 50 \
  --precision fp16
```

### P2：接入 AR-DiT 路由，验证真实加速

这一阶段才回答“速度收益是否为正”。

主对照：

- Full AR-DiT every frame；
- Periodic keyframe + Copy；
- Periodic keyframe + Homography；
- MoGe-2 geometry route；
- MoGe-3 geometry route；
- MoGe-3 teacher distilled lightweight route。

每条路线都要在相同质量工作点下报告：

- 实际 AR-DiT 调用次数；
- 实际 DiT steps；
- active token ratio；
- VAE encode/decode cost；
- router cost；
- geometry cost；
- warp/composite cost；
- action-to-first-visible latency；
- action-to-committed-neural latency；
- p50/p95/p99；
- FalseAccept；
- SafeReuse；
- 长 rollout 漂移。

正收益判据：

```text
G_moge3 - G_baseline < 下游 DiT/VAE/KV/Repair 实际节省
```

如果 MoGe-3 只让图更准，但没有让 DiT 少跑，不能写成加速成功。可以改写为“高质量教师提升轻量几何路由”。

### P3：形成论文图表

建议最终论文至少有这些图：

1. 系统总览：Action -> Pose/Risk -> Geometry Budget -> Reproject/Repair/Regenerate -> Provisional/Committed。
2. MoGe-3 `K_g=0/1/3` 的质量-时延 Pareto。
3. Copy/Homography/MoGe-2/MoGe-3 在不同动作分层下的 SafeReuse。
4. 端到端 p50/p95 延迟柱状图。
5. FalseAccept vs SafeReuse 曲线。
6. 成功/失败案例：近景平移、纯旋转、水面、动态角色、遮挡边界。

## 论文技术骨架

### 标题候选

**Action-Guided Geometry Reuse for Low-Latency Autoregressive Game Generation**

或更强调 MoGe-3：

**MoGe-Guided Reprojection and Repair for Real-Time AR-DiT Game World Models**

### 摘要逻辑

交互式视频世界模型可以生成开放游戏画面，但完整 AR-DiT 推理成本高，难以低延迟响应玩家输入。我们提出一种动作引导的几何复用框架，用 MoGe-3 作为高质量单帧几何参照，从可信关键帧构建 2.5D 几何锚点；根据玩家动作、可见性风险和几何不确定性，在 Copy、Homography、3D Warp、少步 Repair 和完整 Regenerate 间自适应选择。系统将低延迟显示帧与长期 committed AR memory 分离，避免近似结果污染自回归状态。在 H100 实验中，我们将在相同质量约束下比较 MoGe-2、MoGe-3 和蒸馏几何头，报告端到端延迟、错误复用率和长期一致性。

### 主要贡献

1. 提出面向 AR-DiT 游戏生成的动作感知几何预算调度，而不是固定每帧生成或固定每帧重投影。
2. 将 MoGe-3 作为高质量几何参照，系统评估其在重投影复用中的实际下游价值。
3. 设计 `Reuse / Warp / Repair / Regenerate` 多级路由，并以 FalseAccept 和 SafeReuse 衡量质量约束下的可跳过计算。
4. 提出 provisional display 与 committed AR memory 分离，降低交互延迟同时控制长期漂移。
5. 给出同硬件、同输入、同质量阈值的 H100 实验协议，避免把几何榜单提升误写成系统加速。

### 方法章节

1. Problem formulation：交互式 AR-DiT 游戏生成、动作输入、延迟目标。
2. Geometry anchor：可信关键帧和 MoGe-3 2.5D 几何缓存。
3. Action-conditioned motion：玩家输入到相机/场景运动估计。
4. Geometry reprojection：Copy、Homography、Depth Warp 和空洞/冲突处理。
5. Risk-aware routing：质量感知路由器和 DiT step/token 节省。
6. Provisional and committed memory：显示路径和长期记忆分离。
7. Training/calibration：用离线目标帧生成路由标签，部署时只用历史和动作。

### 实验章节

1. Geometry upper bound with GT depth/pose。
2. MoGe-2 vs MoGe-3 geometry A/B。
3. Action-segmented reprojection quality。
4. End-to-end AR-DiT acceleration。
5. Ablations：`K_g`、anchor interval、route threshold、distillation、memory commit。
6. Failure analysis：dynamic object、water/reflection、large disocclusion、teleport/cut。

## 风险与止损条件

必须提前承认这些风险：

- MoGe-3 可能质量更好但太慢，只适合作教师；
- Copy 在短间隔可能很强，必须作为主基线；
- 纯鼠标旋转不需要深度，MoGe-3 主要价值在平移和遮挡；
- 单目深度尺度漂移会影响跨帧投影；
- 水面、镜面、粒子和角色动作不是静态几何能解决的；
- 如果 active token 稀疏化没有真实 wall-clock 收益，理论 FLOPs 节省没有意义；
- 若近似帧写入 AR-KV，可能造成长期漂移。

止损条件：

- GT 几何上界也无法优于 Copy；
- MoGe-3 比 MoGe-2 的重投影 SafeReuse 提升不足以覆盖几何成本；
- FalseAccept 在可接受阈值下过高；
- p95/p99 交互延迟恶化；
- H100 上真实 AR-DiT 路由没有端到端收益。

## 最终建议

本项目应从“用 MoGe-2 做一个可行的几何接口”升级为“以 MoGe-3 为参照，研究几何预算如何服务 AR-DiT 加速”。论文不要写成 MoGe-3 应用文，而要写成系统论文：

**玩家动作决定未来视角，几何预测提供可复用先验，路由器决定哪些区域可以复用、哪些区域少步修复、哪些必须完整生成。**

MoGe-3 在这里最有价值的身份是：

1. H100 上的高质量几何参照；
2. `K_g=0/1/3` 几何预算消融轴；
3. 轻量几何头的离线教师；
4. 证明近景平移、遮挡边界等困难场景是否真的需要更强几何的工具。

如果 H100 结果显示 MoGe-3 三步路线在质量约束下带来正收益，就把 MoGe-3 写为关键模块；如果它质量好但太慢，就把它写为教师和上界，最终部署蒸馏几何头。两种结果都能支撑论文，只是主贡献表述不同。

## 主要来源

- MoGe-3 paper: https://arxiv.org/html/2607.17967v2
- MoGe official repository: https://github.com/microsoft/MoGe
- MoGe-3 ViT-L weights: https://huggingface.co/Ruicheng/moge-3-vitl
- 本仓库既有审计结果：`results/audited/summary.json`

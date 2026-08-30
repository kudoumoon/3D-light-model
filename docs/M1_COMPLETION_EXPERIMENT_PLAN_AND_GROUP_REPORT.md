# M1 补齐实验计划与组会汇报草稿

## 结论先行

当前 M1 的可交付核心是：以 MoGe-3 teacher 为几何监督源，训练一个轻量 reprojection-friendly geometry student，输出 point map / depth / mask / normal / warp confidence，用于下游把可复用区域重投影，把不可靠区域交给更重的 DiT/refiner。

目前最强版本不是单纯“MoGe-3 蒸馏”，而是两段式方案：

1. M1-v7：主几何网络，学习 point map，并用 projection/occupancy/warp-valid 目标让几何更适合 forward reprojection。
2. M1-v10：冻结 v7 几何，只训练 motion-conditioned confidence head。这样保持几何准确性，同时让下游能按目标运动判断哪些像素/tiles 可安全复用。

当前 blocker 是三类：跨域验证规模不足、hard-case coverage gap 最差约 -0.200、下游闭环收益还需要从 proxy 走向正式 end-to-end。

## 当前已有关键结果

### 主模型结果

- Base geometry checkpoint: `runs/reprojection_student/student_video384_tvod_v7_2k_occ075_width64_lr15/checkpoints/best.pt`
- Motion confidence checkpoint: `runs/reprojection_student/student_video384_tvod_v10_frozengeom_motionconf_width64_lr1e4/checkpoints/best.pt`
- Teacher MoGe-3 latency: 25.263 ms
- v7 geometry latency: 7.594 ms，约 3.33x speedup
- v10 motion confidence retest latency: 10.558 ms，约 2.39x speedup
- v10 confidence: AUC_global 0.9525，ECE 0.0221
- v10 threshold 0.8: kept_ratio 0.7489，projected-valid rate 0.9597

### Hard-case 现状

- v10 mean coverage gap: -0.0577
- v10 worst coverage gap: -0.2001
- 最差场景集中在 `game2_mid_left`，该场景 mean coverage gap: -0.1104
- 这说明模型能稳定提供几何信息，但在遮挡边界、快速视角变化、薄结构区域仍会漏掉 target-view coverage。

### 下游闭环已有信号

真实 target-frame 评估显示：

- 低运动片段：copy baseline 很强，warp 不一定提升 PSNR；此时 M1 的价值是 confidence gating，避免错误 warp。
- 高运动片段：warp_better_than_copy_fraction 明显升高，部分长运动/转向片段中超过 0.4–0.75，说明几何重投影在运动变大时有实际收益。
- active DiT proxy 显示，当安全重投影 tile 能减少 active tokens 时，大模型路径可获得正收益；已有 large benchmark 中 active_ratio 0.50/0.25 对应约 1.96x/3.69x speedup。

## 本轮新增创新点：Hard-case Coverage Repair

本轮不增加大量工程兜底，不靠 `if/else` 改结果，而是在训练目标上补两类可解释约束：

1. Coverage-deficit loss：只惩罚 student target-view occupancy 小于 teacher occupancy 的区域，直接修复重投影空洞。
2. Depth-edge-aware point loss：对 teacher depth discontinuity 高的区域加权，因为遮挡边界/薄结构的小 3D 误差最容易导致投影漏洞。

这两个目标都服务于“重投影友好型几何”，不是单纯降低 depth MSE。它们与我们的论文目标一致：几何模块不是追求视觉重建，而是为下游快速可靠复用提供 point map 和 confidence。

## 本轮待跑实验

### E1 Cross-domain validation

目标：证明 M1 不是只在 Matrix-Game validation 内有效。

- 数据：GTA / Temple / Universal，每域至少 30 帧。
- 输出：student point/projection loss、coverage gap、inference latency、confidence AUC/ECE。
- 成功标准：跨域 projection loss 不显著崩溃；coverage gap mean 控制在可用区间；confidence ECE 保持低水平。

### E2 Hard-case repair training

目标：把 worst-case coverage gap 从 -0.200 拉回到更可靠区域。

候选配置：

- conservative: occupancy 0.25, coverage_deficit 0.50, depth_edge 0.10
- mid: occupancy 0.50, coverage_deficit 1.00, depth_edge 0.25
- aggressive: occupancy 0.75, coverage_deficit 1.50, depth_edge 0.35

成功标准：

- coverage_gap_min 明显改善，目标 >= -0.120
- coverage_gap_mean 不劣化
- point/projection loss 不明显上升
- latency 仍保持相对 MoGe-3 的正 speedup

### E3 Downstream closed-loop benefit

目标：证明 M1 几何模块能给下游带来正收益，而不只是“单模块指标好看”。

报告方式：

- 按 low-motion / high-motion 分层；低运动强调 gating，避免错误 warp；高运动强调 warp 替代 copy 的收益。
- 报告 copy、warp、oracle、gated active-DiT proxy。
- 关键指标：safe_tile_fraction、warp_better_than_copy_fraction、estimated active_ratio、DiT speedup。

## 论文图与证据管理

本轮已生成 evidence pack：

- `results/recorded/m1_v11_completion_pack/m1_completion_summary.json`
- `results/recorded/m1_v11_completion_pack/cross_domain_by_scene.csv`
- `results/recorded/m1_v11_completion_pack/hardcase_by_scene.csv`
- `results/recorded/m1_v11_completion_pack/closed_loop_proxy.csv`
- `results/recorded/m1_v11_completion_pack/figures/m1_cross_domain_demo_validation.svg`
- `results/recorded/m1_v11_completion_pack/figures/m1_hardcase_coverage_gap.svg`
- `results/recorded/m1_v11_completion_pack/figures/m1_closed_loop_proxy_speedup.svg`

## Related work 启发

- Matrix-Game official repo 显示 Matrix-Game 1.0/2.0/3.0 持续围绕 open-source interactive world model、real-time streaming 和 long-horizon memory 推进。
- Matrix-Game 2.0 的核心方向是 few-step auto-regressive diffusion，并用大规模 Unreal/GTA 数据、action injection 和 causal distillation 支撑 25 FPS streaming generation。
- Matrix-Game 3.0 进一步把问题推进到 720p real-time long-form generation，引入 camera-aware memory retrieval、error buffer/self-correction 和 few-step distillation；这要求我们的 M1 也必须证明几何信息对下游 streaming/long-horizon 有实际收益。
- WorldWarp 的关键思想是用 online 3D geometric cache/warp 作为 structural scaffold，再用 diffusion 做 fill-and-revise；这直接支持我们把 M1 定位为轻量结构锚点。
- MiniWorld 强调 block-causal Video DiT、rolling KV cache 和 pipelined asynchronous denoising；这说明我们的下游收益评估需要落在 active token / rolling cache / streaming latency，而不是只报单帧 PSNR。

参考来源：

- Matrix-Game official GitHub: https://github.com/SkyworkAI/Matrix-Game
- Matrix-Game 2.0 arXiv: https://arxiv.org/abs/2508.13009
- Matrix-Game 3.0 project page: https://matrix-game-v3.github.io/
- WorldWarp arXiv: https://arxiv.org/abs/2512.19678
- MiniWorld arXiv: https://arxiv.org/abs/2608.01127

## Agent2 literature 审计补充

agent2 的只读审计给出一个核心定位：M1 should turn monocular geometry from passive depth prediction into an action-conditioned compute-allocation signal for real-time world generation. 因此论文中不能把 M1 写成普通 MoGe-3 student，而应强调：

- MoGe-3 是 offline teacher；M1 是 lightweight reprojection-oriented student。
- M1 的主指标不是单纯 depth/point regression，而是 target-view coverage、coverage gap、warp-valid confidence、ECE、false accept / safe reuse。
- 下游收益必须按 motion magnitude 分层：low-motion 下 copy 很强，M1 的价值是 confidence gating；high-motion 下 warp/reuse 才体现质量和速度收益。
- 当前 active-DiT benchmark 只能称为 proxy；正式结论需要 end-to-end 或接近 end-to-end 的 gated-DiT latency/quality 对比。
- 长时序 AR 系统需要避免 memory pollution：建议区分 provisional display 与 committed memory，高风险 warp 不写入长期 KV/world memory。

审稿风险按优先级排序：

1. “只是 MoGe-3 蒸馏”的质疑：用 projection / occupancy / motion-conditioned confidence / coverage-deficit repair 回答。
2. Copy/Homography baseline：必须分低运动和高运动报告。
3. 单目尺度漂移：用 projection error、confidence calibration 和 gating 处理。
4. 加速是否真实：proxy 不能替代 end-to-end，需要继续补 E3。
5. 泛化证据不足：跨域样本量当前仍是硬缺口。

## 当前评分

以 A 会 oral 标准粗评：当前约 84/100。

- 几何模型工程闭环：较强
- 重投影友好创新：已有明确方向，v10 confidence 是亮点
- 泛化证据：不足，需要 E1
- hard-case 鲁棒性：不足，需要 E2
- 下游收益闭环：已有信号但需要 E3 完整化

如果 E1/E2/E3 达到预期，M1 可进入 90 分附近；否则只能作为一个有用但不够强的工程模块。

# M1 组会汇报：Reprojection-Friendly 3D Geometry Module

日期：2026-08-30  
仓库：`kudoumoon/3D-light-model`  
模块定位：M1 / 3D 信息几何模型

## 1. 一句话结论

当前 M1 已经形成可交付的轻量几何模块雏形：以 MoGe-3 作为 offline teacher，训练 reprojection-friendly student，输出 point map / depth / valid mask / normal / warp confidence，为下游重投影和 DiT 局部计算分配提供几何先验。

当前最可靠版本是：

- Base geometry：M1-v7，负责稳定 point map 与 projection-friendly geometry。
- Motion confidence：M1-v10，冻结 v7 几何，只训练 motion-conditioned confidence head。

现阶段结论是：M1 已具备工程可用性和明确创新方向，但还不能声称完全达到 A 会 oral 水平。主要缺口是：跨域验证规模不足、hard-case coverage gap 未修完、下游闭环仍是 proxy 而非完整 end-to-end。

## 2. 我们的 M1 到底做了什么

M1 不是简单把 MoGe-3 重新训练一遍，也不是普通 monocular depth estimator。它的目标是把单目几何从被动 depth prediction 变成 action-conditioned compute-allocation signal。

给定一帧 RGB，M1 输出：

- `point map`：每个像素在 camera-space 的 3D point；
- `depth`：用于 forward splatting 和遮挡关系；
- `valid mask`：标记 source-view 中 geometry 是否可信；
- `normal`：提供局部 surface orientation；
- `warp confidence`：预测给定目标 motion 下哪些像素/tiles 可安全重投影。

下游使用方式：

1. 对高置信区域，直接用 point map 做 camera transform + forward reprojection；
2. 对低置信区域、disocclusion、遮挡边界和动态区域，交给更重的 DiT/refiner；
3. 用 confidence gating 降低 active token ratio，从而获得速度收益。

## 3. 技术链路

### 3.1 Teacher 数据生成

Teacher 采用 MoGe-3，离线导出每帧几何：

- RGB
- point map
- depth
- valid mask
- normal
- intrinsics

这些数据统一保存为 `geometry.npz`，后续训练、评估、重投影脚本都使用同一 contract。

### 3.2 Student 几何蒸馏

Student 是轻量 CNN encoder-decoder。基础训练目标包括：

- point loss：匹配 MoGe-3 point map；
- mask loss：预测 source valid geometry；
- normal loss：保持局部几何方向；
- edge loss：约束 depth gradient，减少边界过平滑。

### 3.3 Reprojection-friendly supervision

只拟合 source-view geometry 不足以保证下游重投影有效。因此训练时采样虚拟相机运动，例如 yaw 和 forward translation，把 teacher/student point map 投影到目标视角，额外加入：

- projection loss：约束 projected coordinates；
- target-view occupancy loss：让 student 的 target-view coverage 接近 teacher；
- projected-valid / warp confidence loss：学习哪些 source pixels 在目标视角仍然有效。

这个设计让 M1 直接优化“能否重投影复用”，而不是只优化几何外观误差。

### 3.4 Motion-conditioned confidence

v9 joint fine-tuning 虽然提升了 confidence，但存在破坏 base geometry coverage 的风险。因此最终采用 v10：

- 冻结 v7 base geometry；
- 只训练 motion-conditioned confidence head；
- 输入目标 yaw/forward motion encoding；
- 输出当前 motion 下的 warp confidence。

这使得几何质量和路由置信度解耦，更适合作为下游稳定接口。

## 4. 当前已有模型与结果

### 4.1 模型 checkpoint

- v7 base geometry checkpoint：`runs/reprojection_student/student_video384_tvod_v7_2k_occ075_width64_lr15/checkpoints/best.pt`
- v10 motion confidence checkpoint：`runs/reprojection_student/student_video384_tvod_v10_frozengeom_motionconf_width64_lr1e4/checkpoints/best.pt`

### 4.2 几何与速度结果

| 指标 | MoGe-3 teacher | M1-v7 / v10 |
|---|---:|---:|
| Geometry latency | 25.263 ms | 7.594 ms |
| Geometry speedup | 1.00x | 3.33x |
| Motion-confidence latency | 25.263 ms teacher baseline | 10.558 ms |
| Confidence speedup | 1.00x | 2.39x |
| v7 point loss | - | 0.00528 |
| v7 projection loss | - | 0.01042 |
| v10 confidence AUC | - | 0.9525 |
| v10 ECE | - | 0.0221 |

v10 threshold 0.8 时：

- kept ratio：0.7489
- projected-valid rate：0.9597

解释：大约 75% 像素可以被 confidence 接受为可复用区域，其中约 96% 实际投影有效。这是 M1 当前最强的下游接口证据。

## 5. 创新点总结

当前 M1 可写成三个创新点。

### 5.1 Reprojection-oriented geometry distillation

不是普通 depth distillation，而是在训练目标中加入 projection loss 和 target-view occupancy，使 student 几何直接服务于 forward reprojection。

### 5.2 Motion-conditioned warp confidence

不是输出单一静态 confidence，而是根据目标 motion 预测当前像素是否可安全重投影。这个 confidence 可直接用于下游 gating / active token selection。

### 5.3 Hard-case coverage repair

本轮已实现但还未完成训练的 v11 方向：

- coverage-deficit loss：只惩罚 `student target occupancy < teacher target occupancy` 的区域，直接针对 target-view holes；
- depth-edge-aware point loss：提高遮挡边界、薄结构、深度突变区域权重。

这两个目标针对当前 worst-case coverage gap，避免靠大量 `if/else` 兜底改结果，而是在训练目标上解决几何可用性问题。

## 6. Hard-case 现状

当前 v10 的 coverage 结果：

| 指标 | 数值 |
|---|---:|
| Mean coverage gap | -0.0577 |
| Worst coverage gap | -0.2001 |
| Yaw10 mean coverage gap | -0.0515 |
| Worst scene | game2_mid_left |
| game2_mid_left student coverage | 0.7191 |
| game2_mid_left teacher coverage | 0.8296 |
| game2_mid_left mean gap | -0.1104 |

解释：整体平均可用，但最差样本仍有接近 20% 的 target-view coverage 缺口。这个问题会影响下游重投影质量，尤其是遮挡边界、近景薄结构和较大视角变化。

当前已完成：

- v11 hard-case repair loss 已接入训练脚本；
- 三组候选配置已写入 GPU 队列；
- 因 GPU 被其他任务占用，本轮没有完成 v11 训练。

## 7. 跨域泛化现状

当前 evidence pack 中跨域验证仍不足：

- 当前覆盖域数：1
- 当前样本数：5
- 论文级目标：每域至少 30 个样本，总计至少 90 个跨域样本

本地可用数据包括：

- Matrix-Game-2 demo images；
- Matrix-Game-3 demo images；
- MoGe example images。

已完成准备工作：

- `evaluate_reprojection_student_warp.py` 已支持 `--split all`；
- GPU 队列已接入跨域 teacher export + student eval；
- 等空卡后会自动跑 `matrixgame2_demo`、`matrixgame3_demo`、`moge_examples` 的本地跨域验证。

限制：本地数据规模仍偏小。如果要在论文中强声称泛化性，建议额外引入更多真实视频/室内/街景/游戏域数据，每域 30–100+ 帧。

## 8. 下游闭环收益现状

当前已有 real target-frame 评估和 active-DiT microbenchmark。结论必须分 motion magnitude 报告。

### 8.1 低运动场景

低运动时 copy baseline 很强，warp 不一定提升 PSNR。M1 在这里的价值不是强行替代 copy，而是用 confidence gating 判断哪些区域可以复用、哪些区域不该动。

### 8.2 高运动场景

高运动时，warp 相对 copy 的优势明显上升。当前闭环 proxy 中，高运动片段的关键结果如下：

| Eval split | 高运动样本数 | safe tile @20px | estimated active ratio | nearest active-DiT speedup | warp better than copy |
|---|---:|---:|---:|---:|---:|
| gta_target_eval | 1 | 0.6211 | 0.3789 | 1.96x | 0.3365 |
| gta_target_eval_long | 2 | 0.5901 | 0.4099 | 1.96x | 0.4041 |
| gta_turn_target_eval | 6 | 0.7757 | 0.2243 | 3.69x | 0.4621 |
| gta_turn_target_eval_long | 2 | 0.4428 | 0.5572 | 1.96x | 0.5136 |

解释：在高运动/转向片段中，warp 更常优于 copy；如果把安全 tile 直接复用，只让不安全区域进入 DiT，则 active token ratio 可降到约 0.22–0.56，对应 active-DiT proxy 约 1.96x–3.69x。

注意：这仍是 proxy，不应写成完整 Matrix-Game end-to-end speedup。正式论文需要补一个 gated-DiT 或近似 end-to-end 的闭环实验。

## 9. 泛化性与适用场景

M1 适合以下场景：

- camera motion 主导的交互式世界模型；
- 单目 RGB 中有稳定几何结构的游戏、室内、街景、建筑、驾驶场景；
- 短时序 streaming 生成中，相邻视角变化可由几何重投影解释的大部分区域；
- 需要 geometry cache / warp cache 降低 DiT 计算量的长视频系统；
- 下游可以接受 confidence gating，即不确定区域交给生成模型处理。

M1 风险场景：

- 大量非刚体动态物体；
- 透明、反射、低纹理区域；
- 极端新视角造成大面积 disocclusion；
- teacher MoGe-3 自身几何不可靠的 out-of-domain 图像；
- 长时序 AR 中，如果错误 warp 写入长期 memory，可能污染后续生成。

建议系统策略：区分 provisional display 和 committed memory。高置信重投影可以用于低延迟显示，但只有高置信或 DiT/refiner 验证后的结果才写入长期 AR/KV/world memory。

## 10. 当前 A 会 oral 标准评估

当前评分：约 84/100。

| 维度 | 当前状态 | 风险 |
|---|---|---|
| 技术链路 | 已完整 | 风险低 |
| 模型速度 | v7 约 3.33x 快于 teacher | 风险低 |
| Confidence | AUC 0.9525 / ECE 0.0221 | 风险低 |
| 重投影友好性 | 平均可用，但 worst gap -0.2001 | 风险中高 |
| 跨域泛化 | 当前只有 5 样本 evidence | 风险高 |
| 下游收益 | proxy 为正，未 end-to-end | 风险高 |
| 创新叙事 | 已从 MoGe distill 转为 geometry routing | 风险中 |

达到 90+ 的必要条件：

1. 跑完 v11 hard-case repair，并证明 worst coverage gap 明显改善；
2. 补齐至少 3 个域、每域 30+ 样本的跨域验证；
3. 补齐 gated-DiT 或接近 end-to-end 的 closed-loop speed/quality 实验；
4. 保留 failure cases，不用工程兜底掩盖失败；
5. 给出可复现 evidence pack，包括 JSON/CSV/图和脚本。

## 11. 已提交/已管理的证据

当前已生成并提交的主要文件：

- `docs/M1_METHOD_DRAFT_ZH.md`
- `docs/M1_COMPLETION_EXPERIMENT_PLAN_AND_GROUP_REPORT.md`
- `tools/summarize_m1_completion_experiments.py`
- `tools/run_m1_completion_gpu_queue.sh`
- `results/recorded/m1_v11_completion_pack/m1_completion_summary.json`
- `results/recorded/m1_v11_completion_pack/cross_domain_by_scene.csv`
- `results/recorded/m1_v11_completion_pack/hardcase_by_scene.csv`
- `results/recorded/m1_v11_completion_pack/closed_loop_proxy.csv`
- `results/recorded/m1_v11_completion_pack/figures/m1_cross_domain_demo_validation.svg`
- `results/recorded/m1_v11_completion_pack/figures/m1_hardcase_coverage_gap.svg`
- `results/recorded/m1_v11_completion_pack/figures/m1_closed_loop_proxy_speedup.svg`

## 12. 下一步实验计划

一旦检测到稳定空卡，按以下顺序执行：

1. v11 hard-case repair conservative/mid/aggressive 三组训练；
2. 对每个候选跑 val eval 和 multi-motion reprojection eval；
3. 自动选择 worst-case coverage gap 改善最大的候选；
4. 对选中模型跑 Matrix-Game-2 / Matrix-Game-3 / MoGe examples 跨域验证；
5. 刷新 evidence pack；
6. push 第二次 GitHub 更新。

如果 v11 未改善 worst-case coverage gap，需要继续迭代：

- 增加 hard motion sampling；
- 增加 confidence false-accept penalty；
- 做 scene/action held-out hard mining；
- 评估是否需要轻量 transformer/UNet bottleneck 提升边界建模能力。

## 13. 组会答辩口径

如果被问“这个模型有什么用”：

> 它给下游世界模型提供可重投影的 dense 3D point map 和 motion-conditioned confidence。高置信区域直接重投影复用，低置信区域交给 DiT/refiner，从而在保持质量的前提下降低 active token 和延迟。

如果被问“是不是只是 MoGe-3 蒸馏”：

> 不是。MoGe-3 是 offline teacher；我们的训练目标加入了 projection、target-view occupancy、motion-conditioned confidence 和 hard-case coverage repair，优化的是重投影可用性和下游路由价值。

如果被问“泛化性如何”：

> 当前在 Matrix-Game validation 上稳定，demo 跨域 evidence 仍不足；我们已经准备好跨域队列，下一轮空卡会补 Matrix-Game-2/3 和 MoGe example。论文级结论需要每域 30–100+ 帧。

如果被问“收益是否为正”：

> 当前 proxy 为正：高运动片段 estimated active ratio 约 0.22–0.56，对应 active-DiT proxy 约 1.96x–3.69x；但这还不是最终 end-to-end speedup，需要下游闭环实验确认。

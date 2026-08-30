# M1 Completion Experiment Pack

## 当前结论

- v10 当前 hard-case 平均 coverage gap 为 -0.0577，最差为 -0.2001；主要问题场景是 `{'scene': 'game2_mid_left', 'num_cases': 12, 'student_coverage_mean': 0.7191326666666668, 'teacher_coverage_mean': 0.8295599166666667, 'coverage_gap_mean': -0.11042724999999998, 'coverage_gap_min': -0.20011999999999996}`。
- 跨域验证当前覆盖 1 个域、5 个样本；论文级目标是每域 >=30。
- 下游闭环 proxy 显示，在高运动片段中可用安全重投影 tile 降低 active DiT token；当前最佳估计 speedup 为 3.69x。
- v11 hard-case 候选数：0；若候选训练完成，将按最差 coverage gap 优先自动排序。

## 必须补齐的实验

1. Cross-domain：GTA/Temple/Universal 每域至少 30 帧，导出 MoGe-3 teacher，再评估 M1-v7/v10 student。
2. Hard-case repair：基于 v7 初始化，开启 coverage-deficit loss 与 depth-edge point loss，重点优化 game2_mid_left、game3_left 等 coverage gap 场景。
3. Downstream closed-loop：用 real target frames 评估 copy/warp/oracle/gated-DiT 分层收益，按低运动/高运动分别报告。

## 论文图输出

- `figures/m1_cross_domain_demo_validation.svg`
- `figures/m1_hardcase_coverage_gap.svg`
- `figures/m1_closed_loop_proxy_speedup.svg`


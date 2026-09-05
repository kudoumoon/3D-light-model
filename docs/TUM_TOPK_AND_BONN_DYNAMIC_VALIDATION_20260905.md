# TUM top-10 回退诊断与 Bonn 动态场景验证

日期：2026-09-05

## 结论

TUM top-10 precision 从 92.77% 降到 89.00%，不是 Geometry Head 退化造成的。两次评估使用同一个冻结几何模型，AUC 也基本不变（0.7928 对 0.7929）。主要变化发生在 confidence 的跨场景分数尺度：联合训练改变了 motion normalization，并让全局 top-10 中 `tum_xyz_test` 的占比从 71.52% 上升到 83.49%。`xyz` 场景比 `rpy` 场景难，因此全局 top-10 precision 随之下降。

替换归一化统计、保持联合模型权重不变，可把 TUM top-10 从 89.00% 恢复到 91.24%。域均衡采样把它恢复到 90.76%，但同时使 ETH3D top-10 从 97.86% 降到 96.76%，TUM ECE 也从 0.0514 增至 0.0786。域均衡是有效消融，不应替代原联合 checkpoint。当前最稳妥的报告方式是保留两条线：TUM-only 给出域内上界，joint model 反映跨域概率质量。

Bonn 三个动态序列上的零样本测试不支持“warp 稳定优于 Copy”这一结论。短间隔下，GT depth warp 和 Student warp 都比 Copy 差；最长间隔时二者接近打平。由于 GT geometry 也未通过，不能把该负结果归因于 Student 深度误差。Bonn 中的运动物体不服从相机刚体变换，WanVAE latent 对像素坐标变换也不是严格等变的，这两项足以解释当前差距。没有动态物体 mask，暂时不能定量拆开各自贡献。

## 一、TUM top-10 回退

### 1. 对照结果

| 训练方式 | TUM AUC | TUM ECE | TUM top-10 precision | ETH3D AUC | ETH3D ECE | ETH3D top-10 precision |
|---|---:|---:|---:|---:|---:|---:|
| TUM-only (`p88`) | 0.7928 | 0.0620 | 92.77% | 0.6420（zero-shot） | 0.1540 | 93.70% |
| TUM+ETH3D joint (`p92`) | 0.7929 | 0.0514 | 89.00% | 0.7611 | 0.0239 | 97.86% |
| domain-balanced joint (`p105`) | 0.7927 | 0.0786 | 90.76% | 0.7483 | 0.0357 | 96.76% |

联合训练显著改善 ETH3D 的 AUC 和校准，同时牺牲 TUM 高置信区域的排序。它是多域优化折中，不是整体崩溃。

### 2. 因果拆解

`p101` 固定模型权重，只交换 motion normalization：

| 权重 | normalization | TUM top-10 | top-10 中 xyz 占比 |
|---|---|---:|---:|
| TUM-only | TUM | 92.77% | 71.52% |
| joint | joint | 89.00% | 83.49% |
| joint | TUM | 91.24% | 77.71% |
| TUM-only | joint | 89.04% | 82.11% |

仅交换 normalization 就能移动 2 个百分点以上，说明分数尺度漂移是主因。剩余差距来自联合 BCE 的权重折中：ETH3D projected cell 的正样本率为 84.53%，TUM 只有 44.85%，普通 pair sampling 会让 ETH3D 的大量正 cell 主导梯度。

`p105` 使用 inverse-domain-frequency sampler，模型结构、loss 和 31K 参数量均未变化。它降低了 top-10 的 xyz 占比至 74.84%，TUM top-10 恢复到 90.76%。代价是两个域的 AUC/ECE 都略有变差，因此不升级默认 checkpoint。

## 二、Bonn 动态场景设置

使用 Bonn RGB-D 的 `balloon`、`balloon2` 和 `person_tracking`。RGB、深度与传感器 GT 位姿按 0.03 秒阈值同步，每 4 帧采样一次。冻结 WanVAE 将 640×352 RGB 编码为 `[16, 44, 80]` latent；GT depth 采用 valid-aware median 8×8 pooling 对齐到 latent grid。未使用 Bonn 训练、校准或阈值选择。

| 序列 | 同步帧 | 采样帧 | 短间隔 pairs |
|---|---:|---:|---:|
| balloon | 438 | 110 | 323 |
| balloon2 | 468 | 117 | 344 |
| person_tracking | 580 | 145 | 428 |
| 合计 | 1486 | 372 | 1095 |

长间隔测试加入 sample delta 8 和 16，共 1767 pairs。这里的 delta 1 对应原始序列约 4 帧。

## 三、Bonn 单源 transport

### 1. 覆盖率与几何上界

短间隔平均结果：

| 指标 | 结果 |
|---|---:|
| source GT depth valid | 90.81% |
| GT geometry renderer coverage | 88.35% |
| Student + GT valid coverage | 87.28% |
| Student inference coverage | 85.72% |
| reconstructable recall | 95.92% |
| reconstructable precision | 58.81% |

Student inference coverage 距 GT renderer 只有 2.63 个百分点，几何输出没有在 Bonn 上失效。precision 较低说明投影覆盖了不少不能由同一刚体表面重建的 cell，这与动态物体和遮挡冲突一致。

### 2. 同支持集 Warp-vs-Copy

| 序列 | Student coverage | Student warp L1 | Copy L1 | Student 胜率 |
|---|---:|---:|---:|---:|
| balloon | 87.15% | 0.2677 | 0.2546 | 36.53% |
| balloon2 | 87.12% | 0.2959 | 0.2816 | 27.62% |
| person_tracking | 83.52% | 0.2770 | 0.2557 | 17.99% |

GT geometry 的总体 warp L1 为 0.2755，Copy 为 0.2623，胜率 27.03%；Student 分别为 0.2802、0.2635 和 26.48%。Teacher 与 Student 的趋势一致。

### 3. 时间间隔

| sample delta | 平均运动 | Student coverage | GT warp / Copy L1 | Student warp / Copy L1 | Student 胜率 |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.26 px | 89.97% | 0.2230 / 0.2149 | 0.2262 / 0.2152 | 29.54% |
| 2 | 6.22 px | 86.57% | 0.2739 / 0.2608 | 0.2789 / 0.2621 | 25.41% |
| 4 | 11.92 px | 80.51% | 0.3308 / 0.3124 | 0.3368 / 0.3144 | 24.44% |
| 8 | 22.49 px | 70.76% | 0.3788 / 0.3615 | 0.3816 / 0.3645 | 28.16% |
| 16 | 40.12 px | 57.54% | 0.3921 / 0.3912 | 0.3951 / 0.3938 | 48.77% |

大运动削弱了 Copy 优势，但没有让 warp 稳定获胜。最大间隔的差距已经很小，后续若获得动态 mask，应先分别评估静态背景和运动物体，不应先改 Geometry Head。

## 四、多源 coverage 与 confidence

`p113` 使用无参数 temporal priority-fill：

| 指标 | 最佳单源 | priority-fill | 变化 |
|---|---:|---:|---:|
| coverage | 90.50% | 94.01% | +3.52 pp |
| safe coverage，L1≤0.2 | 47.58% | 48.29% | +0.70 pp |
| safe coverage，L1≤0.3 | 68.43% | 70.20% | +1.78 pp |
| safe coverage，L1≤0.4 | 80.51% | 83.25% | +2.74 pp |

priority-fill 的平均 L1 为 0.2300，merge latency 约 0.20 ms。nearest-depth union 虽有相同覆盖率，L1 为 0.3239，并使阈值 0.2 的 safe coverage 下降 22.47 pp。Bonn 再次支持“按时间优先填孔”，不支持无条件 nearest merge。

联合域均衡 confidence 在 Bonn 短间隔上的 AUC 为 0.7001，ECE 为 0.2629，top-10 precision 为 53.59%。TUM-only checkpoint 的 AUC 为 0.7092，top-10 precision 为 58.74%。长间隔时联合模型 AUC 升至 0.7263，但 ECE 仍为 0.2681。排序有一定信息，概率值严重过置信，不能直接作为跨域 reuse probability。

## 五、事实、负结果与待验证解释

### Fact

- TUM top-10 回退主要来自 motion normalization 与跨场景 top-k 配额变化；几何模型未变，AUC 未退化。
- 不增参数的域均衡采样可恢复 1.76 pp TUM top-10，但会轻微损害 ETH3D 与校准。
- Student 在 Bonn 的覆盖率接近 GT geometry renderer，多源 priority-fill 仍有稳定的 coverage 和 safe-coverage 收益。

### Negative result

- Bonn 动态序列中，即使使用 GT depth，latent warp 也未稳定优于 Copy。
- TUM/ETH3D 上得到的 confidence 概率不能零样本校准到 Bonn；temperature scaling 同样没有解决该问题。
- 3×3 micro-hole closure 的新增区域多数比 Copy 更差，不采用该方案。

### Hypothesis

- moving object 不满足相机位姿定义的刚体变换，是 person_tracking 最差的主要原因。
- WanVAE latent 并非严格的局部像素表征，显式 3D warp 会引入 feature interpolation error。
- 没有动态 mask 前，两项误差无法严格拆分。下一步最有价值的实验是静态背景 mask / optical-flow residual 分层，而不是继续加 head 或 loss。

## 六、当前决策

1. 不用 `p105` 覆盖 `p92`。`p105` 作为域均衡消融，`p88` 作为 TUM 上界，`p92` 保留为当前联合概率模型。
2. Bonn 结果按负结果报告。M1 在几何 coverage 上跨域成立，但“动态区域可直接 latent reuse”尚未成立。
3. 保留 temporal priority-fill。它没有新增参数，Bonn 上仍改善 safe coverage。
4. 在拿到带动态标注的数据或生成可靠 residual mask 前，不针对 Bonn 微调 Student，避免把 M1 问题和动态内容问题混在一起。

## 实验证据索引

- TUM 因果审计：`results/latent3d/p101_tum_joint_confidence_topk_diagnostic_v1/metrics.json`
- 域均衡训练及分域评估：`p105`、`p106`、`p107`、`p109`
- Bonn 冻结 VAE cache 记录：`p108_bonn_dynamic_cache_v1/metrics.json`
- Bonn 分场景/时间间隔审计：`p120`、`p122`
- Bonn 多源 coverage/confidence：`p113`、`p114`
- Bonn confidence 对照：`p111`、`p112`、`p115`、`p117`、`p118`、`p119`、`p123`

原始 `cache.pt` 不进入 Git；配置、metrics、pair metadata 和脚本进入版本控制。

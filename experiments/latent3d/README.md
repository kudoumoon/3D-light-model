# Geometry-Aligned Latent 3D experiments

本目录从 P0 开始保存 Latent 3D 实验。现有 v7 和 v10 不会被覆盖；其 checkpoint、哈希和主要指标见 `BASELINE_LOCK.json`。

P1 固定 Matrix-Game 2 的 Wan VAE。它使用 16 个 latent 通道，空间压缩率为 8；352×640 RGB 对应 44×80 latent grid。时间压缩率为 4，首个 latent 与首帧的因果对应关系不同于后续 latent。第一版只研究单帧的空间 latent，不把时间压缩后的 cell 当成单一时刻的几何实体。

P2 是训练前的硬门。每个样本必须包含 source RGB、target RGB、可靠的 source-to-target pose 和相机内参。实验将冻结 VAE 后得到 `z_s`、`z_t`，以 MoGe-3 或几何真值投影 `z_s` 到 target latent grid，并与 Copy baseline 比较 latent L1/L2、cosine、coverage、hole ratio，以及 decode 后的 PSNR、SSIM、LPIPS。若 teacher geometry 不能稳定超过 Copy，不启动 student 训练。

当前仓库的 1880 个 MoGe-3 teacher frame 没有可靠的跨帧相对位姿。它们可用于单帧蒸馏，不能作为 P2 的真实 target-latent 证据。下一步需要引入带 calibrated pose 的帧对；数据清单、位姿来源、VAE 哈希和 GPU 信息必须与每次结果一起保存。

P3 只比较可解释的 geometry alignment：center、average、median、minimum、confidence-weighted，以及一个独立训练的 learned pooling。选择标准是 target latent warp 和 decoded image 的质量，不是单独的 depth error。

# 复现指南

## 0. 两种复现级别

1. **结果复核**：用已保存 JSON 重建表格/图表，无需 GPU、权重、视频。运行 README 中的 `summarize_results.py`。
2. **重新实验**：重新下载上游资源、提取几何、Warp、估计离线位姿和计时。需要 CUDA；得到的新耗时不必与历史笔记逐位相同。

`results/recorded` 是历史记录，不要用新实验覆盖它。新输出放入默认忽略的 `runs/`。首次下载与模型加载时间应另行报告。

## 1. 环境与固定版本

原本机虚拟环境在整理时（2026-08-28）检查到：Python 3.12.10；torch 2.11.0+cu128；torchvision 0.26.0+cu128；numpy 2.4.4；opencv-python 5.0.0.93；scipy 1.18.0；matplotlib 3.11.1；Pillow 12.2.0；huggingface-hub 1.27.0。此记录不是跨系统兼容保证。

安装命令见 README。`requirements-analysis.txt` 只包含结果重建/CPU 测试所需包；`requirements.txt` 为几何前端额外依赖。PyTorch 单独安装，避免默认 CPU wheel。随后：

```bash
python tools/setup_dependencies.py
python -m pip install -e third_party/MoGe
```

依赖脚本锁定本次实验实际使用的版本：

- MoGe：`925b8ed835a7a9cdb7578ba15c658a0afc969030`
- Matrix-Game：`71c3cd7f741311f8100f6cf9cde942b6c1378d11`

它只创建指定的第三方目录；已有目录会检查版本，不自动覆盖或重置已有改动。上游 main 后续可能改变安装要求，复现实验请勿直接换到最新版。

## 2. 三张独立图像的几何实验

```bash
python geometry_provider.py --image third_party/Matrix-Game/Matrix-Game-2/demo_images/gta_drive/0000.png --output runs/gta --max-size 640 --num-tokens 1200 --warmup 1 --repeat 3
python geometry_provider.py --image third_party/Matrix-Game/Matrix-Game-2/demo_images/temple_run/0000.png --output runs/temple --max-size 640 --num-tokens 1200 --warmup 1 --repeat 3
python geometry_provider.py --image third_party/Matrix-Game/Matrix-Game-2/demo_images/universal/0003.png --output runs/universal --max-size 640 --num-tokens 1200 --warmup 1 --repeat 3
```

每个目录含 `geometry.npz`、`pointcloud.ply`、RGB/深度/法线/mask 预览和 `metadata.json`。点图是单视角估计，不能通过看点云就断言三维几何正确。

## 3. CPU 与 CUDA Warp

```bash
python reproject.py --geometry runs/gta/geometry.npz --output runs/gta_cpu --yaw 5 --forward 0.10 --splat-radius 1 --warmup 2 --repeat 10
python reproject_torch.py --geometry runs/gta/geometry.npz --output runs/gta_cuda --yaw 5 --forward 0.10 --splat-radius 1 --warmup 10 --repeat 50
```

将 yaw 改为 2/10，或 geometry 改为其他场景，得到覆盖率扫描。模型推理、上传、重投影、下载、显示应分段记录，不能只计最快的一段。

## 4. Active-query 结构代理

```bash
python benchmark_active_dit.py --output runs/proxy_large --tokens 4096 --width 768 --heads 12 --layers 12 --preheat 12 --warmup 3 --repeat 15 --inner 1 --ratios 1.0 0.75 0.5 0.25 0.125
python benchmark_active_dit.py --output runs/proxy_small --tokens 2048 --width 512 --heads 8 --layers 8 --preheat 80 --warmup 5 --repeat 20 --inner 3 --ratios 1.0 0.75 0.5 0.25 0.125 0.0625
```

模型权重为随机初始化，固定种子 7，K/V 已驻留。用于测试运行时伸缩性，不能输出有意义的游戏视频。新版本图表自动使用正确 blocks 数量；历史 large 图曾错误标为 8-block。

## 5. 视频帧对实验

视频位于 `third_party/Matrix-Game/Matrix-Game-2/assets/videos/matrix-game2.mp4`。这是一段拼接展示视频；不能将其中全部相邻画面看成单一连续视角。

固定 crop：Straight `(655,420,654,300)`；Turning `(0,420,654,300)`。后续 resize 到 640×294。原始图片仍含按键 HUD、车身、边框和压缩误差。

运行准备脚本将 source/target 帧一次性提取，并给源帧估计几何：

```bash
python tools/prepare_demo_pairs.py --output runs/demo
```

这会进行 14 次几何提取（每次独立加载模型，不代表高效在线系统）。随后：

```bash
python evaluate_target_pairs.py --frames runs/demo/straight/frames --geometry runs/demo/straight/geometry --output runs/demo/straight/eval_short --sources 0 15 30 45 60 75 90 --deltas 3 6 --tile 16
python evaluate_target_pairs.py --frames runs/demo/straight/frames --geometry runs/demo/straight/geometry --output runs/demo/straight/eval_gap8 --sources 0 15 30 45 60 75 --deltas 8 --tile 16
python evaluate_target_pairs.py --frames runs/demo/straight/frames --geometry runs/demo/straight/geometry --output runs/demo/straight/eval_long --sources 0 15 30 45 60 75 --deltas 15 30 --tile 16
```

将 `straight` 替换成 `turning` 重复评估。准备脚本只提取不晚于 frame 93 的目标帧，因此 90→96、75→105 等会被显式标成 skipped；旧结果中它们被直接略过。新脚本固定 OpenCV seed=7；历史运行没有明确固定 PnP seed，重新运行可能有差异。

每个成功帧对保存 source/target/warped/mask、RGB-MAE 数组、montage 和 metrics；summary 同时保留 failures/skipped。不要拿一次失败前遗留的目录冒充本次成功结果。

## 6. 数据来源与可复核性

`results/provenance.json` 记录 26 份原始记录和示例图的 SHA-256。公开 JSON 仅将个人机器绝对路径改为逻辑相对路径，并统一格式；数值不变。

新增的逐块 Oracle 来自原实验已保存的 source/target/mask/pixel_mae，不是新生成图像；`tools/export_recorded.py` 提供整理流程。公开包不携带全部帧和 dense error arrays，完全重算这些新增指标需先运行帧对实验，或使用原始本地实验目录运行该导出工具。

结果重建需要的全部汇总和逐对数值已随仓库提供。更大规模数据集、游戏真值、LPIPS/FVD、action accuracy、真实 DiT 延迟目前均未提供。

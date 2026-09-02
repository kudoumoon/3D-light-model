# Latent 3D 真实数据集筛选与实验计划

## 结论

最合适的推进顺序不是先下载最大的集合，而是先用 TUM RGB-D 和 Bonn Dynamic 通过不依赖 MoGe 的 feasibility gate，再用 ScanNet 或 ARKitScenes 扩大训练规模，最后以 ETH3D、KITTI/Waymo 和 DL3DV 检查跨域边界。这样每一阶段都回答一个清楚的问题：latent 能否 warp、动态区域的 confidence 是否可信、模型是否能跨场景和跨域。

## 适配度排序

| 优先级 | 数据集 | 真实 RGB | 深度 | 位姿与内参 | 获取方式 | 在本项目中的用途 |
|---:|---|---|---|---|---|---|
| 1 | [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) | Kinect | 注册深度 | mocap GT trajectory；公开 K | 直接下载，主要内容 CC BY 4.0 | 第一条 MoGe-free 硬门；平移、旋转和动态序列都可分开测 |
| 2 | [Bonn RGB-D Dynamic](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/) | RGB-D sensor | 注册深度 | OptiTrack GT pose；公开 K | 直接下载 | 动态物体、遮挡、置信度校准和 hard case |
| 3 | [ScanNet](https://www.scan-net.org/ScanNet/) | 手持 RGB-D | dense sensor depth | 每帧 pose、intrinsics、extrinsics | 注册并接受条款 | 大规模室内训练；必须按 physical scene 切分 |
| 4 | [ARKitScenes](https://github.com/apple/ARKitScenes) | iPad video/RGB | LiDAR depth、confidence、部分 Faro scan | `.traj` pose；逐帧 `.pincam` | 官方脚本，接受 license | 大规模移动端室内训练；天然提供 depth confidence |
| 5 | [ETH3D](https://www.eth3d.net/datasets) | 高分辨率多视图 | laser-rendered depth | `cameras.txt`、`images.txt` | 直接下载 | 静态高质量测试；薄结构、边界、反光区域压力测试 |
| 6 | [DTU Robot Image Data](https://roboimagedata.compute.dtu.dk/) | 真实受控相机 | structured-light scan | calibrated camera positions | 直接下载并引用 | 物体级深度边界与 latent pooling 消融 |
| 7 | [KITTI Raw](https://www.cvlibs.net/datasets/kitti/raw_data.php) | 双目驾驶 | 稀疏 Velodyne | camera/LiDAR/GPS-IMU calibration | 直接下载 | 室外跨域；只在 LiDAR 投影有效像素评价几何 |
| 8 | [Waymo Open](https://waymo.com/open/) | 五相机驾驶 | 多 LiDAR 稀疏深度 | vehicle pose、camera K/extrinsics | 注册，研究许可 | 大规模动态室外；rolling shutter 与 object motion hard case |
| 9 | [nuScenes](https://www.nuscenes.org/nuscenes) | 六相机 | 稀疏 LiDAR/radar | calibrated sensor、ego pose、K | 注册，非商业研究 | 多传感器跨域和动态遮挡，不作为 dense-depth 主训练集 |
| 10 | [DL3DV-10K](https://github.com/DL3DV-10K/Dataset) | 真实多场景视频 | 无原生 dense GT | COLMAP camera parameters | Hugging Face 申请 | 大规模外观泛化；pose 是重建估计，不能写成 GT depth |
| 11 | [7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) | Kinect | dense depth | KinectFusion/ICP pose | 直接下载 | 小型 sanity check；pose 并非独立 mocap |
| 12 | [Tanks and Temples](https://www.tanksandtemples.org/download/) | 真实 4K video | 部分 laser GT | COLMAP pose；K 不完整 | 非商业下载 | NVS 压力测试，因 K 口径不够干净而后置 |

## 分阶段实验

### R0：GT feasibility，不训练 Student

- 数据：TUM `freiburg1_xyz`，随后加 `freiburg1_rpy`。
- 输入：source RGB、target RGB、sensor depth、mocap pose、官方 K。
- 比较：Copy、GT-depth latent warp；pooling 比较 center/average/median/minimum。
- 通过条件：在 1–4 cells 和 ≥4 cells 两个运动区间，warp latent L1 与 cosine 均稳定优于 Copy；同时报告 coverage、hole ratio、decoded PSNR/SSIM。
- 若不通过：先检查 pose convention、RGB-depth 同步、畸变和 VAE latent transportability，不训练 Student。

### R1：动态置信度

使用 Bonn 的静态与动态序列。几何保持冻结，只训练 confidence head。标签由 projected-valid、depth conflict、动态区域风险和实际 latent warp error共同构成。主要指标为 AUC、ECE、kept ratio，以及高置信 cell 的 warp gain。

### R2：规模训练

ScanNet 与 ARKitScenes 二选一作为主训练源，另一套只做跨数据集验证。优先采用传感器 depth；缺失或低 confidence 区域可使用 teacher 补充，但损失中必须保留 supervision source mask，禁止把补全值伪装成传感器 GT。

### R3：跨域与 hard cases

- ETH3D/DTU：薄结构、深度不连续和反光区域。
- KITTI/Waymo：室外尺度、远距离稀疏深度、动态车辆。
- DL3DV：外观和场景类型泛化，只报告 pose-conditioned warp，不报告 GT depth error。

## 数据与评测纪律

1. train/val/test 按物理 scene 切分，不能按帧或 crop 随机切分。
2. pair 按投影到 latent 网格的运动量分为 sub-cell、1–4 cells、≥4 cells，避免 Copy 被大量小运动样本稀释。
3. 每个像素保留监督来源：sensor、laser/MVS、MoGe pseudo、invalid。
4. 真实传感器深度的空洞、量程和飞点不做隐式兜底；所有过滤阈值写入 config。
5. VAE 版本、checkpoint hash、裁剪缩放后的 K 和 source→target 坐标约定必须随 metrics 保存。
6. 数据本体不上传 GitHub，仅保存来源、许可、校验和、处理脚本和样本清单。

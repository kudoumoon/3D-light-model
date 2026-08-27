# 实验贡献规范

新实验不要覆盖 `results/recorded/`。输出先保存到 `runs/日期_实验名/`，确认可分享后再选择性加入新的记录目录。

每次提交应提供：

- 要验证的问题与失败条件；明确 Fact / Inference / Hypothesis。
- 数据来源、完整轨迹划分、版本/权重、输入分辨率、坐标约定、动作/位姿来源。
- 硬件、操作系统、PyTorch/CUDA、精度、warmup/repeat、同步计时边界、原始每次耗时。
- Copy、全量生成、无几何复用等强基线；质量和速度使用同一组输入与预算。
- 失败、跳过、误放行、最坏情形与 p95/p99，不只报告均值或挑选的成功图。
- 保留图表生成代码、JSON、来源 SHA-256；不要提交密钥、权重、虚拟环境或个人路径。

提交前运行：

```bash
python -m unittest discover -s tests -v
python summarize_results.py
python tools/verify_release.py
git diff --check
```

单元测试通过不代表生成质量或端到端实时性成立。使用 target-assisted pose 或 target-error oracle 的结果必须明确为离线评估。

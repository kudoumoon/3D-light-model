# 论文写作 Skill 配置记录

日期：2026-08-30

本次按用户要求配置以下写作辅助 skill，用于 M1 论文内容整理。当前只用于中文 method、related work、experiments/discussion 草稿，不生成论文图。

## 已配置 skill

| Skill | 来源 | 当前用途 |
|---|---|---|
| nature-writing | https://github.com/Yuan1z0825/nature-skills | 方法、相关工作、实验叙述结构规划 |
| nature-polishing | https://github.com/Yuan1z0825/nature-skills | 论文段落逻辑与表达润色 |
| nature-figure | https://github.com/Yuan1z0825/nature-skills | 已安装但暂不调用；等待最终方案敲定后再生成论文图 |
| humanizer-zh | https://github.com/op7418/Humanizer-zh | 中文表达去模板化、减少 AI 味、保留具体证据 |

## 当前写作原则

1. 所有正文内容使用中文，代码名、变量名、模型名和指标名保留原始英文。
2. 所有结果只引用仓库中已有证据；未完成的实验明确标为待补齐，不写成已验证结论。
3. M1 的方法叙述按“任务定义、系统概览、模块细节、训练目标、下游接口、边界”组织。
4. 相关工作按技术机制分组，不按论文年份堆引用。
5. 暂不生成 Figure；等最终方案、实验结果和贡献边界确定后，再调用 nature-figure 设计论文图。

## 本次产出

- `docs/M1_PAPER_METHOD_RELATED_ZH.md`：M1 方法、相关工作、实验叙述骨架和 discussion 中文草稿。

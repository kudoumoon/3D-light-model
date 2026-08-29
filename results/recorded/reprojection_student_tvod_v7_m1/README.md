# Reprojection Student TVOD v7 M1

本目录记录 2026-08-29 的 M1 TVOD 规模化训练关键结果。

不提交 checkpoints、raw `.npz` predictions 或完整 `runs/`，只提交可审阅的 summary JSON 和报告。

主模型建议：

`runs/reprojection_student/student_video384_tvod_v7_2k_occ075_width64_lr15/checkpoints/best.pt`

摘要：

- teacher：`runs/teacher_moge3_video384_v6_2k`
- teacher samples：1880
- eval split samples：375
- reprojection benchmark：30 samples × 4 motions = 120 cases
- 主模型 speedup：3.33x
- 主模型 coverage ratio：0.931
- 主模型 SUP：3.10
- 主模型 mean coverage gap：-0.0577
- 主模型 worst coverage gap：-0.2001

详细解释见：

`docs/M1_TVOD_SCALEUP_REPORT.md`


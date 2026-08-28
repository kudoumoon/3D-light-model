# Reprojection-Friendly Student Prototype

Date: 2026-08-28

This is the first closed-loop student training experiment.  It is a prototype
trained on 29 Matrix-Game demo images with MoGe-3 ViT-L teacher geometry at
`max_size=384`.  It validates the training pipeline, experiment logging, and
reprojection-oriented loss.  It is not yet a final generalizable geometry model.

## Model

- Student: small U-Net style CNN in `tools/train_reprojection_student.py`
- Output channels: scaled point map, mask logits, normal
- Depth constraint: predicted z uses `softplus(z) + 1e-3`
- Teacher: MoGe-3 ViT-L, `refine_steps=0`
- Train/val split: 24 train samples, 5 val samples

## Loss

The objective is intentionally reprojection-oriented:

- point-map distillation loss
- mask loss
- normal loss
- z-edge loss for sharper geometry boundaries
- projection-coordinate loss under small yaw perturbations

## Runs

| Run | Status | Notes |
|---|---|---|
| `student_demo384_v1` | failed | NaN target propagation from invalid teacher points. Recorded under `failed_v1_nan/`. |
| `student_demo384_smoke_zpos` | passed | 1 epoch smoke test after NaN sanitization and positive-z constraint. |
| `student_demo384_v2_zpos` | passed | 60 epoch formal prototype training. |

## Best Checkpoint Evaluation

Best checkpoint from `student_demo384_v2_zpos/checkpoints/best.pt`.

| Metric | Value |
|---|---:|
| val loss | 0.34898 |
| point loss | 0.22051 |
| mask loss | 0.17065 |
| normal loss | 0.21957 |
| edge loss | 0.05437 |
| projection loss | 0.02649 |
| mean median inference | 2.08 ms |
| mean p95 inference | 2.09 ms |

## Val Reprojection Coverage

Yaw 5 deg, forward 0.10, splat radius 1.

| Sample | Teacher coverage | Student coverage |
|---|---:|---:|
| `gta_drive__0005` | 83.97% | 53.55% |
| `temple_run__0005` | 78.93% | 48.99% |
| `universal__0014` | 89.54% | 54.36% |
| `universal__0015` | 78.00% | 59.50% |
| `universal__0016` | 89.11% | 43.43% |

## Current Interpretation

The student is much faster than MoGe-3 on this small H100 benchmark
(`~2 ms` vs `~23 ms` at nearby input scale), and its point-map output can enter
the CUDA reprojection pipeline.  Quality is still far below the teacher:
coverage is roughly 30 percentage points lower on the val split.  The immediate
research task is therefore not speed, but preserving teacher reprojection
coverage under distillation.

## Next Experiments

- Increase teacher data beyond the 29 bundled demo images.
- Add explicit projected valid-mask supervision instead of only source valid
  mask supervision.
- Train tile-level `warp_confidence`, `disocclusion_risk`, and
  `depth_conflict_risk` heads.
- Replace the tiny CNN with a stronger lightweight backbone before claiming
  generalization.


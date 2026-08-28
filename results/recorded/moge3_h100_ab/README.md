# MoGe-3 H100 Geometry and Reprojection A/B

Date: 2026-08-28

This folder contains curated GitHub-friendly artifacts from the MoGe-3 ViT-L H100 run. Large local files such as `checkpoints/*.pt`, `runs/*`, and `geometry.npz` are intentionally not tracked by git.

## Checkpoint

- Model: MoGe-3 ViT-L, `refine_steps=0` recommended for the deliverable geometry baseline
- Upstream MoGe commit: `74fbce054ebed49800de42d0ad0e83495065719a`
- Local checkpoint: `checkpoints/moge-3-vitl/model.pt`
- Size: `1,481,333,394` bytes
- SHA256: `9b41b7b9f65ad80aab7ad686f5e9cc0d1fd33f1964022618dfbcd52fc1fb7925`

## Geometry Inference

Resident RGB/model `infer()` latency on H100, excluding download, decoding, H2D, and export.

| Scene | Model | Steps | p50 ms | p95 ms | Valid mask |
|---|---:|---:|---:|---:|---:|
| GTA | MoGe-2 Small | 0 | 13.97 | 17.12 | 96.31% |
| Temple | MoGe-2 Small | 0 | 13.69 | 13.76 | 49.67% |
| Universal | MoGe-2 Small | 0 | 13.77 | 14.46 | 79.56% |
| GTA | MoGe-3 ViT-L | 0 | 23.33 | 23.40 | 96.36% |
| Temple | MoGe-3 ViT-L | 0 | 23.64 | 23.75 | 70.72% |
| Universal | MoGe-3 ViT-L | 0 | 22.81 | 22.86 | 79.73% |

## Reprojection

Pose perturbation: yaw 5 deg, forward 0.10, splat radius 1.

| Scene | Geometry | Coverage | Resident GPU p50 ms | Upload+GPU p50 ms |
|---|---:|---:|---:|---:|
| GTA | MoGe-2 Small | 88.98% | 1.214 | 1.597 |
| Temple | MoGe-2 Small | 46.13% | 1.021 | 1.474 |
| Universal | MoGe-2 Small | 70.90% | 1.106 | 1.483 |
| GTA | MoGe-3 ViT-L step0 | 88.73% | 1.130 | 1.503 |
| Temple | MoGe-3 ViT-L step0 | 63.40% | 1.104 | 1.532 |
| Universal | MoGe-3 ViT-L step0 | 72.07% | 1.105 | 1.494 |

## Visual Artifacts

- GTA: [warped_rgb](gta/step0/warped_rgb.png), [warped_mask](gta/step0/warped_mask.png), [holes](gta/step0/warped_holes_magenta.png)
- Temple: [warped_rgb](temple/step0/warped_rgb.png), [warped_mask](temple/step0/warped_mask.png), [holes](temple/step0/warped_holes_magenta.png)
- Universal: [warped_rgb](universal/step0/warped_rgb.png), [warped_mask](universal/step0/warped_mask.png), [holes](universal/step0/warped_holes_magenta.png)

## Related Report

See `docs/DELIVERABLE_GEOMETRY_MODEL_AND_REPROJECTION_PLAN.md` for the deliverable model decision and the reprojection-friendly geometry design.


"""Train a motion-conditioned reprojection geometry student.

This experiment keeps the same point/mask/normal geometry outputs as the TVOD
student, but changes the warp-confidence head from a single unconditional map to
a motion-conditioned prediction.  The downstream reprojection module can query a
confidence map for the actual virtual camera motion instead of relying on one
average confidence map for all yaw/forward settings.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MotionConditionedReprojectionStudent(nn.Module):
    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.enc1 = ConvBlock(3, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.enc3 = ConvBlock(width * 2, width * 4)
        self.mid = ConvBlock(width * 4, width * 4)
        self.up2 = ConvBlock(width * 6, width * 2)
        self.up1 = ConvBlock(width * 3, width)
        self.geometry_head = nn.Conv2d(width, 7, 1)
        self.motion_head = nn.Sequential(
            nn.Conv2d(width + 2, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, 1, 1),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        mid = self.mid(e3)
        u2 = F.interpolate(mid, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.up1(torch.cat([u1, e1], dim=1))

    def forward(self, x: torch.Tensor, motion: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.features(x)
        out = self.geometry_head(feat)
        points_scaled = torch.cat([out[:, 0:2], F.softplus(out[:, 2:3]) + 1e-3], dim=1)
        motion = motion.to(dtype=feat.dtype, device=feat.device).view(feat.shape[0], 2, 1, 1)
        motion_map = motion.expand(-1, -1, feat.shape[-2], feat.shape[-1])
        warp_logits = self.motion_head(torch.cat([feat, motion_map], dim=1))
        return {
            "points_scaled": points_scaled,
            "mask_logits": out[:, 3:4],
            "warp_logits": warp_logits,
            "normal": F.normalize(out[:, 4:7], dim=1, eps=1e-6),
        }


class TeacherDataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        by_scene: dict[str, list[dict]] = defaultdict(list)
        for record in manifest["records"]:
            by_scene[record["scene"]].append(record)
        selected = []
        for records in by_scene.values():
            records = sorted(records, key=lambda row: row["sample_id"])
            holdout = max(1, round(len(records) * 0.2))
            if split == "train":
                selected.extend(records[:-holdout])
            elif split == "val":
                selected.extend(records[-holdout:])
            else:
                raise ValueError(split)
        self.root = root
        self.records = selected

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float]:
        record = self.records[index]
        data = np.load(self.root / record["geometry"], allow_pickle=False)
        rgb = data["rgb"].astype(np.float32) / 255.0
        points = data["points"].astype(np.float32)
        mask = data["mask"].astype(bool)
        normal = data["normal"].astype(np.float32)
        depth = data["depth"].astype(np.float32)
        if normal.shape != points.shape:
            normal = np.zeros_like(points, dtype=np.float32)
        valid = mask & np.isfinite(points).all(axis=-1) & np.isfinite(depth) & (depth > 0)
        median_depth = float(np.median(depth[valid])) if valid.any() else 1.0
        median_depth = max(median_depth, 1e-4)
        points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
        normal = np.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)
        points_scaled = np.clip(points / median_depth, -4.0, 4.0).astype(np.float32)
        return {
            "id": record["sample_id"],
            "scene": record["scene"],
            "scale": median_depth,
            "image": torch.from_numpy(rgb).permute(2, 0, 1),
            "points_scaled": torch.from_numpy(points_scaled).permute(2, 0, 1),
            "points": torch.from_numpy(points).permute(2, 0, 1),
            "mask": torch.from_numpy(valid.astype(np.float32))[None],
            "normal": torch.from_numpy(normal).permute(2, 0, 1),
            "intrinsics": torch.from_numpy(data["intrinsics"].astype(np.float32)),
            "rgb_uint8": torch.from_numpy(data["rgb"]),
        }


def rotation_yaw(yaw_deg: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    angle = math.radians(yaw_deg)
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], device=device, dtype=dtype)


def encode_motion(yaw: float, forward: float, device: torch.device) -> torch.Tensor:
    return torch.tensor([[yaw / 10.0, forward / 0.10]], dtype=torch.float32, device=device)


def project(points: torch.Tensor, intrinsics: torch.Tensor, yaw: float, forward: float) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, height, width = points.shape
    xyz = points.permute(0, 2, 3, 1).reshape(batch, -1, 3)
    center = torch.tensor([0.0, 0.0, forward], dtype=xyz.dtype, device=xyz.device)
    rot = rotation_yaw(yaw, xyz.device, xyz.dtype)
    target = (xyz - center) @ rot
    front = target[..., 2] > 1e-4
    z = target[..., 2].clamp_min(1e-4)
    xy = target[..., :2] / z[..., None]
    k = intrinsics
    u = k[:, 0, 0:1] * xy[..., 0] + k[:, 0, 2:3]
    v = k[:, 1, 1:2] * xy[..., 1] + k[:, 1, 2:3]
    coords = torch.stack([u.reshape(batch, 1, height, width), v.reshape(batch, 1, height, width)], dim=2)
    inside = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    projected_valid = (front & inside).reshape(batch, 1, height, width)
    return coords, projected_valid.float()


def gradient_loss(pred_z: torch.Tensor, target_z: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask_x = valid[..., :, 1:] * valid[..., :, :-1]
    mask_y = valid[..., 1:, :] * valid[..., :-1, :]
    dx = (pred_z[..., :, 1:] - pred_z[..., :, :-1]) - (target_z[..., :, 1:] - target_z[..., :, :-1])
    dy = (pred_z[..., 1:, :] - pred_z[..., :-1, :]) - (target_z[..., 1:, :] - target_z[..., :-1, :])
    return (dx.abs() * mask_x).sum() / mask_x.sum().clamp_min(1.0) + (dy.abs() * mask_y).sum() / mask_y.sum().clamp_min(1.0)


def target_view_occupancy(coords: torch.Tensor, valid: torch.Tensor, grid_h: int = 32, grid_w: int = 64) -> torch.Tensor:
    batch = coords.shape[0]
    u = coords[:, :, 0].reshape(batch, -1)
    v = coords[:, :, 1].reshape(batch, -1)
    mask = valid.reshape(batch, -1).clamp(0.0, 1.0)
    inside = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    mask = mask * inside.float()
    x = (u.clamp(0.0, 1.0) * (grid_w - 1)).clamp(0.0, grid_w - 1)
    y = (v.clamp(0.0, 1.0) * (grid_h - 1)).clamp(0.0, grid_h - 1)
    x0 = x.floor().long()
    y0 = y.floor().long()
    x1 = (x0 + 1).clamp(max=grid_w - 1)
    y1 = (y0 + 1).clamp(max=grid_h - 1)
    wx = x - x0.float()
    wy = y - y0.float()
    occ = torch.zeros(batch, grid_h * grid_w, dtype=coords.dtype, device=coords.device)
    for xi, yi, weight in (
        (x0, y0, (1.0 - wx) * (1.0 - wy)),
        (x1, y0, wx * (1.0 - wy)),
        (x0, y1, (1.0 - wx) * wy),
        (x1, y1, wx * wy),
    ):
        occ.scatter_add_(1, yi * grid_w + xi, mask * weight)
    return (1.0 - torch.exp(-occ)).reshape(batch, 1, grid_h, grid_w)


def compute_loss(model: nn.Module, sample: dict, device: torch.device, weights: dict[str, float]) -> tuple[torch.Tensor, dict]:
    image = sample["image"].unsqueeze(0).to(device)
    target_scaled = sample["points_scaled"].unsqueeze(0).to(device)
    target_points = sample["points"].unsqueeze(0).to(device)
    valid = sample["mask"].unsqueeze(0).to(device)
    target_normal = sample["normal"].unsqueeze(0).to(device)
    intrinsics = sample["intrinsics"].unsqueeze(0).to(device)
    scale = torch.tensor(float(sample["scale"]), device=device).view(1, 1, 1, 1)
    yaw = random.choice([-10.0, -7.5, -5.0, -2.5, 2.5, 5.0, 7.5, 10.0])
    forward = random.choice([0.05, 0.10, 0.15])
    motion = encode_motion(yaw, forward, device)

    pred = model(image, motion)
    pred_scaled = pred["points_scaled"]
    pred_points = pred_scaled * scale
    point = (F.smooth_l1_loss(pred_scaled, target_scaled, reduction="none") * valid).sum() / (valid.sum() * 3).clamp_min(1.0)
    mask = F.binary_cross_entropy_with_logits(pred["mask_logits"], valid)
    normal = ((1.0 - (pred["normal"] * target_normal).sum(dim=1, keepdim=True).clamp(-1, 1)) * valid).sum() / valid.sum().clamp_min(1.0)
    edge = gradient_loss(pred_scaled[:, 2:3], target_scaled[:, 2:3], valid)
    flow_pred, _ = project(pred_points, intrinsics, yaw=yaw, forward=forward)
    flow_teacher, warp_target = project(target_points, intrinsics, yaw=yaw, forward=forward)
    warp_target = warp_target * valid
    proj = (F.smooth_l1_loss(flow_pred, flow_teacher, reduction="none") * valid.unsqueeze(2)).sum() / (valid.sum() * 2).clamp_min(1.0)
    warp = F.binary_cross_entropy_with_logits(pred["warp_logits"], warp_target)
    occ_pred = target_view_occupancy(flow_pred, valid)
    occ_teacher = target_view_occupancy(flow_teacher, warp_target)
    occupancy = F.smooth_l1_loss(occ_pred, occ_teacher)
    total = (
        weights["point"] * point
        + weights["mask"] * mask
        + weights["normal"] * normal
        + weights["edge"] * edge
        + weights["projection"] * proj
        + weights["warp"] * warp
        + weights.get("occupancy", 0.0) * occupancy
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "point": float(point.detach().cpu()),
        "mask": float(mask.detach().cpu()),
        "normal": float(normal.detach().cpu()),
        "edge": float(edge.detach().cpu()),
        "projection": float(proj.detach().cpu()),
        "warp": float(warp.detach().cpu()),
        "occupancy": float(occupancy.detach().cpu()),
        "warp_target_mean": float(warp_target.detach().mean().cpu()),
    }


def average(rows: list[dict]) -> dict:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def load_reprojection_student_init(
    model: MotionConditionedReprojectionStudent,
    checkpoint: Path,
    device: torch.device,
) -> dict[str, int | str]:
    """Initialize shared geometry layers from a TVOD ReprojectionStudentWarp checkpoint.

    The old checkpoint has an 8-channel head:
    points xyz, source mask, unconditional warp, normal xyz.  The motion model
    has a 7-channel geometry head: points xyz, source mask, normal xyz.  The
    motion-conditioned warp head is intentionally left randomly initialized.
    """

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    old = state["model"]
    new = model.state_dict()
    copied = 0
    skipped = 0
    for key, value in old.items():
        if key.startswith("head."):
            continue
        if key in new and tuple(new[key].shape) == tuple(value.shape):
            new[key] = value
            copied += 1
        else:
            skipped += 1
    if "head.weight" in old and "geometry_head.weight" in new:
        source_rows = torch.tensor([0, 1, 2, 3, 5, 6, 7], dtype=torch.long, device=old["head.weight"].device)
        if old["head.weight"].shape[1:] == new["geometry_head.weight"].shape[1:]:
            new["geometry_head.weight"] = old["head.weight"].index_select(0, source_rows)
            new["geometry_head.bias"] = old["head.bias"].index_select(0, source_rows)
            copied += 2
        else:
            skipped += 2
    model.load_state_dict(new)
    return {"checkpoint": checkpoint.as_posix(), "copied_tensors": copied, "skipped_tensors": skipped}


def freeze_base_geometry(model: MotionConditionedReprojectionStudent) -> dict[str, int]:
    """Freeze v7-initialized geometry layers and train only motion confidence.

    This keeps point map / source mask / normal behavior identical to the
    delivered TVOD geometry model while learning a target-motion-specific
    reprojection reliability head.
    """

    frozen = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        keep_trainable = name.startswith("motion_head.")
        parameter.requires_grad_(keep_trainable)
        if keep_trainable:
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()
    return {"frozen_parameters": frozen, "trainable_parameters": trainable}


@torch.inference_mode()
def export_prediction(model: nn.Module, sample: dict, device: torch.device, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    image = sample["image"].unsqueeze(0).to(device)
    default_motion = encode_motion(5.0, 0.10, device)
    pred = model(image, default_motion)
    scale = float(sample["scale"])
    points = (pred["points_scaled"][0].permute(1, 2, 0).cpu().numpy() * scale).astype(np.float32)
    source_conf = torch.sigmoid(pred["mask_logits"][0, 0]).cpu().numpy().astype(np.float32)
    warp_confidence = torch.sigmoid(pred["warp_logits"][0, 0]).cpu().numpy().astype(np.float32)
    mask = source_conf > 0.5
    normal = pred["normal"][0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
    rgb = sample["rgb_uint8"].numpy().astype(np.uint8)
    depth = points[..., 2].astype(np.float32)
    intrinsics = sample["intrinsics"].numpy().astype(np.float32)
    motion_confidence = {}
    for name, yaw, forward in [
        ("yaw_p5_fwd10", 5.0, 0.10),
        ("yaw_m5_fwd10", -5.0, 0.10),
        ("yaw_p10_fwd10", 10.0, 0.10),
        ("yaw_m10_fwd10", -10.0, 0.10),
    ]:
        m = encode_motion(yaw, forward, device)
        motion_pred = model(image, m)
        motion_confidence[f"warp_confidence_{name}"] = torch.sigmoid(motion_pred["warp_logits"][0, 0]).cpu().numpy().astype(np.float32)
    np.savez_compressed(
        out_dir / "geometry.npz",
        points=points,
        depth=depth,
        mask=mask,
        source_confidence=source_conf,
        warp_confidence=warp_confidence,
        intrinsics=intrinsics,
        normal=normal,
        rgb=rgb,
        coordinate_convention=np.array("opencv_x_right_y_down_z_forward"),
        **motion_confidence,
    )
    cv2.imwrite(str(out_dir / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "mask.png"), mask.astype(np.uint8) * 255)
    cv2.imwrite(str(out_dir / "warp_confidence.png"), np.clip(warp_confidence * 255.0, 0, 255).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=ROOT / "runs/teacher_moge3_video_384")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/reprojection_student")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--point-weight", type=float, default=1.0)
    parser.add_argument("--mask-weight", type=float, default=0.25)
    parser.add_argument("--normal-weight", type=float, default=0.10)
    parser.add_argument("--edge-weight", type=float, default=0.20)
    parser.add_argument("--projection-weight", type=float, default=5.0)
    parser.add_argument("--warp-weight", type=float, default=0.25)
    parser.add_argument("--occupancy-weight", type=float, default=0.75)
    parser.add_argument("--init-reprojection-checkpoint", type=Path, default=None)
    parser.add_argument("--freeze-base-geometry", action="store_true")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    run_name = args.name or time.strftime("student_motion_%Y%m%d_%H%M%S")
    run_dir = args.output / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    config = {key: (value.as_posix() if isinstance(value, Path) else value) for key, value in vars(args).items()} | {
        "run_name": run_name,
        "model": "MotionConditionedReprojectionStudent",
        "loss_weights": {
            "point": args.point_weight,
            "mask": args.mask_weight,
            "normal": args.normal_weight,
            "edge": args.edge_weight,
            "projection": args.projection_weight,
            "warp": args.warp_weight,
            "occupancy": args.occupancy_weight,
        },
        "note": "Motion-conditioned warp-confidence TVOD student for reprojection reliability.",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    train_set = TeacherDataset(args.teacher, "train")
    val_set = TeacherDataset(args.teacher, "val")
    model = MotionConditionedReprojectionStudent(width=args.width).to(device)
    init_report = None
    if args.init_reprojection_checkpoint is not None:
        init_report = load_reprojection_student_init(model, args.init_reprojection_checkpoint, device)
        (run_dir / "init_report.json").write_text(
            json.dumps(init_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    freeze_report = None
    if args.freeze_base_geometry:
        freeze_report = freeze_base_geometry(model)
        (run_dir / "freeze_report.json").write_text(
            json.dumps(freeze_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
    best_val = float("inf")
    metrics_path = run_dir / "metrics.jsonl"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_order = list(range(len(train_set)))
        random.shuffle(train_order)
        train_rows = []
        for index in train_order:
            sample = train_set[index]
            loss, metrics = compute_loss(model, sample, device, config["loss_weights"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_rows.append(metrics)
        model.eval()
        val_rows = []
        for index in range(len(val_set)):
            sample = val_set[index]
            _, metrics = compute_loss(model, sample, device, config["loss_weights"])
            val_rows.append(metrics)
        row = {"epoch": epoch, "train": average(train_rows), "val": average(val_rows)}
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        val_loss = row["val"]["loss"]
        state = {"model": model.state_dict(), "config": config, "epoch": epoch, "val_loss": val_loss}
        torch.save(state, run_dir / "checkpoints/last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, run_dir / "checkpoints/best.pt")

    model.eval()
    val_exports = []
    for index in range(len(val_set)):
        sample = val_set[index]
        pred_dir = run_dir / "predictions" / str(sample["id"])
        export_prediction(model, sample, device, pred_dir)
        val_exports.append({"sample_id": sample["id"], "scene": sample["scene"], "geometry": pred_dir.relative_to(run_dir).as_posix() + "/geometry.npz"})
    summary = {
        "run_dir": run_dir.as_posix(),
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "best_val_loss": best_val,
        "last_epoch": args.epochs,
        "init_report": init_report,
        "freeze_report": freeze_report,
        "val_exports": val_exports,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

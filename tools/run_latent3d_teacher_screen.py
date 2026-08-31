"""P2/P3 screening: frozen Wan VAE, teacher geometry and estimated-pose pairs.

This is explicitly a screening experiment.  It accepts only pairs previously
filtered by PnP-RANSAC and records that their poses are estimated, not GT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
VAE_SOURCE = ROOT / "third_party/Matrix-Game/Matrix-Game-2/wan/vae/wanx_vae_src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VAE_SOURCE))

from latent_geometry_alignment import align_depth_to_latent
from latent_geometry_head import points_from_depth
from latent_reprojection_loss import forward_splat_latent, latent_reprojection_loss
from vae import WanVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=12)
    parser.add_argument("--min-inliers", type=int, default=200)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.6)
    parser.add_argument("--max-median-reprojection-px", type=float, default=1.5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crop_box(height: int, width: int, output_height: int = 352, output_width: int = 640) -> tuple[int, int, int, int]:
    if height / width > output_height / output_width:
        crop_height = int(width * output_height / output_width)
        return 0, (height - crop_height) // 2, width, crop_height
    crop_width = int(height * output_width / output_height)
    return (width - crop_width) // 2, 0, crop_width, height


def prepare_rgb(rgb: np.ndarray) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    left, top, crop_width, crop_height = crop_box(*rgb.shape[:2])
    image = Image.fromarray(rgb).crop((left, top, left + crop_width, top + crop_height))
    image = image.resize((640, 352), Image.Resampling.BICUBIC)
    transform = v2.Compose((v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3)))
    return transform(image)[None, :, None], (left, top, crop_width, crop_height)


def transform_geometry(
    depth: np.ndarray, valid: np.ndarray, intrinsics: np.ndarray, crop: tuple[int, int, int, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply exactly the image crop/resize, then convert K to normalized output coordinates."""
    height, width = depth.shape
    left, top, crop_width, crop_height = crop
    depth_crop = torch.from_numpy(depth[top:top + crop_height, left:left + crop_width])[None, None].float()
    valid_crop = torch.from_numpy(valid[top:top + crop_height, left:left + crop_width])[None, None].float()
    depth_out = F.interpolate(depth_crop, size=(352, 640), mode="nearest-exact")
    valid_out = F.interpolate(valid_crop, size=(352, 640), mode="nearest-exact")
    pixel_k = np.array(
        [[intrinsics[0, 0] * width, intrinsics[0, 1] * width, intrinsics[0, 2] * width],
         [intrinsics[1, 0] * height, intrinsics[1, 1] * height, intrinsics[1, 2] * height],
         [0.0, 0.0, 1.0]], dtype=np.float32,
    )
    affine = np.array([[640 / crop_width, 0.0, -left * 640 / crop_width], [0.0, 352 / crop_height, -top * 352 / crop_height], [0.0, 0.0, 1.0]], dtype=np.float32)
    pixel_k = affine @ pixel_k
    normalized = np.array(
        [[pixel_k[0, 0] / 640, pixel_k[0, 1] / 640, pixel_k[0, 2] / 640],
         [pixel_k[1, 0] / 352, pixel_k[1, 1] / 352, pixel_k[1, 2] / 352],
         [0.0, 0.0, 1.0]], dtype=np.float32,
    )
    return depth_out, valid_out, torch.from_numpy(normalized)[None]


def global_ssim(first: torch.Tensor, second: torch.Tensor) -> float:
    mean_first, mean_second = first.mean(), second.mean()
    variance_first = first.var(unbiased=False)
    variance_second = second.var(unbiased=False)
    covariance = ((first - mean_first) * (second - mean_second)).mean()
    return float(((2 * mean_first * mean_second + 0.01**2) * (2 * covariance + 0.03**2) / ((mean_first.square() + mean_second.square() + 0.01**2) * (variance_first + variance_second + 0.03**2))).cpu())


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one pre-approved GPU through CUDA_VISIBLE_DEVICES")
    manifest = json.loads((args.teacher_root / "manifest.json").read_text())
    records = {record["sample_id"]: record for record in manifest["records"]}
    raw_pairs = json.loads(args.pairs.read_text())["pairs"]
    pairs = [pair for pair in raw_pairs if pair.get("ok") and pair.get("inliers", 0) >= args.min_inliers and pair.get("inlier_ratio", 0.0) >= args.min_inlier_ratio and pair.get("median_reprojection_px", float("inf")) <= args.max_median_reprojection_px]
    pairs = pairs[:args.max_pairs]
    if not pairs:
        raise RuntimeError("no pose-screened pairs satisfy the configured reliability gate")
    device = torch.device("cuda:0")
    vae = WanVAE(pretrained_path=str(args.vae_checkpoint)).to(device, torch.bfloat16).eval().requires_grad_(False)
    tiler = {"tiled": True, "tile_size": (44, 80), "tile_stride": (23, 38)}
    methods = ("center", "average", "median", "minimum")
    rows: list[dict] = []
    for pair in pairs:
        source = np.load(args.teacher_root / records[pair["source"]]["geometry"])
        target = np.load(args.teacher_root / records[pair["target"]]["geometry"])
        source_rgb, crop = prepare_rgb(source["rgb"])
        target_rgb, _ = prepare_rgb(target["rgb"])
        depth, valid, intrinsics = transform_geometry(source["depth"], source["mask"].astype(np.float32), source["intrinsics"], crop)
        rotation, _ = cv2.Rodrigues(np.asarray(pair["rvec"], dtype=np.float32))
        transform = torch.eye(4).unsqueeze(0)
        transform[0, :3, :3] = torch.from_numpy(rotation)
        transform[0, :3, 3] = torch.tensor(pair["tvec"], dtype=torch.float32)
        with torch.inference_mode():
            source_latent_5d = vae.encode(source_rgb.to(torch.bfloat16), device=device, **tiler).to(device)
            target_latent_5d = vae.encode(target_rgb.to(torch.bfloat16), device=device, **tiler).to(device)
            if source_latent_5d.shape[2] != 1 or target_latent_5d.shape[2] != 1:
                raise RuntimeError("P2 is restricted to one spatial latent frame")
            source_latent = source_latent_5d[:, :, 0]
            target_latent = target_latent_5d[:, :, 0]
            decoded_copy = vae.decode(source_latent.unsqueeze(2), device=device, **tiler).float().to(device)
        for method in methods:
            aligned_depth, aligned_valid = align_depth_to_latent(depth, valid, (44, 80), method)
            points = points_from_depth(aligned_depth.to(device), intrinsics.to(device))
            warp = forward_splat_latent(source_latent, points, aligned_valid.to(device), intrinsics.to(device), transform.to(device))
            metrics = latent_reprojection_loss(warp, target_latent)
            with torch.inference_mode():
                decoded_warp = vae.decode(warp.latent.to(torch.bfloat16).unsqueeze(2), device=device, **tiler).float().to(device)
            target_rgb_device = target_rgb.to(device)
            copy_mse = (decoded_copy - target_rgb_device).square().mean()
            warp_mse = (decoded_warp - target_rgb_device).square().mean()
            rows.append({
                "pair": {key: pair[key] for key in ("source", "target", "scene", "inliers", "inlier_ratio", "median_reprojection_px")},
                "alignment": method,
                "warp_latent_l1": float(metrics["l1"]),
                "warp_latent_cosine_loss": float(metrics["cosine"]),
                "warp_latent_cosine_similarity": float(1.0 - metrics["cosine"]),
                "copy_latent_l1": float((source_latent - target_latent).abs().mean()),
                "copy_latent_l2": float((source_latent - target_latent).square().mean().sqrt()),
                "copy_latent_cosine_similarity": float(F.cosine_similarity(source_latent, target_latent, dim=1).mean()),
                "latent_coverage": float(metrics["coverage"]),
                "latent_hole_ratio": float(metrics["hole_ratio"]),
                "decoded_warp_psnr": float(-10 * torch.log10(warp_mse.clamp_min(1e-12))),
                "decoded_copy_psnr": float(-10 * torch.log10(copy_mse.clamp_min(1e-12))),
                "decoded_warp_global_ssim": global_ssim(decoded_warp, target_rgb_device),
                "decoded_copy_global_ssim": global_ssim(decoded_copy, target_rgb_device),
                "decoded_lpips": None,
                "lpips_status": "not measured: package unavailable in the experiment environment",
            })
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "stage": "P2/P3 estimated-pose screening", "pose_status": "estimated by MoGe-point-assisted SIFT + PnP-RANSAC; not ground truth", "repository_commit": "e2a44f3", "vae_checkpoint": str(args.vae_checkpoint.resolve()), "vae_sha256": sha256(args.vae_checkpoint), "methods": list(methods), "pair_count": len(pairs), "rows": rows}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"pair_count": len(pairs), "row_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

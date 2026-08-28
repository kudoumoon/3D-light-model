"""Export a small MoGe-3 teacher dataset for student distillation.

The output is intentionally written under runs/ by default.  Each sample keeps
the same geometry.npz contract used by the reprojection tools.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rgb(path: Path, max_size: int) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if max(rgb.shape[:2]) > max_size:
        scale = max_size / max(rgb.shape[:2])
        rgb = cv2.resize(
            rgb,
            (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return rgb


def depth_vis(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask & np.isfinite(depth) & (depth > 0)
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not valid.any():
        return out
    lo, hi = np.percentile(depth[valid], [2, 98])
    normalized = 1.0 - np.clip((depth - lo) / max(float(hi - lo), 1e-6), 0, 1)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def iter_images(root: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in exts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        default=ROOT / "third_party/Matrix-Game/Matrix-Game-2/demo_images",
    )
    parser.add_argument("--repo", type=Path, default=ROOT / "third_party/MoGe")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/moge-3-vitl/model.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/teacher_moge3_demo")
    parser.add_argument("--max-size", type=int, default=384)
    parser.add_argument("--num-tokens", type=int, default=1200)
    parser.add_argument("--refine-steps", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    sys.path.insert(0, str(args.repo.resolve()))
    model_class = importlib.import_module("moge.model.v3").MoGeModel
    model = model_class.from_pretrained(args.checkpoint).cuda().eval()

    images = iter_images(args.images)
    if not images:
        raise RuntimeError(f"No images found under {args.images}")

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, image_path in enumerate(images):
        rel = image_path.relative_to(args.images)
        sample_id = rel.with_suffix("").as_posix().replace("/", "__")
        sample_dir = args.output / sample_id
        geometry_path = sample_dir / "geometry.npz"
        metadata_path = sample_dir / "metadata.json"
        if geometry_path.exists() and metadata_path.exists() and not args.overwrite:
            records.append(json.loads(metadata_path.read_text(encoding="utf-8")))
            continue

        rgb = load_rgb(image_path, args.max_size)
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).cuda().float().div_(255)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.infer(
                image,
                num_tokens=args.num_tokens,
                use_fp16=True,
                force_projection=True,
                apply_mask=True,
                refine_steps=args.refine_steps,
                return_per_step=False,
            )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        arrays = {k: v.detach().cpu().numpy() for k, v in output.items() if torch.is_tensor(v)}
        arrays.setdefault("normal", np.empty((0,), dtype=np.float32))
        arrays.update(rgb=rgb, coordinate_convention=np.array("opencv_x_right_y_down_z_forward"))
        sample_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(geometry_path, **arrays)
        cv2.imwrite(str(sample_dir / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(
            str(sample_dir / "depth_vis.png"),
            cv2.cvtColor(depth_vis(arrays["depth"], arrays["mask"].astype(bool)), cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(str(sample_dir / "mask.png"), arrays["mask"].astype(np.uint8) * 255)

        mask = arrays["mask"].astype(bool)
        depth = arrays["depth"]
        valid = mask & np.isfinite(depth) & (depth > 0)
        record = {
            "sample_id": sample_id,
            "scene": rel.parts[0] if len(rel.parts) > 1 else "default",
            "source_image": rel.as_posix(),
            "geometry": geometry_path.relative_to(args.output).as_posix(),
            "height": int(rgb.shape[0]),
            "width": int(rgb.shape[1]),
            "image_sha256": sha256(image_path),
            "teacher_ms": round(elapsed_ms, 3),
            "valid_fraction": float(valid.mean()),
            "depth_median": float(np.median(depth[valid])) if valid.any() else None,
        }
        metadata_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        records.append(record)
        print(json.dumps({"done": index + 1, "total": len(images), **record}, ensure_ascii=False))

    manifest = {
        "dataset": "matrix_game_demo_moge3_teacher",
        "num_samples": len(records),
        "source_root": args.images.as_posix(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "max_size": args.max_size,
        "num_tokens": args.num_tokens,
        "refine_steps": args.refine_steps,
        "records": records,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": args.output.as_posix(), "num_samples": len(records)}, indent=2))


if __name__ == "__main__":
    main()


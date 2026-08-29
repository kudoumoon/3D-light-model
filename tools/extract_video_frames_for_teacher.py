"""Extract multi-crop video frames for MoGe-3 teacher export.

This creates an image tree consumable by ``export_moge3_teacher_dataset.py``.
Each crop is treated as a scene by the exporter because the first path component
is the scene name.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]


def crop_frame(frame, crop: str):
    h, w = frame.shape[:2]
    # Keep square-ish crops; export_moge3_teacher_dataset resizes max side later.
    side = min(h, w)
    if crop == "left":
        x0 = 0
    elif crop == "mid_left":
        x0 = round((w - side) * 0.25)
    elif crop == "center":
        x0 = round((w - side) * 0.50)
    elif crop == "mid_right":
        x0 = round((w - side) * 0.75)
    elif crop == "right":
        x0 = w - side
    else:
        raise ValueError(crop)
    y0 = max(0, (h - side) // 2)
    return frame[y0:y0 + side, x0:x0 + side]


def extract_video(video: Path, output: Path, prefix: str, frames_per_crop: int, stride: int, start: int, crops: list[str]) -> list[dict]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    rows = []
    frame_indices = [start + i * stride for i in range(frames_per_crop)]
    frame_indices = [idx for idx in frame_indices if idx < total]
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        for crop in crops:
            scene = f"{prefix}_{crop}"
            scene_dir = output / scene / "frames"
            scene_dir.mkdir(parents=True, exist_ok=True)
            out_path = scene_dir / f"frame_{frame_idx:06d}.png"
            crop_img = crop_frame(frame, crop)
            cv2.imwrite(str(out_path), crop_img)
            rows.append({
                "video": video.as_posix(),
                "scene": scene,
                "frame_index": frame_idx,
                "crop": crop,
                "path": out_path.relative_to(output).as_posix(),
            })
    cap.release()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/source_frames_matrixgame_v6_2k")
    parser.add_argument("--frames-per-crop", type=int, default=260)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    videos = [
        ("game2", ROOT / "third_party/Matrix-Game/Matrix-Game-2/assets/videos/matrix-game2.mp4"),
        ("game3", ROOT / "third_party/Matrix-Game/Matrix-Game-3/assets/videos/matrix-game3.mp4"),
    ]
    crops = ["left", "mid_left", "center", "mid_right", "right"]
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for prefix, video in videos:
        all_rows.extend(extract_video(video, args.output, prefix, args.frames_per_crop, args.stride, args.start, crops))
    manifest = {
        "output": args.output.as_posix(),
        "num_images": len(all_rows),
        "frames_per_crop": args.frames_per_crop,
        "stride": args.stride,
        "start": args.start,
        "crops": crops,
        "records": all_rows,
    }
    (args.output / "source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["output", "num_images", "frames_per_crop", "stride", "start", "crops"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

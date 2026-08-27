"""Extract a fixed gameplay tile from an official Matrix-Game demo montage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--x", type=int, required=True)
    parser.add_argument("--y", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    metadata_path = args.output / "metadata.json"
    previous = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
    crop_xywh = [args.x, args.y, args.width, args.height]
    if min(args.x, args.y) < 0 or min(args.width, args.height) <= 0:
        raise ValueError("invalid crop bounds")
    if previous and (previous["source_video"] != str(args.video.resolve()) or previous["crop_xywh"] != crop_xywh):
        raise ValueError("output contains another video/crop; choose a new output folder")

    capture = cv2.VideoCapture(str(args.video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not capture.isOpened() or fps <= 0:
        raise RuntimeError("video unavailable or invalid FPS")
    extracted = []
    for index in args.frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {index}")
        crop = image[args.y:args.y + args.height, args.x:args.x + args.width]
        if crop.shape[:2] != (args.height, args.width):
            raise ValueError(f"Crop out of bounds at frame {index}: {crop.shape}")
        path = args.output / f"frame_{index:04d}.png"
        cv2.imwrite(str(path), crop)
        extracted.append({"frame": index, "time_sec": index / fps, "path": str(path.resolve())})
    capture.release()
    merged = {entry["frame"]: entry for entry in (previous or {}).get("frames", [])}
    merged.update({entry["frame"]: entry for entry in extracted})
    metadata = {
        "source_video": str(args.video.resolve()),
        "fps": fps,
        "crop_xywh": crop_xywh,
        "frames": [merged[index] for index in sorted(merged)],
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

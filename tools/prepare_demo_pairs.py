"""Prepare the same bounded demo crops; does not run Matrix-Game generation."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "runs/demo")
    p.add_argument("--video", type=Path, default=ROOT / "third_party/Matrix-Game/Matrix-Game-2/assets/videos/matrix-game2.mp4")
    args = p.parse_args()
    if not args.video.is_file():
        raise SystemExit("Official demo video missing. Run tools/setup_dependencies.py first.")
    sources = [0, 15, 30, 45, 60, 75, 90]
    indices = sorted(set(sources + [s + d for s in sources for d in [3, 6, 8, 15, 30] if s + d <= 93]))
    for label, x in [("straight", 655), ("turning", 0)]:
        out = args.output / label
        run("extract_video_crop.py", "--video", args.video, "--output", out / "frames",
            "--x", x, "--y", 420, "--width", 654, "--height", 300, "--frames", *indices)
        for source in sources:
            run("geometry_provider.py", "--image", out / f"frames/frame_{source:04d}.png",
                "--output", out / f"geometry/frame_{source:04d}", "--max-size", 640,
                "--num-tokens", 1200, "--warmup", 1, "--repeat", 3)


if __name__ == "__main__":
    main()

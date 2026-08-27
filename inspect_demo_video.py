"""Extract a contact sheet and selected frames from an official demo video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.video))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    indices = [round(i * (count - 1) / max(1, args.samples - 1)) for i in range(args.samples)]
    images = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = capture.read()
        if not ok:
            continue
        cv2.imwrite(str(args.output / f"frame_{index:04d}.png"), bgr)
        images.append((cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), index))
    capture.release()

    columns = 4
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(16, 3.1 * rows), constrained_layout=True)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, (image, index) in zip(axes.ravel(), images):
        axis.imshow(image)
        axis.set_title(f"frame {index} / {index / fps:.2f}s")
        axis.axis("off")
    fig.savefig(args.output / "contact_sheet.png", dpi=160)
    plt.close(fig)
    print(f"frames={count}, fps={fps:.3f}, duration={count / fps:.3f}s")


if __name__ == "__main__":
    main()

"""Compare teacher/student geometry with CUDA forward reprojection.

The script samples validation records from a teacher manifest, finds matching
student predictions exported by ``evaluate_reprojection_student_warp.py``, runs
``reproject_torch.py`` for multiple virtual camera motions, and summarizes
coverage/speed gaps.  It intentionally records representative warped images so
paper-facing qualitative checks are reproducible.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_json(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
    return json.loads(proc.stdout)


def load_val_records(teacher_root: Path) -> list[dict]:
    manifest = json.loads((teacher_root / "manifest.json").read_text(encoding="utf-8"))
    by_scene: dict[str, list[dict]] = {}
    for record in manifest["records"]:
        by_scene.setdefault(record["scene"], []).append(record)
    selected: list[dict] = []
    for records in by_scene.values():
        records = sorted(records, key=lambda row: row["sample_id"])
        holdout = max(1, round(len(records) * 0.2))
        selected.extend(records[-holdout:])
    return selected


def pick_records(records: list[dict], per_scene: int) -> list[dict]:
    by_scene: dict[str, list[dict]] = {}
    for record in records:
        by_scene.setdefault(record["scene"], []).append(record)
    picked: list[dict] = []
    for scene in sorted(by_scene):
        rows = sorted(by_scene[scene], key=lambda row: row["sample_id"])
        if len(rows) <= per_scene:
            picked.extend(rows)
            continue
        if per_scene == 1:
            picked.append(rows[len(rows) // 2])
            continue
        positions = [round(i * (len(rows) - 1) / (per_scene - 1)) for i in range(per_scene)]
        picked.extend(rows[pos] for pos in positions)
    return picked


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def grouped(rows: list[dict], key: str) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return {
        name: {
            "num_cases": len(items),
            "student_coverage_mean": mean([r["student_coverage"] for r in items]),
            "teacher_coverage_mean": mean([r["teacher_coverage"] for r in items]),
            "coverage_gap_mean": mean([r["coverage_gap"] for r in items]),
            "coverage_gap_min": min(r["coverage_gap"] for r in items),
        }
        for name, items in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-scene", type=int, default=3)
    parser.add_argument("--splat-radius", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    records = pick_records(load_val_records(args.teacher), args.per_scene)
    motions = [
        {"name": "yaw_p5_fwd10", "yaw": 5.0, "forward": 0.10},
        {"name": "yaw_m5_fwd10", "yaw": -5.0, "forward": 0.10},
        {"name": "yaw_p10_fwd10", "yaw": 10.0, "forward": 0.10},
        {"name": "yaw_m10_fwd10", "yaw": -10.0, "forward": 0.10},
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for record in records:
        sample_id = record["sample_id"]
        teacher_geom = args.teacher / record["geometry"]
        student_geom = args.student_eval / "predictions" / sample_id / "geometry.npz"
        if not student_geom.is_file():
            raise FileNotFoundError(student_geom)
        for motion in motions:
            common = [
                sys.executable,
                "reproject_torch.py",
                "--yaw", str(motion["yaw"]),
                "--forward", str(motion["forward"]),
                "--splat-radius", str(args.splat_radius),
                "--warmup", str(args.warmup),
                "--repeat", str(args.repeat),
            ]
            teacher_out = args.output / sample_id / motion["name"] / "teacher"
            student_out = args.output / sample_id / motion["name"] / "student"
            teacher = run_json(common + ["--geometry", str(teacher_geom), "--output", str(teacher_out)])
            student = run_json(common + ["--geometry", str(student_geom), "--output", str(student_out)])
            rows.append({
                "sample_id": sample_id,
                "scene": record["scene"],
                "motion": motion["name"],
                "yaw": motion["yaw"],
                "forward": motion["forward"],
                "teacher_coverage": teacher["coverage_ratio"],
                "student_coverage": student["coverage_ratio"],
                "coverage_gap": student["coverage_ratio"] - teacher["coverage_ratio"],
                "teacher_reproject_ms": teacher["resident_gpu_ms_median"],
                "student_reproject_ms": student["resident_gpu_ms_median"],
            })
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    by_scene = grouped(rows, "scene")
    by_motion = grouped(rows, "motion")
    yaw10_rows = [r for r in rows if abs(float(r["yaw"])) >= 10.0]
    worst_cases = sorted(rows, key=lambda row: row["coverage_gap"])[:10]
    worst_scene = min(by_scene.items(), key=lambda item: item[1]["coverage_gap_mean"]) if by_scene else None
    summary = {
        "teacher": args.teacher.as_posix(),
        "student_eval": args.student_eval.as_posix(),
        "num_samples": len(records),
        "num_motions": len(motions),
        "num_cases": len(rows),
        "student_coverage_mean": mean([r["student_coverage"] for r in rows]),
        "teacher_coverage_mean": mean([r["teacher_coverage"] for r in rows]),
        "coverage_gap_mean": mean([r["coverage_gap"] for r in rows]),
        "coverage_gap_min": min([r["coverage_gap"] for r in rows]) if rows else 0.0,
        "yaw10_coverage_gap_mean": mean([r["coverage_gap"] for r in yaw10_rows]),
        "yaw10_student_coverage_mean": mean([r["student_coverage"] for r in yaw10_rows]),
        "yaw10_teacher_coverage_mean": mean([r["teacher_coverage"] for r in yaw10_rows]),
        "worst_scene": {
            "scene": worst_scene[0],
            **worst_scene[1],
        } if worst_scene else None,
        "student_reproject_ms_median_mean": mean([r["student_reproject_ms"] for r in rows]),
        "teacher_reproject_ms_median_mean": mean([r["teacher_reproject_ms"] for r in rows]),
        "by_scene": by_scene,
        "by_motion": by_motion,
        "worst_cases": worst_cases,
        "motions": motions,
        "rows": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

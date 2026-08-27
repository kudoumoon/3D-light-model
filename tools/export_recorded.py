"""Curate pre-existing local experiment records without modifying their sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--release", type=Path, required=True)
    args = p.parse_args()
    src, dst = args.source.resolve(), args.release.resolve()
    sys.path.insert(0, str(dst))
    from quality_metrics import tile_candidate_oracle_fraction
    selected = []
    for name in ["gta_0000_bench", "temple_0000_bench", "universal_0003_bench"]:
        selected.append(f"results/{name}/metadata.json")
    for scene in ["gta_0000", "temple", "universal"]:
        for yaw in [2, 5, 10]:
            selected.append(f"results/{scene}_yaw{yaw}/reprojection.json")
        selected.append(f"results/{scene}_yaw5_cuda/reprojection_cuda.json")
    selected += ["results/gta_0000_right/reprojection.json"]
    for size in ["active_dit_bench", "active_dit_bench_large"]:
        selected.append(f"results/{size}/active_dit_benchmark.json")
    quality_runs = ["matrix_game2_gta_target_eval", "matrix_game2_gta_turn_target_eval"]
    quality_runs += [x + suffix for suffix in ["_k8", "_long"] for x in quality_runs[:2]]
    selected += [f"results/{name}/summary.json" for name in quality_runs]
    selected += ["results/speed_gain/speed_gain.json", "results/target_quality_summary/target_quality_summary.json"]
    manifest = []
    def sanitize(value):
        if isinstance(value, dict):
            return {k: sanitize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [sanitize(v) for v in value]
        if isinstance(value, str):
            normalized = value.replace("\\", "/")
            prefix = src.as_posix() + "/"
            if normalized.lower().startswith(prefix.lower()):
                return normalized[len(prefix):]
        return value
    for relative in selected:
        source = src / relative
        out = dst / "results/recorded" / Path(relative).relative_to("results")
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = sanitize(json.loads(source.read_text(encoding="utf-8")))
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        manifest.append({"source_relative": relative, "published": out.relative_to(dst).as_posix(),
                         "source_sha256": sha(source), "published_sha256": sha(out),
                         "transformation": "JSON formatting and machine-local path relativization only; numeric values unchanged"})
    rows, failures = [], []
    for run in quality_runs:
        data = json.loads((src / "results" / run / "summary.json").read_text(encoding="utf-8"))
        track = "turning" if "turn_" in run else "straight"
        group = "gap8" if run.endswith("_k8") else "long" if run.endswith("_long") else "short"
        failures.extend({"run": run, **f} for f in data["failures"])
        for original in data["rows"]:
            row = dict(original)
            pair = f"{row['source_frame']:04d}_to_{row['target_frame']:04d}"
            base = src / "results" / run / pair
            source_bgr = cv2.imread(str(base / "source.png"))
            target_bgr = cv2.imread(str(base / "target.png"))
            mask = cv2.imread(str(base / "mask.png"), cv2.IMREAD_GRAYSCALE) > 0
            error = np.load(base / "pixel_mae.npy", allow_pickle=False)
            copy_error = np.abs(source_bgr.astype(np.float32) - target_bgr.astype(np.float32)).mean(-1) / 255
            row["tile_candidate_oracle_pass_fraction"] = {
                str(t): tile_candidate_oracle_fraction(copy_error, error, mask, t / 255, row["tile_size"])
                for t in [10, 20, 30, 40]
            }
            row.update(track=track, group=group, recorded_run=run,
                       layout_transition_risk=row["target_frame"] >= 90,
                       audit_added_fields_date="2026-08-28")
            rows.append(row)
    audit = {"source": "Six recorded target-evaluation summaries and their saved RGB/mask/error arrays",
             "warning": "Targets and target-assisted PnP are offline evaluation only. Legacy best fields are pixel-min oracles, not tile selectors.",
             "rows": rows, "failures": failures}
    (dst / "results/audited/quality_pairs.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8", newline="\n")
    figures = {"gta_geometry_montage.png": "local_geometry_figures/gta_geometry_montage.png",
               "temple_geometry_montage.png": "local_geometry_figures/temple_geometry_montage.png",
               "universal_geometry_montage.png": "local_geometry_figures/universal_geometry_montage.png",
               "gta_pointcloud.png": "local_geometry_figures/gta_pointcloud.png",
               "straight_0030_to_0038_montage.png": "target_quality_figures/straight_0030_to_0038_montage.png",
               "transition_0090_to_0093_montage.png": "target_quality_figures/turn_0090_to_0093_montage.png"}
    original_outputs = src.parent.parent / "outputs"
    for name, relative in figures.items():
        origin = original_outputs / relative
        out = dst / "assets/examples" / name
        shutil.copy2(origin, out)
        manifest.append({"source_relative": "outputs/" + relative, "published": out.relative_to(dst).as_posix(),
                         "source_sha256": sha(origin), "published_sha256": sha(out),
                         "transformation": "unchanged; legacy titles may require the adjacent caveats"})
    provenance = {"experiment_date": "2026-08-16", "export_date": "2026-08-28",
                  "upstream_commits": {"microsoft/MoGe": "925b8ed835a7a9cdb7578ba15c658a0afc969030",
                                       "SkyworkAI/Matrix-Game": "71c3cd7f741311f8100f6cf9cde942b6c1378d11"},
                  "files": manifest}
    (dst / "results/provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"recorded_json": len(selected), "valid_pairs": len(rows), "failed_pairs": len(failures),
                      "layout_risk_pairs": sum(r["layout_transition_risk"] for r in rows)}, indent=2))


if __name__ == "__main__":
    main()

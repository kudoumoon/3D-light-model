#!/usr/bin/env python3
"""在独立验证场景上校准现有 Geometry Head 的 valid 阈值；不训练新参数。"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latent_geometry_head_v2 import LatentGeometryHeadV2

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-scenes", nargs="+", default=("tum_xyz_val", "tum_rpy_val"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one checked idle GPU")
    if args.output.exists(): raise FileExistsError(args.output)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    indices = [i for i, scene in enumerate(cache["scenes"]) if scene in set(args.validation_scenes)]
    if not indices: raise RuntimeError("validation split is empty")
    device = torch.device("cuda:0")
    started = time.perf_counter()
    model = LatentGeometryHeadV2().to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"])
    probabilities, labels = [], []
    with torch.inference_mode():
        for start in range(0, len(indices), args.batch_size):
            batch = indices[start:start + args.batch_size]
            latent = cache["latent"][batch].to(device)
            intrinsics = cache["intrinsics"][batch].to(device)
            probabilities.append(model(latent, intrinsics).latent_valid.cpu())
            labels.append(cache["valid"][batch].bool())
    probability, label = torch.cat(probabilities), torch.cat(labels)
    rows = []
    for threshold in np.linspace(0.05, 0.95, 19):
        prediction = probability >= threshold
        tp = (prediction & label).sum().item(); fp = (prediction & ~label).sum().item(); fn = (~prediction & label).sum().item()
        rows.append({"threshold": float(threshold), "iou": tp/max(1,tp+fp+fn), "precision": tp/max(1,tp+fp), "recall": tp/max(1,tp+fn), "predicted_valid_rate": float(prediction.float().mean())})
    selected = max(rows, key=lambda row: (row["iou"], row["precision"]))
    args.output.mkdir(parents=True)
    report = {"schema_version":1,"stage":"zero-parameter latent-valid threshold calibration","repository_commit":subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip(),"validation_scenes":args.validation_scenes,"validation_frames":len(indices),"selection_metric":"source-depth valid IoU","selected":selected,"sweep":rows,"runtime":{"seconds":time.perf_counter()-started,"gpu":torch.cuda.get_device_name(0),"peak_allocated_bytes":torch.cuda.max_memory_allocated()},"new_trainable_parameters":0,"public_shape_changed":False}
    (args.output/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(selected,indent=2)); print("实验已完成")
if __name__ == "__main__": main()

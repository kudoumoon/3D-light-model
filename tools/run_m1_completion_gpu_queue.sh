#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TEACHER="runs/teacher_moge3_video384_v6_2k"
BASE_CKPT="runs/reprojection_student/student_video384_tvod_v7_2k_occ075_width64_lr15/checkpoints/best.pt"
HARD_SUMMARY="results/recorded/m1_v10_frozen_geometry_motion_confidence/v10_reproj_summary.json"
PACK_DIR="results/recorded/m1_v11_completion_pack"
LOG_DIR="${PACK_DIR}/logs"
PYTHON_BIN="${ROOT}/.venv/bin/python"
mkdir -p "$LOG_DIR"

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); if ($2 < 1000 && $3 < 5) print $1}'
}

stable_free_gpus() {
  local first second gpu
  mapfile -t first < <(free_gpus)
  sleep 10
  mapfile -t second < <(free_gpus)
  for gpu in "${first[@]}"; do
    printf '%s\n' "${second[@]}" | grep -qx "$gpu" && printf '%s\n' "$gpu"
  done
}

wait_for_gpus() {
  local min_count="$1"
  local gpus
  while true; do
    mapfile -t gpus < <(stable_free_gpus)
    if [ "${#gpus[@]}" -ge "$min_count" ]; then
      printf '%s\n' "${gpus[@]}"
      return 0
    fi
    date '+[%F %T] no free gpu; polling again in 45s' | tee -a "${LOG_DIR}/gpu_queue.log" >&2
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tee -a "${LOG_DIR}/gpu_queue.log" >&2
    sleep 45
  done
}

run_variant() {
  local gpu="$1"
  local name="$2"
  local lr="$3"
  local occupancy="$4"
  local deficit="$5"
  local edge="$6"
  local strength="$7"
  local cap="$8"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/train_reprojection_student_warp.py \
    --teacher "$TEACHER" \
    --output runs/reprojection_student \
    --name "$name" \
    --epochs 80 \
    --width 64 \
    --lr "$lr" \
    --init-checkpoint "$BASE_CKPT" \
    --hardcase-summary "$HARD_SUMMARY" \
    --hardcase-strength "$strength" \
    --hardcase-cap "$cap" \
    --occupancy-weight "$occupancy" \
    --coverage-deficit-weight "$deficit" \
    --depth-edge-point-weight "$edge" \
    --device cuda \
    > "${LOG_DIR}/${name}.train.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/evaluate_reprojection_student_warp.py \
    --teacher "$TEACHER" \
    --checkpoint "runs/reprojection_student/${name}/checkpoints/best.pt" \
    --output "runs/reprojection_student/${name}_eval" \
    --split val \
    --warmup 10 \
    --repeat 30 \
    --device cuda \
    > "${LOG_DIR}/${name}.eval.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/compare_reprojection_models.py \
    --teacher "$TEACHER" \
    --student-eval "runs/reprojection_student/${name}_eval" \
    --output "runs/reprojection_student/${name}_reprojection_multi" \
    --per-scene 3 \
    --warmup 5 \
    --repeat 20 \
    --device cuda \
    > "${LOG_DIR}/${name}.reprojection.log" 2>&1
}


run_cross_domain_eval() {
  local gpu="$1"
  local checkpoint="$2"
  local tag="$3"
  local datasets=(
    "matrixgame2_demo third_party/Matrix-Game/Matrix-Game-2/demo_images runs/teacher_moge3_cross_matrixgame2_demo_384"
    "matrixgame3_demo third_party/Matrix-Game/Matrix-Game-3/demo_images runs/teacher_moge3_cross_matrixgame3_demo_384"
    "moge_examples third_party/MoGe/example_images runs/teacher_moge3_cross_moge_examples_384"
  )
  for spec in "${datasets[@]}"; do
    read -r domain images teacher_out <<< "$spec"
    if [ ! -f "${teacher_out}/manifest.json" ]; then
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/export_moge3_teacher_dataset.py \
        --images "$images" \
        --output "$teacher_out" \
        --max-size 384 \
        --num-tokens 1200 \
        --refine-steps 0 \
        > "${LOG_DIR}/${tag}.${domain}.teacher.log" 2>&1
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/evaluate_reprojection_student_warp.py \
      --teacher "$teacher_out" \
      --checkpoint "$checkpoint" \
      --output "runs/reprojection_student/${tag}_${domain}_eval" \
      --split all \
      --warmup 10 \
      --repeat 30 \
      --device cuda \
      > "${LOG_DIR}/${tag}.${domain}.eval.log" 2>&1
  done
}

VARIANTS=(
  "student_video384_tvod_v11b_hardcoverage_cons_lr8e5 8e-5 0.25 0.50 0.10 6.0 2.0"
  "student_video384_tvod_v11b_hardcoverage_mid_lr8e5 8e-5 0.50 1.00 0.25 8.0 2.2"
  "student_video384_tvod_v11b_hardcoverage_aggr_lr6e5 6e-5 0.75 1.50 0.35 10.0 2.5"
)

pending=("${VARIANTS[@]}")
while [ "${#pending[@]}" -gt 0 ]; do
  mapfile -t GPUS < <(wait_for_gpus 1)
  batch_count="${#GPUS[@]}"
  if [ "$batch_count" -gt "${#pending[@]}" ]; then
    batch_count="${#pending[@]}"
  fi
  pids=()
  launched_names=()
  for ((i=0; i<batch_count; i++)); do
    read -r name lr occupancy deficit edge strength cap <<< "${pending[$i]}"
    date '+[%F %T] launching hard-case candidate' | tee -a "${LOG_DIR}/gpu_queue.log"
    printf 'gpu=%s name=%s\n' "${GPUS[$i]}" "$name" | tee -a "${LOG_DIR}/gpu_queue.log"
    run_variant "${GPUS[$i]}" "$name" "$lr" "$occupancy" "$deficit" "$edge" "$strength" "$cap" &
    pids+=("$!")
    launched_names+=("$name")
  done
  failures=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failures=$((failures + 1))
    fi
  done
  if [ "$failures" -gt 0 ]; then
    date '+[%F %T] at least one hard-case candidate failed; stopping queue for inspection' | tee -a "${LOG_DIR}/gpu_queue.log"
    exit 1
  fi
  pending=("${pending[@]:$batch_count}")
done

"$PYTHON_BIN" tools/summarize_m1_completion_experiments.py --output "$PACK_DIR" > "${LOG_DIR}/completion_pack_refresh.log" 2>&1
SELECTED_CKPT="$BASE_CKPT"
if [ -f "${PACK_DIR}/m1_completion_summary.json" ]; then
  SELECTED_RUN=$("$PYTHON_BIN" - <<'PYSEL'
import json
from pathlib import Path
path = Path('results/recorded/m1_v11_completion_pack/m1_completion_summary.json')
data = json.loads(path.read_text(encoding='utf-8'))
row = data.get('hardcase_candidates', {}).get('selected_candidate')
print(row['run_name'] if row else '')
PYSEL
)
  if [ -n "$SELECTED_RUN" ]; then
    SELECTED_CKPT="runs/reprojection_student/${SELECTED_RUN}/checkpoints/best.pt"
  fi
fi
date '+[%F %T] selected checkpoint for cross-domain' | tee -a "${LOG_DIR}/gpu_queue.log"
printf 'checkpoint=%s\n' "$SELECTED_CKPT" | tee -a "${LOG_DIR}/gpu_queue.log"
mapfile -t CROSS_GPUS < <(wait_for_gpus 1)
run_cross_domain_eval "${CROSS_GPUS[0]}" "$SELECTED_CKPT" "m1_selected_crossdomain"
"$PYTHON_BIN" tools/summarize_m1_completion_experiments.py --output "$PACK_DIR" > "${LOG_DIR}/completion_pack_refresh_after_crossdomain.log" 2>&1

#!/bin/bash
# =============================================
# Automated 3DGS Pipeline (Train → JSON → Render → Eval)
# Iterates over all scenes in the ROOT folder
# =============================================

set -euo pipefail

# Usage: bash script/eval_4_4.sh [ROOT] [OUTPUT_ROOT]
#   ROOT        : dataset folder containing the scenes (default: data/lerf_mask)
#   OUTPUT_ROOT : trained models from script/train_4_2.sh (default: output/lerf_mask)

# ===== Basic Settings =====
ROOT="${1:-data/lerf_mask}"
OUTPUT_ROOT="${2:-output/lerf_mask}"
PLY_ITER=30000
NUM_VIEWS=20                # 20 azimuths x 10 elevations = 200 object-centric views
IMG_W=986
IMG_H=728
KERNEL_SIZE=1.0

# ===== Evaluation Settings =====
EVAL_SCRIPT="object_centric_evaluation/noRef_metrics.py"
EVAL_CSV="${OUTPUT_ROOT}/all_scores_summary_met3r.csv"

# Per-metric skip flags (set to true to skip that metric)
SKIP_CLIP_IQA=false
SKIP_MUSIQ=false

SKIP_MET3R=false
MET3R_STRIDES="1"      # Multiple strides for (i, i+K) pairs
MET3R_BATCH=24
MET3R_IMG_SIZE=256
MET3R_BACKBONE="mast3r"    # mast3r / dust3r / raft
MET3R_FEAT="dino16"        # dino16 / dinov2 / maskclip / vit / clip / resnet50
MET3R_UPSAMPLER="featup"   # featup / nearest / bilinear / bicubic
MET3R_DISTANCE="cosine"    # cosine / lpips / rmse / psnr / mse / ssim

mkdir -p "${OUTPUT_ROOT}"

# Scripts under object_centric_evaluation/ import repo-root modules (scene, utils, ...).
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

# =============================================
# ===== SCENE LOOP =====
# =============================================
for SCENE_PATH in "${ROOT}"/*; do
    if [ -d "$SCENE_PATH" ]; then
        echo "====================================="
        echo "Processing scene: $(basename "$SCENE_PATH")"
        echo "====================================="
        SCENE_NAME=$(basename "$SCENE_PATH")
        OBJECT_NUMBER=${SCENE_NAME##*_}

        OUTPUT_PATH="${OUTPUT_ROOT}/${SCENE_NAME}"

        PLY_PATH=$(ls -td "${OUTPUT_PATH}"/point_cloud/iteration_${PLY_ITER}/point_cloud.ply 2>/dev/null | head -n 1 || true)
        JSON_PATH="${OUTPUT_PATH}/transforms_test.json"

        if [ -z "${PLY_PATH}" ] || [ ! -f "${PLY_PATH}" ]; then
            echo "  [WARN] PLY not found: ${OUTPUT_PATH}/point_cloud/iteration_${PLY_ITER}/point_cloud.ply"
            echo "  [WARN] Skipping this scene."
            echo ""
            continue
        fi

        ITER_DIR_NAME=$(basename "$(dirname "${PLY_PATH}")")
        ITER_NUM_OURS=$(echo "${ITER_DIR_NAME}" | grep -oE '[0-9]+' || true)

        if [ -z "${ITER_NUM_OURS}" ]; then
            echo "  [WARN] Failed to parse iteration number: ${ITER_DIR_NAME}"
            echo "  [WARN] Skipping this scene."
            echo ""
            continue
        fi


        echo "====================================="
        echo "▶ [1] Generating JSON for ${SCENE_NAME}"
        echo "====================================="
        python object_centric_evaluation/nerf_dir_camera.py \
            --ply "${PLY_PATH}" \
            --out "${JSON_PATH}" \
            --images_dir "${SCENE_PATH}/images" \
            --prefix frame_ --ext jpg --pad 5 --start 1 \
            --num_views ${NUM_VIEWS} \
            --img_w ${IMG_W} --img_h ${IMG_H} \
            --nerf_negz --absolute --store_no_ext --roll180 \
            --elev_start_deg -80 --elev_end_deg 0 --elev_steps 10

        echo "OBJECT_NUMBER: ${OBJECT_NUMBER}"
        echo "====================================="
        echo "▶ [3] Rendering: ${SCENE_NAME}"
        echo "====================================="
        python object_centric_evaluation/render_object.py \
            -m "${OUTPUT_PATH}" \
            --camera_json "${JSON_PATH}" \
            --iteration ${ITER_NUM_OURS} \
            --images_ext .jpg \
            --out_name hemisphere_render

        # =============================================
        # ▶ [4] Evaluation
        # =============================================
        echo "====================================="
        echo "▶ [4] Evaluating: ${SCENE_NAME}"
        echo "====================================="

        # Auto-search for hemisphere_render folder
        RENDER_DIR="${OUTPUT_PATH}/hemisphere_render"
        if [ ! -d "${RENDER_DIR}" ]; then
            RENDER_DIR=$(find "${OUTPUT_PATH}" -maxdepth 3 -type d -name "hemisphere_render" 2>/dev/null | head -n 1 || true)
        fi

        SCORE_JSON="${OUTPUT_PATH}/scores_${SCENE_NAME}.json"

        if [ -z "${RENDER_DIR}" ] || [ ! -d "${RENDER_DIR}" ]; then
            echo "  [WARN] hemisphere_render folder not found. Skipping evaluation."
        else
            # Combine skip flags
            EXTRA_FLAGS=""
            [ "${SKIP_CLIP_IQA}" = true ] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip_clip_iqa"
            [ "${SKIP_MUSIQ}"    = true ] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip_musiq"

            # ✅ met3r
            [ "${SKIP_MET3R}"    = true ] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip_met3r"
            if [ "${SKIP_MET3R}" != true ]; then
                EXTRA_FLAGS="${EXTRA_FLAGS} --met3r_strides ${MET3R_STRIDES}"
                EXTRA_FLAGS="${EXTRA_FLAGS} --met3r_batch ${MET3R_BATCH}"
                EXTRA_FLAGS="${EXTRA_FLAGS} --met3r_img_size ${MET3R_IMG_SIZE}"
                EXTRA_FLAGS="${EXTRA_FLAGS} --met3r_backbone ${MET3R_BACKBONE}"
                EXTRA_FLAGS="${EXTRA_FLAGS} --met3r_feat ${MET3R_FEAT}"
                EXTRA_FLAGS="${EXTRA_FLAGS} --met3r_upsampler ${MET3R_UPSAMPLER}"
                EXTRA_FLAGS="${EXTRA_FLAGS} --met3r_distance ${MET3R_DISTANCE}"
            fi

            python "${EVAL_SCRIPT}" \
                --render_dir  "${RENDER_DIR}" \
                --out_json    "${SCORE_JSON}" \
                --out_csv     "${EVAL_CSV}" \
                --scene_name  "${SCENE_NAME}" \
                ${EXTRA_FLAGS}

            echo "  JSON saved : ${SCORE_JSON}"
            echo "  CSV  appended : ${EVAL_CSV}"
        fi

        echo "====================================="
        echo "Done: ${SCENE_NAME} pipeline complete"
        echo "Output: ${OUTPUT_PATH}"
        echo "====================================="
        echo ""
    fi
done

echo "All scenes completed!"

# =============================================
# ▶ [5] Aggregate CSV → Append AVERAGE row and print final summary
# =============================================
echo "====================================="
echo "▶ [5] Aggregating all scores..."
echo "====================================="

export _EVAL_CSV="${EVAL_CSV}"

python3 - <<'PYEOF'
import os, csv, sys
import numpy as np

csv_path = os.environ.get("_EVAL_CSV", "")
if not csv_path or not os.path.exists(csv_path):
    print(f"[summary] CSV file not found: {csv_path}")
    sys.exit(0)

# Read existing data (excluding AVERAGE rows)
rows = []
with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if not row.get("scene", "").startswith("AVERAGE"):
            rows.append(row)

if not rows:
    print("[summary] No data found")
    sys.exit(0)

METRICS = {
    "clip_iqa_mean" : ("↑", "0~1"),
    "musiq_mean"    : ("↑", "0~100"),
    "met3r_mean"    : ("↓", "0~1"),
}

def col_mean(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return round(float(np.mean(vals)), 6) if vals else ""

avg_row = {
    "scene"        : f"AVERAGE ({len(rows)} scenes)",
    "num_rendered" : "",
}
for m in METRICS:
    avg_row[m] = col_mean(rows, m)

# Overwrite CSV (data rows + AVERAGE footer)
fieldnames = ["scene", "num_rendered",
              "clip_iqa_mean", "musiq_mean", "met3r_mean"]
# Write to a temp file first so a failure never truncates the existing CSV;
# columns not in `fieldnames` (e.g. from an older CSV format) are dropped.
tmp_path = csv_path + ".tmp"
with open(tmp_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    writer.writerow(avg_row)
os.replace(tmp_path, csv_path)

# Console output
print(f"\n{'─'*55}")
print(f"  Total scenes : {len(rows)}")
print(f"{'─'*55}")
for m, (arrow, scale) in METRICS.items():
    v = avg_row[m]
    label = {"clip_iqa_mean": "CLIP-IQA+", "musiq_mean": "MUSIQ", "met3r_mean": "MEt3R"}[m]
    if v != "":
        print(f"  {label:<12} {arrow}  avg = {float(v):.4f}   ({scale})")
    else:
        print(f"  {label:<12}     skip")
print(f"{'─'*55}")
print(f"  CSV saved : {csv_path}")
print(f"{'─'*55}\n")
PYEOF

#!/bin/bash
# =============================================
# 3DGS Training → Rendering → Metrics pipeline
# for all scenes under $ROOT
# =============================================

SCENE_NAME=""
ROOT=""
OUTPUT_ROOT="."
CSV_FILE=""

#export CUDA_VISIBLE_DEVICES=0

for SCENE_PATH in "$ROOT"/*; do
    if [ -d "$SCENE_PATH" ]; then
        SCENE=$(basename "$SCENE_PATH")

        IMG_DIR="$SCENE_PATH/images"
        MASK_DIR="$SCENE_PATH/masks"
        ORI_DIR="$SCENE_PATH/images_ori"
        SCENE_NAME="${SCENE%%_*}"
        OUT_DIR="$OUTPUT_ROOT/${SCENE_NAME}"
        

        echo "Evaluating metrics: $SCENE"
        python metrics_object_mIoU.py -m "$OUT_DIR" -mask "$MASK_DIR" | tee metrics_tmp.log

        SSIM=$(grep "SSIM" metrics_tmp.log | awk '{print $3}')
        PSNR=$(grep "PSNR" metrics_tmp.log | awk '{print $3}')
        LPIPS=$(grep "LPIPS" metrics_tmp.log | awk '{print $2}')
        mIoU=$(grep "mIoU" metrics_tmp.log | awk '{print $2}')

    echo "Finished: $SCENE"


    fi
done

echo "All scenes processed."

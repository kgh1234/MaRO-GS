#!/bin/bash
# =============================================
# 3DGS Training → Rendering → Metrics pipeline
# for all scenes under $ROOT
#
# Usage: bash script/train_4_2.sh <SCENE_NAME>
#   e.g. bash script/train_4_2.sh lerf_mask
# =============================================

SCENE_NAME="${1:-lerf_mask}"
ROOT="data/${SCENE_NAME}"
OUTPUT_ROOT="output/${SCENE_NAME}"
CSV_FILE="$OUTPUT_ROOT/${SCENE_NAME}_metrics.csv"


if [ ! -d "$ROOT" ]; then
    echo "Dataset folder not found: $ROOT"
    echo "Usage: bash script/train_4_2.sh <SCENE_NAME>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0

mkdir -p log "$OUTPUT_ROOT"

for SCENE_PATH in "$ROOT"/*; do
    if [ -d "$SCENE_PATH" ]; then
        SCENE=$(basename "$SCENE_PATH")

        IMG_DIR="$SCENE_PATH/images"
        MASK_DIR="$SCENE_PATH/mask"
        ORI_DIR="$SCENE_PATH/images_ori"
        OUT_DIR="$OUTPUT_ROOT/${SCENE}"
        FPS=""; GAUSSIAN_COUNT=""


        echo "====================================="
        echo "Processing scene: $SCENE"
        echo "====================================="

        echo " Training..."
        
        TRAIN_START=$(date +%s)
        LOGFILE="log/vram_${SCENE}.log"
        nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader -l 2 > "$LOGFILE" &
        VRAM_PID=$!

        python train.py -s "$SCENE_PATH" -m "$OUT_DIR" --mask_dir "$MASK_DIR" --prune_ratio 1.0 --geometric_filtering True --region_filtering True --gaussian_merge True --eval --port 2950

        TRAIN_END=$(date +%s)
        TRAIN_TIME=$((TRAIN_END - TRAIN_START))
        echo "Training time: ${TRAIN_TIME}s"

        kill $VRAM_PID 2>/dev/null
        VRAM_MAX=$(awk 'BEGIN{max=0}{if($1>max)max=$1}END{print max}' "$LOGFILE")
        rm -f "$LOGFILE"

        echo "Rendering: $SCENE"
 
        RENDER_START=$(date +%s)
        python render_with_mask.py -m "$OUT_DIR"
        RENDER_END=$(date +%s)
        RENDER_TIME=$((RENDER_END - RENDER_START))
        echo "Rendering time: ${RENDER_TIME}s"


        python render_FPS.py -m "$OUT_DIR" | tee log/fps_tmp.log
        FPS=$(grep -oP 'FPS\s*:\s*\K[0-9.e+-]+' log/fps_tmp.log)

        echo "FPS: $FPS"

        echo "Evaluating metrics: $SCENE"
        python metrics_object_mIoU.py -m "$OUT_DIR" --mask_dir "$MASK_DIR" | tee log/metrics_tmp.log


        POINT_CLOUD_DIR="$OUT_DIR/point_cloud"
        if [ -d "$POINT_CLOUD_DIR" ]; then
            LATEST_ITER_DIR=$(ls -d "$POINT_CLOUD_DIR"/iteration_* 2>/dev/null | sort -V | tail -n 1)
            if [ -n "$LATEST_ITER_DIR" ]; then
                PLY_PATH="$LATEST_ITER_DIR/point_cloud.ply"
                if [ -f "$PLY_PATH" ]; then
                    GAUSSIAN_COUNT=$(grep -a -m1 "element vertex" "$PLY_PATH" | awk '{print $3}')
                    echo "Gaussian: $GAUSSIAN_COUNT (from $(basename "$LATEST_ITER_DIR"))"
                else
                    echo "Gaussian: PLY not found in $(basename "$LATEST_ITER_DIR")"
                fi
            else
                echo "Gaussian: No iteration_* folder found under $POINT_CLOUD_DIR"
            fi
        else
            echo "Gaussian: point_cloud folder not found in $OUT_DIR"
        fi

        SSIM=$(grep "SSIM" log/metrics_tmp.log | awk '{print $3}')
        PSNR=$(grep "PSNR" log/metrics_tmp.log | awk '{print $3}')
        LPIPS=$(grep -oP 'LPIPS\s*:\s*\K[0-9.e+-]+' log/metrics_tmp.log)
        MIOU=$(grep "mIoU" log/metrics_tmp.log | awk '{print $3}')

        SSIM=${SSIM:-0}
        PSNR=${PSNR:-0}
        LPIPS=${LPIPS:-0}
        MIOU=${MIOU:-0}


        if [ ! -f "$CSV_FILE" ]; then
            echo "scene,SSIM,PSNR,LPIPS,mIoU,train_time_s,render_time_s,FPS,max_vram_MiB,num_gaussians" > "$CSV_FILE"
        fi
        echo "$SCENE,$SSIM,$PSNR,$LPIPS,$MIOU,$TRAIN_TIME,$RENDER_TIME,$FPS,$VRAM_MAX,$GAUSSIAN_COUNT" >> "$CSV_FILE"

        echo "Metrics for $SCENE appended to $CSV_FILE"
        echo "Finished: $SCENE"
        echo

    fi
done

echo "All scenes processed."

#!/bin/bash
# =============================================
# 3DGS Training → Rendering → Metrics pipeline
# for all scenes under $ROOT
# =============================================

SCENE_NAME="lerf/figurines"
ROOT="data/$SCENE_NAME"
OUTPUT_ROOT="output/$SCENE_NAME"
CSV_FILE="output/${SCENE_NAME}_metrics.csv"
DATASET_NAME=$(basename "$SCENE_NAME")
EVAL_SUMMARY="log/eval_${DATASET_NAME}_summary.csv"
OBJECT_LIST="object_green_apple,object_green_toy_chair,object_old_camera,object_porcelain_hand,object_red_apple,object_red_toy_chair,object_rubber_duck_with_red_hat"

#export CUDA_VISIBLE_DEVICES=0

IFS=',' read -r -a OBJECTS <<< "$OBJECT_LIST"
mkdir -p log
echo "object,miou,biou" > "$EVAL_SUMMARY"

for OBJECT in "${OBJECTS[@]}"; do
    echo "Processing object: $ROOT/$OBJECT"

    if [ -d "$ROOT" ]; then
        SCENE=$(basename "$SCENE_NAME")

        IMG_DIR="$ROOT/images"
        MASK_DIR="$ROOT/$OBJECT"
        #ORI_DIR="$SCENE_PATH/images_ori"
        OUT_DIR="$OUTPUT_ROOT/$OBJECT"
        
        echo "Processing scene: $ROOT"


        echo "====================================="
        echo "Processing scene: $SCENE"
        echo "====================================="

        echo " Training..."
        mkdir -p log
        
        TRAIN_START=$(date +%s)
        LOGFILE="log/vram_${SCENE}.log"
        nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader -l 2 > "$LOGFILE" &
        VRAM_PID=$!
#  
        python train.py -s "$ROOT" -m "$OUT_DIR" --mask_dir "$MASK_DIR" --prune_ratio 1.0 --train_split

        TRAIN_END=$(date +%s)
        TRAIN_TIME=$((TRAIN_END - TRAIN_START))
        echo "Training time: ${TRAIN_TIME}s"

        kill $VRAM_PID 2>/dev/null
        VRAM_MAX=$(awk 'BEGIN{max=0}{if($1>max)max=$1}END{print max}' "$LOGFILE")
        rm -f "$LOGFILE"

        # echo "Rendering: $SCENE"
        python render_lerf_mask.py -m "$OUT_DIR" --skip_train

        EVAL_LOG="log/eval_${DATASET_NAME}_${OBJECT}.log"
        python script/eval_lerf_mask.py "$DATASET_NAME" "$OBJECT" | tee "$EVAL_LOG"
        OBJ_MIOU=$(awk -F': ' '/Overall Mean IoU:/ {v=$2} END {print v}' "$EVAL_LOG")
        OBJ_BIOU=$(awk -F': ' '/Overall Boundary Mean IoU:/ {v=$2} END {print v}' "$EVAL_LOG")
        if [ -n "$OBJ_MIOU" ] && [ -n "$OBJ_BIOU" ]; then
            echo "$OBJECT,$OBJ_MIOU,$OBJ_BIOU" >> "$EVAL_SUMMARY"
            echo "[EvalSummary] $OBJECT mIoU=$OBJ_MIOU BIoU=$OBJ_BIOU"
        else
            echo "[EvalSummary][WARN] Failed to parse eval result for $OBJECT"
        fi


    fi
done

awk -F',' '
NR > 1 {
    miou += $2
    biou += $3
    n += 1
}
END {
    if (n > 0) {
        printf("[EvalSummary] objects=%d mean_mIoU=%.6f mean_BIoU=%.6f\n", n, miou / n, biou / n)
    } else {
        print("[EvalSummary][WARN] No valid eval rows.")
    }
}' "$EVAL_SUMMARY"
echo "[EvalSummary] Saved per-object results: $EVAL_SUMMARY"
echo "All scenes processed."

#!/bin/bash
# validation_orchestrator.sh

# Ensure required variables are passed from SBATCH
: "${APPTAINER_IMAGE:?Need to set APPTAINER_IMAGE}"
: "${BASE_CONFIG:?Need to set BASE_CONFIG}"
: "${FULL_LOG_DIR:?Need to set FULL_LOG_DIR}"

CONFIG_CSV="/home/nfs/jtverhoog/mmdet/mmdetection3d/projects/cmt/inference/runs.csv"
RESULTS_CSV="/home/nfs/jtverhoog/mmdet/mmdetection3d/projects/cmt/inference/validation_results.csv"
MODELS=("A" "B" "C" "D" "E")
SUBSETS=("full" "boston_day_clear" "boston_day_rain" "sing_night_clear" "sing_day_clear" "sing_night_rain")

# 1. Initialize Results CSV Header
if [ ! -f "$RESULTS_CSV" ]; then
    echo "Run_Name,Description,Model_ID,Status,full,boston_day_clear,boston_day_rain,sing_night_clear,sing_day_clear,sing_night_rain" > "$RESULTS_CSV"
fi

# 2. Metric Extraction Function (Looks for NDS in the log file)
extract_metric() {
    # Searches for the NuScenes Detection Score in the log file
    # Adjust the pattern if your MMDetection3D version outputs different keys
    local val=$(grep -oP '"pts_bbox_NuScenes/NDS": \K[0-9.]+' "$1" | tail -1)
    echo "${val:-0.0}"
}

# 3. Read Config (Ignore Header)
tail -n +2 "$CONFIG_CSV" | while IFS=, read -r NAME DESC BASE_DIR ROUND EPOCH OA OB OC OD OE; do
    
    OVERRIDES=("$OA" "$OB" "$OC" "$OD" "$OE")

    for i in "${!MODELS[@]}"; do
        MODEL_ID=${MODELS[$i]}
        OVERRIDE_PATH=${OVERRIDES[$i]}
        
        # Skip if already marked as 'Success' in the results CSV
        if grep -q "^$NAME,.*,$MODEL_ID,Success" "$RESULTS_CSV" 2>/dev/null; then
            echo ">> Skipping $NAME Model $MODEL_ID (Completed)"
            continue
        fi

        # Path Resolution
        if [ -n "$OVERRIDE_PATH" ]; then
            MODEL_PATH="$OVERRIDE_PATH"
        else
            MODEL_PATH="${BASE_DIR}/round_${ROUND}/Model${MODEL_ID}/epoch_${ROUND}.pth"
        fi

        # Handle Missing Model Files
        if [ ! -f "$MODEL_PATH" ]; then
            echo "!! Missing: $MODEL_PATH"
            echo "$NAME,\"$DESC\",$MODEL_ID,Missing,NaN,NaN,NaN,NaN,NaN,NaN" >> "$RESULTS_CSV"
            continue
        fi

        echo ">> Starting Validation: $NAME - Model $MODEL_ID"
        MODEL_SCORES=()
        
        # Create a specific directory for this model's logs
        CUR_LOG_ROOT="${FULL_LOG_DIR}/${NAME}/Model${MODEL_ID}"
        mkdir -p "$CUR_LOG_ROOT"

        for SUBSET in "${SUBSETS[@]}"; do
            # Define Data Root for this subset
            SUBSET_ROOT="/tudelft.net/staff-umbrella/MscThesisjverhoog/nuscenes_subsets_full/$SUBSET"
            [ "$SUBSET" == "full" ] && SUBSET_ROOT="/tudelft.net/staff-umbrella/MscThesisjverhoog/nuscenes_shadow_root"

            LOG_FILE="${CUR_LOG_ROOT}/${SUBSET}_output.log"
            echo "   Evaluating $SUBSET..."

            # Execute Validation inside Apptainer
            # We bind the subset root directly to the expected data directory
            apptainer exec --nv --cleanenv \
                --bind "$HOME/mmdet:/workspace/mmdet" \
                --bind "$HOME:$HOME" \
                --bind "/tudelft.net/staff-umbrella:/tudelft.net/staff-umbrella" \
                --bind "${SUBSET_ROOT}:/workspace/mmdet/mmdetection3d/data/nuscenes" \
                "$APPTAINER_IMAGE" \
                bash -lc "
                    cd /workspace/mmdet/mmdetection3d
                    python tools/test.py $BASE_CONFIG $MODEL_PATH \
                        --eval bbox \
                        --cfg-options \
                        data.test.data_root='/workspace/mmdet/mmdetection3d/data/nuscenes/'
                "> "$LOG_FILE" 2>&1 < /dev/null

            # Extract result and store
            SCORE=$(extract_metric "$LOG_FILE")
            MODEL_SCORES+=("$SCORE")
        done

        # 4. Save results for the model row
        # Using a temporary variable to join scores with commas
        SCORES_JOINED=$(IFS=,; echo "${MODEL_SCORES[*]}")
        echo "$NAME,\"$DESC\",$MODEL_ID,Success,$SCORES_JOINED" >> "$RESULTS_CSV"
    done
done
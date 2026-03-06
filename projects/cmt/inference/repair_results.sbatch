#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --job-name=repair_metrics
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/cmt_fedavg_all_val/logs/repair_%j.out

# --- PATH CONFIGURATION ---
CONFIG_CSV="/home/nfs/jtverhoog/mmdet/mmdetection3d/projects/cmt/inference/runs.csv"
RESULTS_CSV="/home/nfs/jtverhoog/mmdet/mmdetection3d/projects/cmt/inference/validation_results.csv"
LOG_ROOT="/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/cmt_fedavg_all_val/logs/run1"

MODELS=("A" "B" "C" "D" "E")
SUBSETS=("full" "boston_day_clear" "boston_day_rain" "sing_night_clear" "sing_day_clear" "sing_night_rain")

echo ">> Starting repair. Extracting NDS from $LOG_ROOT"

# 1. Overwrite the old results file with a fresh header
echo "Run_Name,Description,Model_ID,Status,full,boston_day_clear,boston_day_rain,sing_night_clear,sing_day_clear,sing_night_rain" > "$RESULTS_CSV"

# 2. Process the runs.csv (skipping the first header line)
tail -n +2 "$CONFIG_CSV" | while IFS=, read -r NAME DESC BASE_DIR ROUND EPOCH OA OB OC OD OE; do
    
    # We ignore empty names/lines
    [ -z "$NAME" ] && continue

    for MODEL_ID in "${MODELS[@]}"; do
        MODEL_SCORES=()
        LOGS_FOUND=0
        
        echo "   Processing $NAME - Model $MODEL_ID"

        for SUBSET in "${SUBSETS[@]}"; do
            # Path matches your pwd: LOG_ROOT/RunName/ModelID/subset_output.log
            LOG_FILE="${LOG_ROOT}/${NAME}/Model${MODEL_ID}/${SUBSET}_output.log"
            
            if [ -f "$LOG_FILE" ]; then
                # The corrected regex for your specific CMT logs
                SCORE=$(grep -oP 'NDS: \K[0-9.]+' "$LOG_FILE" | head -1)
                
                if [ -z "$SCORE" ]; then
                    MODEL_SCORES+=("0.0")
                else
                    MODEL_SCORES+=("$SCORE")
                    ((LOGS_FOUND++))
                fi
            else
                MODEL_SCORES+=("NaN")
            fi
        done

        # Determine Status
        if [ $LOGS_FOUND -eq 6 ]; then
            STATUS="Success"
        elif [ $LOGS_FOUND -gt 0 ]; then
            STATUS="Partial($LOGS_FOUND/6)"
        else
            STATUS="Missing"
        fi
        
        # Join scores with commas and write to CSV
        SCORES_JOINED=$(IFS=,; echo "${MODEL_SCORES[*]}")
        echo "$NAME,\"$DESC\",$MODEL_ID,$STATUS,$SCORES_JOINED" >> "$RESULTS_CSV"
    done
done

echo ">> Done! Overwritten: $RESULTS_CSV"
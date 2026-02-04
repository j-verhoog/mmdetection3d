#!/bin/bash
#SBATCH --partition=general # request partition
#SBATCH --qos=medium # Quality of service, short default
#SBATCH --job-name=create_data_all2
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/nuscenes_data_prep/remaining_recur_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/nuscenes_data_prep/remaining_recur_%j.err
#SBATCH --time=18:00:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

# --- CONFIGURATION ---
BASE_DATASETS_FOLDER="/tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/nuscenes_subsets"
APPTAINER_IMAGE="/tudelft.net/staff-umbrella/MscThesisjverhoog/image/mmdet3d_v1rc5.sif"

# --- MANUAL OVERRIDE ---
# If this list is NOT empty, the script will skip discovery and ONLY process these paths.
# Example: MANUAL_OVERRIDE=("/path/to/subset1" "/path/to/subset2")
# --- MANUAL OVERRIDE ---
MANUAL_OVERRIDE=()

# --- PHASE 1: DISCOVERY ---
TARGET_PATHS=()

# Paths to explicitly ignore (substring matching)
MANUAL_EXCLUDE=(
    "Exp2_TotalFair_SingleClient/boston_day_clear"
    "Exp2_TotalFair_SingleClient/boston_day_rain"
    "Exp2_TotalFair_SingleClient/singapore_day_clear"
    "Exp2_TotalFair_SingleClient/singapore_night_clear"
    "Exp2_TotalFair_SingleClient/singapore_night_rain"
)

if [ ${#MANUAL_OVERRIDE[@]} -gt 0 ]; then
    echo "!!! MANUAL OVERRIDE DETECTED !!!"
    TARGET_PATHS=("${MANUAL_OVERRIDE[@]}")
else
    echo "Searching for nuScenes root directories in: $BASE_DATASETS_FOLDER"
    
    # Find directories containing v1.0-trainval metadata
    # Added -L to follow symlinks if your structure uses them
    while read -r META_DIR; do
        NU_ROOT=$(dirname "$META_DIR")
        
        # --- EXCLUSION CHECK ---
        SKIP_DIR=false
        for EXCLUDE_PATH in "${MANUAL_EXCLUDE[@]}"; do
            # Check if the exclude string is contained within the found path
            if [[ "$NU_ROOT" == *"$EXCLUDE_PATH"* ]]; then
                echo "Skipping excluded directory: $NU_ROOT"
                SKIP_DIR=true
                break
            fi
        done
        
        if [ "$SKIP_DIR" = true ]; then
            continue
        fi
        # -----------------------

        # Validation: check for sample.json and samples/ folder
        if [ -f "$META_DIR/sample.json" ] && [ -d "$NU_ROOT/samples" ]; then
            TARGET_PATHS+=("$NU_ROOT")
            echo "Found valid nuScenes root: $NU_ROOT"
        else
            echo "Skipping invalid candidate: $NU_ROOT (Missing sample.json or samples/)"
        fi
    done < <(find -L "$BASE_DATASETS_FOLDER" -type d -name "v1.0-trainval")
fi

echo "--------------------------------------------------------"
echo "TOTAL SUBSETS TO PROCESS: ${#TARGET_PATHS[@]}"
echo "--------------------------------------------------------"

# --- PHASE 2: PROCESSING ---
TMP_BASE="/tmp/jtverhoog_${SLURM_JOB_ID}"
mkdir -p "$TMP_BASE"/{wandb,torch,xdg,tmp,home}

for NU_ROOT_SUBSET in "${TARGET_PATHS[@]}"; do
    SUBSET_NAME=$(basename "$NU_ROOT_SUBSET")
    
    echo ">>> PROCESSING: $SUBSET_NAME"
    echo ">>> PATH:       $NU_ROOT_SUBSET"

    apptainer exec --cleanenv \
    --bind "$HOME/mmdet:/workspace/mmdet"   \
    --bind "/tudelft.net/staff-umbrella/MscThesisjverhoog:/tudelft.net/staff-umbrella/MscThesisjverhoog"   \
    --bind "/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets:/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets"   \
    --bind "$TMP_BASE:/workspace/tmp" \
    --env TMPDIR=/workspace/tmp/tmp \
    --env TORCH_HOME=/workspace/tmp/torch   \
    --env XDG_CACHE_HOME=/workspace/tmp/xdg   \
    "$APPTAINER_IMAGE" \
    bash -lc "
    cd /workspace/mmdet/mmdetection3d && \
    python tools/create_data.py nuscenes \
      --root-path '$NU_ROOT_SUBSET' \
      --version v1.0 \
      --out-dir '$NU_ROOT_SUBSET' \
      --extra-tag nuscenes \
      --workers 16
    "

    if [ $? -eq 0 ]; then
        echo "Done with $SUBSET_NAME"
    else
        echo "!!! FAILED: $SUBSET_NAME"
    fi
    echo "--------------------------------------------------------"
done

echo "Execution complete."
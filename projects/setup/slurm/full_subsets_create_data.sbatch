#!/bin/bash
#SBATCH --partition=general # request partition
#SBATCH --qos=medium # Quality of service, short default
#SBATCH --job-name=nuscenes_full_subsets
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/nuscenes_data_prep/fullsubsets_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/nuscenes_data_prep/fullsubsets_%j.err
#SBATCH --time=12:00:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

# check if v1.0 or v1.0trainval is correct if it cancels running
export WANDB_PROJECT="test"
export WANDB_NAME="pointpillars_test"
export WANDB_API_KEY=$(awk '$1=="machine" && $2=="api.wandb.ai" {found=1}found && $1=="password" {print $2; exit}' "$HOME/.netrc")

/usr/bin/scontrol show job -d "$SLURM_JOB_ID" # check sbatch working

# define image and work dirs
export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/MscThesisjverhoog/image/mmdet3d_v1rc5.sif
export WORK=/tudelft.net/staff-umbrella/MscThesisjverhoog/pointpillars/test
mkdir -p "$WORK"

TMP_BASE="/tmp/jtverhoog_${SLURM_JOB_ID}"
mkdir -p "$TMP_BASE"/{wandb,torch,xdg,tmp,home}

SUBSETS=("boston_day_clear" "boston_night_clear" "singapore_day_rain" "singapore_night_rain")
BASE_PATH="/tudelft.net/staff-umbrella/MscThesisjverhoog/nuscenes_subsets_full"

for SUBSET_NAME in "${SUBSETS[@]}"; do

    # Construct the full path for the current subset
    NU_ROOT_SUBSET="${BASE_PATH}/${SUBSET_NAME}"

    echo "========================================"
    echo "=== PROCESSING SUBSET: $SUBSET_NAME ==="
    echo "=== PATH: $NU_ROOT_SUBSET ==="
    echo "========================================"
    
    echo "===GOING INTO CONTAINER==="
    apptainer exec --cleanenv \
    --bind "$HOME/mmdet:/workspace/mmdet"   \
    --bind "/tudelft.net/staff-umbrella/MscThesisjverhoog:/tudelft.net/staff-umbrella/MscThesisjverhoog"   \
    --bind "/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets:/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets"   \
    --bind "$TMP_BASE:/workspace/tmp" \
    --env TMPDIR=/workspace/tmp/tmp \
    --env TORCH_HOME=/workspace/tmp/torch   \
    --env XDG_CACHE_HOME=/workspace/tmp/xdg   \
    --env WANDB_DIR=/workspace/tmp/wandb   \
    --env WANDB_NAME="$WANDB_NAME" \
    --env WANDB_API_KEY="$WANDB_API_KEY" \
    "$APPTAINER_IMAGE" \
    bash -lc "
    cd /workspace/mmdet/mmdetection3d && \
    python tools/create_data.py nuscenes \
      --root-path $NU_ROOT_SUBSET \
      --version v1.0 \
      --out-dir $NU_ROOT_SUBSET \
      --extra-tag nuscenes \
      --workers 16
    "
    echo "=== COMPLETED SUBSET: $SUBSET_NAME ==="
        sleep 2 # Optional: short pause between runs to separate logs
    done


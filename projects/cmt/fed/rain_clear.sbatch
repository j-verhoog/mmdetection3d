#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --job-name=cmt_fedavg_TEST
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/cmt/fedavg_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/cmt/fedavg_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# --- FEDERATED SETTINGS ---
NUM_ROUNDS=2
EPOCHS_PER_ROUND=1 #### NOT WORKING SET IN CONFIG
EVAL_INTERVAL=1 ### SET IN CONFIG
SAMPLES_PER_GPU=8

# Ratios for averaging (e.g., 10k samples vs 5k samples)
SIZE_A=265
SIZE_B=125

# Initial Checkpoint (Optional: set to a path if starting from pre-trained, else leave empty)
# GLOBAL_MODEL="/path/to/pretrained.pth"
GLOBAL_MODEL=""

DAY_CLEAR_DATAROOT="/tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/nuscenes_subsets/Default_NoFair_SingleClient/boston_day_clear"
DAY_RAIN_DATAROOT="/tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/nuscenes_subsets/Default_NoFair_SingleClient/boston_day_rain"

CONFIG_A=$DAY_CLEAR_DATAROOT
CONFIG_B=$DAY_RAIN_DATAROOT



# --- STANDARD SETUP ---
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}
export WANDB_PROJECT="cmt_fedavg"
export WANDB_API_KEY=$(awk '$1=="machine" && $2=="api.wandb.ai" {found=1}found && $1=="password" {print $2; exit}' "$HOME/.netrc")

export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/MscThesisjverhoog/image/mmdet3d_v1rc5.sif
export WORK=/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/$WANDB_PROJECT/$SLURM_JOB_ID
mkdir -p "$WORK"

TMP_BASE="/tmp/jtverhoog_${SLURM_JOB_ID}"

# --- NODE SETUP ---
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 mkdir -p "$TMP_BASE"/{wandb,torch,xdg,tmp,home}
module use /opt/insy/modulefiles
module load cuda
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))

# --- TRAINING FUNCTION ---
# This function runs one training session inside the container
run_training () {
    local MODEL_NAME=$1
    local DATASET_PATH=$2
    local LOAD_FROM=$3
    local ROUND=$4
    local WORK_DIR="$WORK/round_${ROUND}/${MODEL_NAME}"
    
    # Logic to handle the load_from argument dynamically
    local LOAD_ARG=""
    if [ ! -z "$LOAD_FROM" ]; then
        # FIX: Removed internal quotes to prevent parsing issues in cfg-options
        LOAD_ARG="load_from=${LOAD_FROM}"
    fi

    echo "=== ROUND $ROUND | Training Model $MODEL_NAME ==="
    
    srun --ntasks=2 --gpus-per-task=1 bash -lc "
      apptainer exec --nv --cleanenv \
      --bind \"$HOME/mmdet:/workspace/mmdet\" \
      --bind \"$TMP_BASE:/workspace/tmp\" \
      --bind \"$WORK:/workspace/work_dirs\" \
      --bind \"/tudelft.net/staff-umbrella/MscThesisjverhoog:/tudelft.net/staff-umbrella/MscThesisjverhoog\" \
      --bind \"/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets:/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets\" \
      --bind \"$DATASET_PATH:/workspace/mmdet/mmdetection3d/data/nuscenes\" \
      --env SLURM_PROCID=\"\$SLURM_PROCID\" \
      --env SLURM_NTASKS=\"\$SLURM_NTASKS\" \
      --env SLURM_NODELIST=\"\$SLURM_NODELIST\" \
      --env SLURM_JOB_ID=\"\$SLURM_JOB_ID\" \
      --env TORCH_HOME=\"/workspace/tmp/torch/\$SLURM_PROCID\" \
      --env XDG_CACHE_HOME=\"/workspace/tmp/xdg/\$SLURM_PROCID\" \
      --env WANDB_DIR=\"/workspace/tmp/wandb/\$SLURM_PROCID\" \
      --env TMPDIR=\"/workspace/tmp/tmp/\$SLURM_PROCID\" \
      --env WANDB_PROJECT=\"$WANDB_PROJECT\" \
      --env WANDB_NAME=\"R${ROUND}_${MODEL_NAME}_\$SLURM_PROCID\" \
      --env WANDB_API_KEY=\"$WANDB_API_KEY\" \
      --env OMP_NUM_THREADS=1 \
      --env MKL_NUM_THREADS=1 \
      --env NCCL_DEBUG=INFO \
      --env NCCL_ASYNC_ERROR_HANDLING=1 \
      --env NCCL_BLOCKING_WAIT=1 \
      --env TORCH_DISTRIBUTED_DEBUG=DETAIL \
      --env MASTER_ADDR=\"$MASTER_ADDR\" \
      --env MASTER_PORT=\"$MASTER_PORT\" \
      --env NCCL_SOCKET_IFNAME=\"campus\" \
      \"$APPTAINER_IMAGE\" \
      bash -lc '
        cd /workspace/mmdet/mmdetection3d && \
        export PYTHONPATH=/workspace/mmdet/mmdetection3d:/workspace/mmdet:\$PYTHONPATH && \
        python -u tools/train.py \
        projects/cmt/configs/improved_lightweight_cmt_iterated.py \
          --launcher slurm \
          --work-dir $WORK_DIR \
          --cfg-options ${LOAD_ARG} \
            dist_params.port=$MASTER_PORT \
            data.samples_per_gpu=$SAMPLES_PER_GPU
      '
    "
}

# --- MERGE FUNCTION ---
# Runs the python merge script inside the container (single process, no srun needed really, but kept for env consistency)
run_merge () {
    local CKPT_A=$1
    local CKPT_B=$2
    local OUT_PATH=$3
    
    echo "=== Merging Models ==="
    apptainer exec --nv --cleanenv \
      --bind "$HOME/mmdet:/workspace/mmdet" \
      --bind "$WORK:/workspace/work_dirs" \
      "$APPTAINER_IMAGE" \
      bash -lc "
        cd /workspace/mmdet/mmdetection3d && \
        export PYTHONPATH=/workspace/mmdet/mmdetection3d:/workspace/mmdet:\$PYTHONPATH && \
        python tools/fed_avg_merge.py \
            $CKPT_A \
            $CKPT_B \
            $OUT_PATH \
            --weight-a $SIZE_A --weight-b $SIZE_B
      "
}


# ==========================================
# MAIN LOOP
# ==========================================

for ((i=1; i<=NUM_ROUNDS; i++)); do
    echo "-------------------------------------"
    echo "       STARTING FEDAVG ROUND $i      "
    echo "-------------------------------------"

    # 1. Train Model A
    # Note: We use the GLOBAL_MODEL from the previous round
    run_training "ModelA" "$CONFIG_A" "$GLOBAL_MODEL" "$i"
    CKPT_A="$WORK/round_${i}/ModelA/latest.pth"

    # 2. Train Model B
    # Note: We use the SAME global model start point
    run_training "ModelB" "$CONFIG_B" "$GLOBAL_MODEL" "$i"
    CKPT_B="$WORK/round_${i}/ModelB/latest.pth"

    # 3. Aggregate
    GLOBAL_MODEL="$WORK/round_${i}/global_model.pth"
    run_merge "$CKPT_A" "$CKPT_B" "$GLOBAL_MODEL"
    
    echo "Round $i complete. Global model saved to: $GLOBAL_MODEL"
done

# Cleanup
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 rm -rf "$TMP_BASE"
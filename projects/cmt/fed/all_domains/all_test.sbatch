#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --job-name=fedAVG_all_test
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/cmt/fedavg_all_test.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/cmt/fedavg_all_test.err
#SBATCH --time=2:00:00 #hr per round for rainclear
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G

set -e # stop if something fails

# --- FEDERATED SETTINGS ---
NUM_ROUNDS=20 # approx an hour per round of 1 Epoch
EPOCHS_PER_ROUND=1 
SAMPLES_PER_GPU=16
SIZE_A=265  # A MUST BE BIGGEST for logging
SIZE_B=125
SIZE_C=226
SIZE_D=71
SIZE_E=13

START_ROUND=1  # <--- NEW: Set to 1 for new run, or X to start loop at round X, round 16 mains load from global 15.pth
RESUME_CHECKPOINT="/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/cmt_fedavg/12215490/round_15/global_model.pth"

# Paths
DAY_CLEAR_DATAROOT="/tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/cmt_subsets/Default_NoFair_SingleClient/boston_day_clear"
DAY_RAIN_DATAROOT="/tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/cmt_subsets/Default_NoFair_SingleClient/boston_day_rain"
DAY_CLEAR_DATAROOT_SING="/tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/cmt_subsets/Default_NoFair_SingleClient/singapore_day_clear"
NIGHT_CLEAR_DATAROOT="/tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/cmt_subsets/Default_NoFair_SingleClient/singapore_night_clear"
NIGHT_RAIN_DATAROOT="/tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/cmt_subsets/Default_NoFair_SingleClient/singapore_night_rain"

CONFIG_A=$DAY_CLEAR_DATAROOT
CONFIG_B=$DAY_RAIN_DATAROOT
CONFIG_C=$DAY_CLEAR_DATAROOT_SING
CONFIG_D=$NIGHT_CLEAR_DATAROOT
CONFIG_E=$NIGHT_RAIN_DATAROOT

# Initial backbone weights (only used in Round 1)
PRETRAINED_BACKBONE="ckpts/nuim_r50.pth"

# --- STANDARD SETUP ---
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}
export WANDB_PROJECT="cmt_fedavg_all"
# Gets API key robustly
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

# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_instance_id,gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true)

# --- TRAINING FUNCTION ---
run_training () {
    local MODEL_NAME=$1
    local DATASET_PATH=$2
    local CHECKPOINT=$3
    local ROUND=$4
    local GLOBAL_EPOCHS=$5    # Changed from TARGET_EPOCHS
    local STOP_EPOCH=$6       # NEW ARGUMENT
    local IS_RESUME=$7
    
    local WORK_DIR="$WORK/round_${ROUND}/${MODEL_NAME}"
    
    # Configure Load vs Resume arguments
    local CFG_LOAD_ARG=""
    local RESUME_ARG=""
    
    if [ "$IS_RESUME" == "true" ]; then
        # In rounds > 1, we RESUME. This loads optimizer state and epoch count.
        RESUME_ARG="--resume-from ${CHECKPOINT}"
    else
        # In round 1, we LOAD weights only (backbone). Optimizer starts fresh.
        CFG_LOAD_ARG="load_from='${CHECKPOINT}'"
    fi

    echo "=== ROUND $ROUND | Training $MODEL_NAME | Stop Epoch: $STOP_EPOCH (Global: $GLOBAL_EPOCHS) ==="
    
    srun --ntasks=1 --gpus-per-task=1 bash -lc "
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
      --env WANDB_NAME=\"Rain_R${ROUND}_${MODEL_NAME}\" \
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
        projects/cmt/fed/all_domains/improved_lightweight_cmt_iterated.py \
          --launcher slurm \
          --work-dir $WORK_DIR \
          $RESUME_ARG \
          --cfg-options ${CFG_LOAD_ARG} dist_params.port=$MASTER_PORT \
            data.samples_per_gpu=$SAMPLES_PER_GPU \
            total_epochs=$GLOBAL_EPOCHS \
            custom_hooks.1.stop_epoch=$STOP_EPOCH 
      '
    "
}

# --- MERGE FUNCTION ---
run_merge () {
    local CKPT_A=$1
    local CKPT_B=$2
    local CKPT_C=$3
    local CKPT_D=$4
    local CKPT_E=$5
    local OUT_PATH=$6
    
    echo "=== Merging Models ==="
    apptainer exec --nv --cleanenv \
      --bind "$HOME/mmdet:/workspace/mmdet" \
      --bind "$WORK:/workspace/work_dirs" \
      --bind "/tudelft.net/staff-umbrella/MscThesisjverhoog:/tudelft.net/staff-umbrella/MscThesisjverhoog" \
      "$APPTAINER_IMAGE" \
      bash -lc "
        cd /workspace/mmdet/mmdetection3d && \
        export PYTHONPATH=/workspace/mmdet/mmdetection3d:/workspace/mmdet:\$PYTHONPATH && \
        python projects/cmt/fed/all_domains/fed_avg_merge.py \
            $CKPT_A \
            $CKPT_B \
            $CKPT_C \
            $CKPT_D \
            $CKPT_E \
            $OUT_PATH \
            --weight-a $SIZE_A --weight-b $SIZE_B --weight-c $SIZE_C --weight-d $SIZE_D --weight-e $SIZE_E
      "
}

# ==========================================
# MAIN LOOP
# ==========================================

# --- NEW LOGIC START ---
# Initialize Global Model based on START_ROUND
if [ "$START_ROUND" -eq 1 ]; then
    echo "Starting fresh run from Round 1"
    GLOBAL_MODEL="$PRETRAINED_BACKBONE"
else
    echo "Resuming run from Round $START_ROUND using checkpoint: $RESUME_CHECKPOINT"
    GLOBAL_MODEL="$RESUME_CHECKPOINT"
fi

# CALCULATE GLOBAL TOTAL ONCE
# This ensures the LR scheduler sees the full picture (e.g., 20 epochs) from Round 1
GLOBAL_MAX_EPOCHS=$((NUM_ROUNDS * EPOCHS_PER_ROUND))

for ((i=START_ROUND; i<=NUM_ROUNDS; i++)); do
    # This is where we want to STOP in this specific round
    CURRENT_STOP_EPOCH=$((i * EPOCHS_PER_ROUND))
    
    echo "-------------------------------------"
    echo "       STARTING FEDAVG ROUND $i      "
    echo "       Scheduler Horizon: $GLOBAL_MAX_EPOCHS epochs"
    echo "       Stop Training At:  $CURRENT_STOP_EPOCH epochs"
    echo "-------------------------------------"

    if [ "$i" -eq 1 ]; then
        IS_RESUME="false"
    else
        IS_RESUME="true"
    fi

    # UPDATE: Pass both GLOBAL total and LOCAL stop to the training function
    # Note: I am updating the function call arguments below
    run_training "ModelA" "$CONFIG_A" "$GLOBAL_MODEL" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME"
    CKPT_A="$WORK/round_${i}/ModelA/epoch_${CURRENT_STOP_EPOCH}.pth" # Use specific epoch

    run_training "ModelB" "$CONFIG_B" "$GLOBAL_MODEL" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME"
    CKPT_B="$WORK/round_${i}/ModelB/epoch_${CURRENT_STOP_EPOCH}.pth" # Use specific epoch

    run_training "ModelC" "$CONFIG_C" "$GLOBAL_MODEL" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME"
    CKPT_C="$WORK/round_${i}/ModelC/epoch_${CURRENT_STOP_EPOCH}.pth" # Use specific epoch

    run_training "ModelD" "$CONFIG_D" "$GLOBAL_MODEL" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME"
    CKPT_D="$WORK/round_${i}/ModelD/epoch_${CURRENT_STOP_EPOCH}.pth" # Use specific epoch

    run_training "ModelE" "$CONFIG_E" "$GLOBAL_MODEL" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME"
    CKPT_E="$WORK/round_${i}/ModelE/epoch_${CURRENT_STOP_EPOCH}.pth" # Use specific epoch

    # 3. Aggregate
    GLOBAL_MODEL="$WORK/round_${i}/global_model.pth"
    run_merge "$CKPT_A" "$CKPT_B" "$CKPT_C" "$CKPT_D" "$CKPT_E" "$GLOBAL_MODEL"
    
    echo "Round $i complete. Global model saved."
done

# Measure GPU usage of your job (result) for all allocated GPUs
/usr/bin/nvidia-smi --query-accounted-apps='gpu_instance_id,gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true

# Cleanup
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 rm -rf "$TMP_BASE"
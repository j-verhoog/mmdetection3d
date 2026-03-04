#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --job-name=baseline_sep_ctn
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/cmt/baseline_sept_job%j_rank%t.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/cmt/baseline_sept_job%j_rank%t.err
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G

set -e # stop if something fails
echo "=== Starting Job ID: $SLURM_JOB_ID ==="

# --- TRAINING SETTINGS ---
TOTAL_EPOCHS=20
SAMPLES_PER_GPU=16

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
export RUN_NAME="NoComm_baseline_sep_20to40"

# Gets API key robustly
export WANDB_API_KEY=$(awk '$1=="machine" && $2=="api.wandb.ai" {found=1}found && $1=="password" {print $2; exit}' "$HOME/.netrc")

export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/MscThesisjverhoog/image/mmdet3d_v1rc5.sif
export WORK=/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/$WANDB_PROJECT/$RUN_NAME
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
    local WANDB_RUN_ID=$4
    
    local WORK_DIR="$WORK/${MODEL_NAME}"

    echo "=== Training $MODEL_NAME for $TOTAL_EPOCHS epochs ==="
    
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
      --env WANDB_NAME=\"NoComm_baseline_${MODEL_NAME}\" \
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
      --env WANDB_RUN_ID=\"$WANDB_RUN_ID\" \
      --env WANDB_RESUME=\"allow\" \
      \"$APPTAINER_IMAGE\" \
      bash -lc '
        cd /workspace/mmdet/mmdetection3d && \
        export PYTHONPATH=/workspace/mmdet/mmdetection3d:/workspace/mmdet:\$PYTHONPATH && \
        python -u tools/train.py \
        projects/cmt/fed/all_domains/improved_lightweight_cmt_iterated.py \
          --launcher slurm \
          --work-dir $WORK_DIR \
          --cfg-options load_from="$CHECKPOINT" dist_params.port=$MASTER_PORT \
            data.samples_per_gpu=$SAMPLES_PER_GPU \
            total_epochs=$TOTAL_EPOCHS
      '
    "
}

# ==========================================
# MAIN: Train each model independently
# ==========================================

export WANDB_ID_A="baseline_A_12236408"
export WANDB_ID_B="baseline_B_12236408"
export WANDB_ID_C="baseline_C_12236408"
export WANDB_ID_D="baseline_D_12236408"
export WANDB_ID_E="baseline_E_12236408"

PRETRAINED_BACKBONE_A="/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/cmt_fedavg_all/run_20260227_185206/ModelA/epoch_20.pth"
PRETRAINED_BACKBONE_B="/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/cmt_fedavg_all/run_20260227_185206/ModelB/epoch_20.pth"
PRETRAINED_BACKBONE_C="/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/cmt_fedavg_all/run_20260301_142404/ModelC/epoch_20.pth"
PRETRAINED_BACKBONE_D="/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/cmt_fedavg_all/run_20260302_130200/ModelD/epoch_20.pth"
PRETRAINED_BACKBONE_E="/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/cmt_fedavg_all/run_20260302_130200/ModelE/epoch_20.pth"

run_training "ModelA" "$CONFIG_A" "$PRETRAINED_BACKBONE_A" "$WANDB_ID_A"
run_training "ModelB" "$CONFIG_B" "$PRETRAINED_BACKBONE_B" "$WANDB_ID_B"
run_training "ModelC" "$CONFIG_C" "$PRETRAINED_BACKBONE_C" "$WANDB_ID_C"
run_training "ModelD" "$CONFIG_D" "$PRETRAINED_BACKBONE_D" "$WANDB_ID_D"
run_training "ModelE" "$CONFIG_E" "$PRETRAINED_BACKBONE_E" "$WANDB_ID_E"

echo "All 5 baseline models trained independently for $TOTAL_EPOCHS epochs."

# Measure GPU usage of your job (result) for all allocated GPUs
/usr/bin/nvidia-smi --query-accounted-apps='gpu_instance_id,gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true

# Cleanup
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 rm -rf "$TMP_BASE"
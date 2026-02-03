#!/bin/bash
#SBATCH --partition=general # request partition
#SBATCH --qos=short # Quality of service, short default
#SBATCH --job-name=cmt_distr
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/cmt/distr_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/cmt/distr_%j.err
#SBATCH --time=01:00:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium
#SBATCH --nodes=4              # Use four machines
#SBATCH --ntasks=8             # Total tasks (2 per node)
#SBATCH --ntasks-per-node=2    # Spread them evenly
#SBATCH --gres=gpu:a40:2       # Request 2 GPUs per node (Total 8)
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# cmt baseline test

# FOR NAMING STICK TO: A–Z  a–z  0–9  .  _  -
export WANDB_PROJECT="cmt"
export WANDB_NAME="distr_baseline_original"
export WANDB_API_KEY=$(awk '$1=="machine" && $2=="api.wandb.ai" {found=1}found && $1=="password" {print $2; exit}' "$HOME/.netrc")

/usr/bin/scontrol show job -d "$SLURM_JOB_ID" # check sbatch working

# define image and work dirs
export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/MscThesisjverhoog/image/mmdet3d_v1rc5.sif
export WORK=/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/$WANDB_PROJECT/$WANDB_NAME
mkdir -p "$WORK"

# --- LOCAL DATA SETUP ---
NETWORK_DATA_ROOT="/tudelft.net/staff-umbrella/MscThesisjverhoog/nuscenes_shadow_root"
TMP_BASE="/tmp/jtverhoog_${SLURM_JOB_ID}"

# FIX: Use srun to create directories on ALL nodes, not just the master
echo "Creating tmp directories on all nodes..."
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 \
    mkdir -p "$TMP_BASE"/{wandb,torch,xdg,tmp,home}

# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true)

echo "=== LOADING CUDA ==="
module use /opt/insy/modulefiles
module load cuda

# Get the hostname of the first node in the list
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))

echo "Master node is: $MASTER_ADDR"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"


echo "=== ENTERING CONTAINER =="
srun --ntasks=8 --ntasks-per-node=2 --gpus-per-task=1 bash -lc "
  apptainer exec --nv --cleanenv \
  --bind \"$HOME/mmdet:/workspace/mmdet\" \
  --bind \"$TMP_BASE:/workspace/tmp\" \
  --bind \"$WORK:/workspace/work_dirs\" \
  --bind \"/tudelft.net/staff-umbrella/MscThesisjverhoog:/tudelft.net/staff-umbrella/MscThesisjverhoog\" \
  --bind \"/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets:/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets\" \
  --bind \"$NETWORK_DATA_ROOT:/workspace/mmdet/mmdetection3d/data/nuscenes\" \
  --env SLURM_PROCID=\"\$SLURM_PROCID\" \
  --env SLURM_NTASKS=\"\$SLURM_NTASKS\" \
  --env SLURM_NODELIST=\"\$SLURM_NODELIST\" \
  --env SLURM_JOB_ID=\"\$SLURM_JOB_ID\" \
  --env TORCH_HOME=\"/workspace/tmp/torch/\$SLURM_PROCID\" \
  --env XDG_CACHE_HOME=\"/workspace/tmp/xdg/\$SLURM_PROCID\" \
  --env WANDB_DIR=\"/workspace/tmp/wandb/\$SLURM_PROCID\" \
  --env TMPDIR=\"/workspace/tmp/tmp/\$SLURM_PROCID\" \
  --env WANDB_PROJECT=\"$WANDB_PROJECT\" \
  --env WANDB_NAME=\"${WANDB_NAME}_\$SLURM_PROCID\" \
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
    # IMPORTANT: Ensure CMT custom modules are in the PYTHONPATH
    export PYTHONPATH=$PYTHONPATH:$(pwd) && \
    python -u tools_CMT/train.py \
      projects/cmt/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs.py \
      --launcher slurm \
      --work-dir /workspace/work_dirs \
      --cfg-options \
        evaluation.interval=5 \
        data.samples_per_gpu=2 \
        dist_params.port='"$MASTER_PORT"'
  '
"



# Clean up local disk on ALL nodes
echo "Cleaning up tmp directories on all nodes..."
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 \
    rm -rf "$TMP_BASE"
    
# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true






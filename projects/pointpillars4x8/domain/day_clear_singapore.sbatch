#!/bin/bash
#SBATCH --partition=general # request partition
#SBATCH --qos=medium # Quality of service, short default
#SBATCH --job-name=dcS_distr    # takes 14/32 hours for 24 epochs == ??/16 hours for 12 epoch
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/pointpillars4x8/dcS_distr_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/pointpillars4x8/dcS_distr_%j.err
#SBATCH --time=08:00:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium
#SBATCH --nodes=2              # Use two machines
#SBATCH --ntasks=4             # Total tasks (2 per node)
#SBATCH --ntasks-per-node=2    # Spread them evenly
#SBATCH --gres=gpu:a40:2       # Request 2 GPUs per node (Total 4)
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# FOR NAMING STICK TO: A–Z  a–z  0–9  .  _  -
export WANDB_PROJECT="pointpillars4x8"
export WANDB_NAME="day_clear_Singapore"
export WANDB_API_KEY=$(awk '$1=="machine" && $2=="api.wandb.ai" {found=1}found && $1=="password" {print $2; exit}' "$HOME/.netrc")

/usr/bin/scontrol show job -d "$SLURM_JOB_ID" # check sbatch working

# Setup random port for communication between the 2 GPUs
export PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))

# define image and work dirs
export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/MscThesisjverhoog/image/mmdet3d_v1rc5.sif
export WORK=/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/$WANDB_PROJECT/$WANDB_NAME
mkdir -p "$WORK"

# --- LOCAL DATA SETUP ---
NETWORK_DATA_ROOT="/tudelft.net/staff-umbrella/MscThesisjverhoog/nuscenes_subsets_full/singapore_day_clear"
TMP_BASE="/tmp/jtverhoog_${SLURM_JOB_ID}"
mkdir -p "$TMP_BASE"/{wandb,torch,xdg,tmp,home}

# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true)

echo "=== LOADING CUDA ==="
module use /opt/insy/modulefiles
module load cuda

# setup ports for multigpu run
export MASTER_ADDR=$(srun -N1 -n1 hostname -s)
export MASTER_PORT=$PORT
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"


echo "=== ENTERING CONTAINER =="
srun --ntasks=2 --gpus-per-task=1 bash -lc "
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
  \"$APPTAINER_IMAGE\" \
  bash -lc '
    cd /workspace/mmdet/mmdetection3d && \
    python -u tools/train.py \
      configs/pointpillars/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d.py \
      --launcher slurm \
      --work-dir /workspace/work_dirs \
      --cfg-options load_from=checkpoints/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth \
      optimizer.lr=0.0001 evaluation.interval=1 runner.max_epochs=12 data.samples_per_gpu=8 lr_config.step=[9,11]
  '
"




# Clean up local disk after job
rm -rf "$TMP_BASE"

# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true






#!/bin/bash
#SBATCH --partition=general # request partition
#SBATCH --qos=short # Quality of service, short default
#SBATCH --job-name=pp_distr_test
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/pointpillars/distr_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/pointpillars/distr_%j.err
#SBATCH --time=01:00:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium
#SBATCH --nodes=1              # Force 1 node (Shared /tmp)
#SBATCH --gres=gpu:a40:2       # REQUEST 2 GPUs (Max per DAIC A40 node)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16     # 2 GPUs * 8 Cores = 16 Cores
#SBATCH --mem=128G

# added something to pointpillars config for wandb logging
# try disabling dev/shm 

# FOR NAMING STICK TO: A–Z  a–z  0–9  .  _  -
export WANDB_PROJECT="pointpillars"
export WANDB_NAME="distr"
export WANDB_API_KEY=$(awk '$1=="machine" && $2=="api.wandb.ai" {found=1}found && $1=="password" {print $2; exit}' "$HOME/.netrc")

/usr/bin/scontrol show job -d "$SLURM_JOB_ID" # check sbatch working

# Setup random port for communication between the 3 GPUs
export PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))

# define image and work dirs
export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/MscThesisjverhoog/image/mmdet3d_v1rc5.sif
export WORK=/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/$WANDB_PROJECT/$WANDB_NAME
mkdir -p "$WORK"

# --- LOCAL DATA SETUP ---
NETWORK_DATA_ROOT="/tudelft.net/staff-umbrella/MscThesisjverhoog/nuscenes_shadow_root"
TMP_BASE="/tmp/jtverhoog_${SLURM_JOB_ID}"
mkdir -p "$TMP_BASE"/{wandb,torch,xdg,tmp,home}

# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true)

echo "=== LOADING CUDA ==="
module use /opt/insy/modulefiles
module load cuda

echo "=== ENTERING CONTAINER =="
apptainer exec --nv --cleanenv \
--bind "$HOME/mmdet:/workspace/mmdet"   \
--bind "$TMP_BASE:/workspace/tmp" \
--bind "$WORK:/workspace/work_dirs" \
--bind "/tudelft.net/staff-umbrella/MscThesisjverhoog:/tudelft.net/staff-umbrella/MscThesisjverhoog"   \
--bind "/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets:/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets"   \
--bind "$NETWORK_DATA_ROOT:/workspace/mmdet/mmdetection3d/data/nuscenes" \
--env TMPDIR=/workspace/tmp/tmp \
--env TORCH_HOME=/workspace/tmp/torch   \
--env XDG_CACHE_HOME=/workspace/tmp/xdg   \
--env WANDB_DIR=/workspace/tmp/wandb   \
--env WANDB_PROJECT="$WANDB_PROJECT" \
--env WANDB_NAME="$WANDB_NAME" \
--env WANDB_API_KEY="$WANDB_API_KEY" \
--env OMP_NUM_THREADS=1 \
--env MKL_NUM_THREADS=1 \
--env NCCL_SHM_DISABLE=1 \
"$APPTAINER_IMAGE" \
bash -lc "
cd /workspace/mmdet/mmdetection3d && \
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
print('nccl_available', torch.distributed.is_nccl_available())
print('nccl', torch.cuda.nccl.version() if torch.cuda.is_available() else None)
print('gpus', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
&& \
bash tools/dist_train.sh \
  configs/pointpillars/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d.py \
  2 \
  --work-dir /workspace/work_dirs \
  --cfg-options \
  evaluation.interval=6 \
  data.samples_per_gpu=8 \
  data.workers_per_gpu=6
"

# Clean up local disk after job
rm -rf "$TMP_BASE"

# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true






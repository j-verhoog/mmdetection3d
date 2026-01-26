#!/bin/bash
#SBATCH --partition=general # request partition
#SBATCH --qos=medium # Quality of service, short default
#SBATCH --job-name=pointpillars_distr_test
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/pointpillars_distr_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/pointpillars_distr_%j.err
#SBATCH --time=00:30:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium
#SBATCH --nodes=1              # Force 1 node (Shared /tmp)
#SBATCH --gres=gpu:a40:3       # REQUEST 3 GPUs (Max per DAIC A40 node)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24     # 3 GPUs * 8 Cores = 24 Cores
#SBATCH --mem=256G

# added something to pointpillars config for wandb logging

# FOR NAMING STICK TO: A–Z  a–z  0–9  .  _  -
export WANDB_PROJECT="test"
export WANDB_NAME="pointpillars_distributed"
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
LOCAL_DATA_ROOT="${TMP_BASE}/nuscenes"
mkdir -p "$TMP_BASE"/{wandb,torch,xdg,tmp,home}
mkdir -p "$LOCAL_DATA_ROOT"

echo "=== COPYING DATA TO LOCAL SSD ==="
# This works because all 3 GPUs are on this ONE node sharing this /tmp
rsync -ah --info=progress2 "$NETWORK_DATA_ROOT/" "$LOCAL_DATA_ROOT/"

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
--bind "$LOCAL_DATA_ROOT:/workspace/mmdet/mmdetection3d/data/nuscenes" \
--env TMPDIR=/workspace/tmp/tmp \
--env TORCH_HOME=/workspace/tmp/torch   \
--env XDG_CACHE_HOME=/workspace/tmp/xdg   \
--env WANDB_DIR=/workspace/tmp/wandb   \
--env WANDB_PROJECT="$WANDB_PROJECT" \
--env WANDB_NAME="$WANDB_NAME" \
--env WANDB_API_KEY="$WANDB_API_KEY" \
--env OMP_NUM_THREADS=1 \
--env MKL_NUM_THREADS=1 \
"$APPTAINER_IMAGE" \
bash -lc "
# FIX 2: Cd into the directory before running tools
cd /workspace/mmdet/mmdetection3d && \
bash tools/dist_train.sh \
  configs/pointpillars/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d.py \
  3 \
  --work-dir /workspace/work_dirs \
  --cfg-options \
  evaluation.interval=24 \
  data.samples_per_gpu=5 \
  data.workers_per_gpu=6
"

# Clean up local disk after job
rm -rf "$TMP_BASE"

# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true






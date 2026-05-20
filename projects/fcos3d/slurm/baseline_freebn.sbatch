#!/bin/bash
#SBATCH --partition=general 
#SBATCH --qos=long # Quality of service, short default
#SBATCH --job-name=fcos3d_baseline
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/fcos3d/freebn_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/fcos3d/freebn_%j.err
#SBATCH --time=6-00:00:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium
#SBATCH --gres=gpu:a40:1 # request A40
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G # A40 clusters have 503GB spread on 3 gpu's
#SBATCH --tmp=50G 

# make it fail loudly
set -euo pipefail
trap 'echo "[ERROR] line=$LINENO cmd=$BASH_COMMAND" >&2' ERR

# try batchsize 16 for it was originally trained on 2x8

# FOR NAMING STICK TO: A–Z  a–z  0–9  .  _  -
export WANDB_PROJECT="fcos3d"
export WANDB_NAME="baseline_freebn"
export WANDB_API_KEY=$(awk '$1=="machine" && $2=="api.wandb.ai" {found=1}found && $1=="password" {print $2; exit}' "$HOME/.netrc")

/usr/bin/scontrol show job -d "$SLURM_JOB_ID" # check sbatch working

# define image and work dirs
export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/MscThesisjverhoog/image/mmdet3d_v1rc5.sif
export WORK=/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/$WANDB_PROJECT/$WANDB_NAME
mkdir -p "$WORK"

# --- SMART LOCAL DATA SETUP ---
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
--env WANDB_NAME="$WANDB_NAME" \
--env WANDB_API_KEY="$WANDB_API_KEY" \
--env WANDB_PROJECT="$WANDB_PROJECT" \
--env OMP_NUM_THREADS=1 \
--env MKL_NUM_THREADS=1 \
"$APPTAINER_IMAGE" \
bash -lc "
cd /workspace/mmdet/mmdetection3d && \
python -u tools/train.py configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py \
--work-dir /workspace/work_dirs \
--cfg-options \
data.workers_per_gpu=4 \
data.samples_per_gpu=8 \
optimizer.lr=0.001
"
# half lr for half batch size

# Clean up local disk after job
rm -rf "$TMP_BASE"

# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true






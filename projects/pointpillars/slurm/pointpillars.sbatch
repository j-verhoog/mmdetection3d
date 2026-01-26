#!/bin/bash
#SBATCH --partition=general,cor,me-cor # request partition, can submit to cor, me-cor but also maybe nto needed.
#SBATCH --qos=medium # Quality of service, short default
#SBATCH --job-name=pointpillars_baseline
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/pointpillars/baseline_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/pointpillars/baseline_%j.err
#SBATCH --time=36:00:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium
#SBATCH --gres=gpu:a40:1 # request A40
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G # A40 clusters have 503GB spread on 3 gpu's
#SBATCH --tmp=250G # Lidar only is like 150GB, keyframes 45GB, 3GB pkl files

# added something to pointpillars config for wandb logging

# FOR NAMING STICK TO: A–Z  a–z  0–9  .  _  -
export WANDB_PROJECT="pointpillars"
export WANDB_NAME="baseline_full_dataset"
export WANDB_API_KEY=$(awk '$1=="machine" && $2=="api.wandb.ai" {found=1}found && $1=="password" {print $2; exit}' "$HOME/.netrc")

/usr/bin/scontrol show job -d "$SLURM_JOB_ID" # check sbatch working

# define image and work dirs
export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/MscThesisjverhoog/image/mmdet3d_v1rc5.sif
export WORK=/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/$WANDB_PROJECT/$WANDB_NAME
mkdir -p "$WORK"

# --- SMART LOCAL DATA SETUP ---
NETWORK_DATA_ROOT="/tudelft.net/staff-umbrella/MscThesisjverhoog/nuscenes_shadow_root"
TMP_BASE="/tmp/jtverhoog_${SLURM_JOB_ID}"
LOCAL_DATA_ROOT="${TMP_BASE}/nuscenes"

mkdir -p "$TMP_BASE"/{wandb,torch,xdg,tmp,home}
mkdir -p "$LOCAL_DATA_ROOT"

# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true)

echo "=== LOADING CUDA ==="
module use /opt/insy/modulefiles
module load cuda

LOCAL_DATA_ROOT=$NETWORK_DATA_ROOT
# echo "=== COPYING LIDAR DATA TO LOCAL SSD ==="
# # 1. We enter the network directory to make the relative paths in rsync work easily
# cd "$NETWORK_DATA_ROOT"

# # 2. RUN SMART COPY, time it and let it be quiet
# # -r: recursive
# # -L: COPY THE ACTUAL FILE (Follow Symlink) - Crucial for speed!
# # --include: Specific folders we need for PointPillars
# # --exclude='*': Ignore everything else (Images, Radar) to save space
# time rsync -rL \
# --include='*.pkl' \
# --include='*.json' \
# --include='nuscenes_gt_database/***' \
# --include='v1.0-trainval/***' \
# --include='maps/***' \
# --include='samples/LIDAR_TOP/***' \
# --include='sweeps/LIDAR_TOP/***' \
# --include='samples/' \
# --include='sweeps/' \
# --exclude='*' \
# . "$LOCAL_DATA_ROOT/"

# # Return to previous dir just in case
# cd -

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
--env WANDB_NAME="$WANDB_NAME" \
--env WANDB_API_KEY="$WANDB_API_KEY" \
--env WANDB_PROJECT="$WANDB_PROJECT" \
--env OMP_NUM_THREADS=1 \
--env MKL_NUM_THREADS=1 \
"$APPTAINER_IMAGE" \
bash -lc "
cd /workspace/mmdet/mmdetection3d && \
python -u tools/train.py configs/pointpillars/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d.py \
--work-dir /workspace/work_dirs \
--cfg-options \
evaluation.interval=3 \
data.samples_per_gpu=16 \
data.workers_per_gpu=4
"

# Clean up local disk after job
rm -rf "$TMP_BASE"

# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true






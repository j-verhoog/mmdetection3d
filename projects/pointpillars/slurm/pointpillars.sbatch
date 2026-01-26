#!/bin/bash
#SBATCH --partition=general # request partition
#SBATCH --qos=short # Quality of service, short default
#SBATCH --job-name=pointpillars_test
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/pointpillars_test_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/pointpillars_test_%j.err
#SBATCH --time=0:30:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium
#SBATCH --gres=gpu:a40:1 # request A40
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G # A40 has 503GB spread on 3 gpu's

# added something to pointpillars config for wandb logging

# FOR NAMING STICK TO: A–Z  a–z  0–9  .  _  -
export WANDB_PROJECT="test"
export WANDB_NAME="pointpillars_test"
export WANDB_API_KEY=$(awk '$1=="machine" && $2=="api.wandb.ai" {found=1}found && $1=="password" {print $2; exit}' "$HOME/.netrc")

/usr/bin/scontrol show job -d "$SLURM_JOB_ID" # check sbatch working

# define image and work dirs
export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/MscThesisjverhoog/image/mmdet3d_v1rc5.sif
export WORK=/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/$WANDB_PROJECT/$WANDB_NAME
mkdir -p "$WORK"
export DATA_ROOT="/tudelft.net/staff-umbrella/MscThesisjverhoog/nuscenes_shadow_root"

# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true)

echo "=== LOADING CUDA ==="
module use /opt/insy/modulefiles
module load cuda

echo "=== CREATING TEMP_DIRS ==="
TMP_BASE="/tmp/jtverhoog_${SLURM_JOB_ID}"
mkdir -p "$TMP_BASE"/{wandb,torch,xdg,tmp,home}

echo "=== ENTERING CONTAINER =="
apptainer exec --nv --cleanenv \
--bind "$HOME/mmdet:/workspace/mmdet"   \
--bind "$TMP_BASE:/workspace/tmp" \
--bind "$WORK:/workspace/work_dirs" \
--bind "/tudelft.net/staff-umbrella/MscThesisjverhoog:/tudelft.net/staff-umbrella/MscThesisjverhoog"   \
--bind "/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets:/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets"   \
--bind "$DATA_ROOT:/workspace/mmdet/mmdetection3d/data/nuscenes" \
--env TMPDIR=/workspace/tmp/tmp \
--env TORCH_HOME=/workspace/tmp/torch   \
--env XDG_CACHE_HOME=/workspace/tmp/xdg   \
--env WANDB_DIR=/workspace/tmp/wandb   \
--env WANDB_NAME="$WANDB_NAME" \
--env WANDB_API_KEY="$WANDB_API_KEY" \
"$APPTAINER_IMAGE" \
bash -lc "
cd /workspace/mmdet/mmdetection3d && \
python -u tools/train.py configs/pointpillars/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d.py \
--work-dir /workspace/work_dirs \
--cfg-options \
evaluation.interval=1 \
"

# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true






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
#SBATCH --mem=32G

export WANDB_PROJECT="test"
export WANDB_NAME="pointpillars_test"

/usr/bin/scontrol show job -d "$SLURM_JOB_ID" # check sbatch working

# define image and work dirs
export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/MscThesisjverhoog/image/mmdet3d_v1rc5.sif
export WORK=/tudelft.net/staff-umbrella/MscThesisjverhoog/pointpillars/test
mkdir -p "$WORK"
mkdir -p /tmp/jtverhoog/{wandb,torch,xdg}

# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true)


# ---- Cuda ----
module use /opt/insy/modulefiles
module load cuda

apptainer run --nv   \
--bind "$HOME/mmdet:/workspace/mmdet"   \
--bind "/tmp/jtverhoog:/workspace/tmp" \
--bind "/tudelft.net/staff-umbrella/MscThesisjverhoog:/tudelft.net/staff-umbrella/MscThesisjverhoog"   \
--bind "$WORK:/workspace/work_dirs" \
--env TMPDIR=/workspace/tmp \
--env WANDB_DIR=/workspace/tmp/wandb   \
--env TORCH_HOME=/workspace/tmp/torch   \
--env XDG_CACHE_HOME=/workspace/tmp/xdg   \
--env WANDB_NAME="$WANDB_NAME" \
"$APPTAINER_IMAGE" \
bash -lc 'cd /workspace/mmdet/mmdetection3d && \
python tools/train.py configs/pointpillars/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d.py \
--work-dir /workspace/work_dirs
--cfg-options evaluation.interval=1

'

# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true






#!/bin/bash
#SBATCH --partition=general # request partition
#SBATCH --qos=short # Quality of service, short default
#SBATCH --job-name=nuscenes_data_prep
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/nuscenes_data_prep_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/nuscenes_data_prep_%j.err
#SBATCH --time=4:00:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

export WANDB_PROJECT="test"
export WANDB_NAME="pointpillars_test"

/usr/bin/scontrol show job -d "$SLURM_JOB_ID" # check sbatch working

# define image and work dirs
export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/MscThesisjverhoog/image/mmdet3d_v1rc5.sif
export WORK=/tudelft.net/staff-umbrella/MscThesisjverhoog/pointpillars/test
mkdir -p "$WORK"

TMP_BASE="/tmp/jtverhoog_${SLURM_JOB_ID}"
mkdir -p "$TMP_BASE"/{wandb,torch,xdg}
mkdir -p /tudelft.net/staff-umbrella/MscThesisjverhoog/nuscenes_pkl

apptainer exec  \
--bind "$HOME/mmdet:/workspace/mmdet"   \
--bind "/tmp/jtverhoog:/workspace/tmp" \
--bind "/tudelft.net/staff-umbrella/MscThesisjverhoog:/tudelft.net/staff-umbrella/MscThesisjverhoog"   \
--bind "/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets:/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets"   \
--bind "$WORK:/workspace/work_dirs" \
--env TMPDIR=/workspace/tmp \
--env WANDB_DIR=/workspace/tmp/wandb   \
--env TORCH_HOME=/workspace/tmp/torch   \
--env XDG_CACHE_HOME=/workspace/tmp/xdg   \
--env WANDB_NAME="$WANDB_NAME" \
"$APPTAINER_IMAGE" \
bash -lc 'cd /workspace/mmdet/mmdetection3d && \
python tools/create_data.py nuscenes \
  --root-path /tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets/nuscenes \
  --version v1.0 \
  --out-dir /tudelft.net/staff-umbrella/MscThesisjverhoog/nuscenes_pkl \
  --extra-tag nuscenes
'

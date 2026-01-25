#!/bin/bash
#SBATCH --partition=general # request partition
#SBATCH --qos=short # Quality of service, short default
#SBATCH --job-name=nuscenes_data_prep
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/nuscenes_data_prep/%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/nuscenes_data_prep/%j.err
#SBATCH --time=4:00:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

export WANDB_PROJECT="test"
export WANDB_NAME="pointpillars_test"
export WANDB_API_KEY=$(awk '$1=="machine" && $2=="api.wandb.ai" {found=1}found && $1=="password" {print $2; exit}' "$HOME/.netrc")

/usr/bin/scontrol show job -d "$SLURM_JOB_ID" # check sbatch working

# define image and work dirs
export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/MscThesisjverhoog/image/mmdet3d_v1rc5.sif
export WORK=/tudelft.net/staff-umbrella/MscThesisjverhoog/pointpillars/test
mkdir -p "$WORK"

TMP_BASE="/tmp/jtverhoog_${SLURM_JOB_ID}"
mkdir -p "$TMP_BASE"/{wandb,torch,xdg,tmp,home}
mkdir -p /tudelft.net/staff-umbrella/MscThesisjverhoog/nuscenes_pkl

echo "===WORK WITH SHADOW ROOT==="
NU_ROOT_REAL="/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets/nuscenes"
NU_ROOT_SHADOW="/tudelft.net/staff-umbrella/MscThesisjverhoog/nuscenes_shadow_root"

mkdir -p "$NU_ROOT_SHADOW"

# Symlink the parts nuScenes devkit expects
for d in samples sweeps maps lidarseg panoptic v1.0-trainval v1.0-test v1.0-mini; do
  if [ -e "$NU_ROOT_REAL/$d" ] && [ ! -e "$NU_ROOT_SHADOW/$d" ]; then
    ln -s "$NU_ROOT_REAL/$d" "$NU_ROOT_SHADOW/$d"
  fi
done

echo "===GOING INTO CONTAINER==="
apptainer exec --cleanenv \
--home /workspace/tmp/home \
--bind "$HOME/mmdet:/workspace/mmdet"   \
--bind "/tudelft.net/staff-umbrella/MscThesisjverhoog:/tudelft.net/staff-umbrella/MscThesisjverhoog"   \
--bind "/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets:/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets"   \
--bind "$TMP_BASE:/workspace/tmp" \
--env TMPDIR=/workspace/tmp/tmp \
--env TORCH_HOME=/workspace/tmp/torch   \
--env XDG_CACHE_HOME=/workspace/tmp/xdg   \
--env WANDB_DIR=/workspace/tmp/wandb   \
--env WANDB_NAME="$WANDB_NAME" \
--env WANDB_API_KEY="$WANDB_API_KEY" \
"$APPTAINER_IMAGE" \
bash -lc '
cd /workspace/mmdet/mmdetection3d && \
python tools/create_data.py nuscenes \
  --root-path /tudelft.net/staff-umbrella/MscThesisjverhoog/nuscenes_shadow_root \
  --version v1.0 \
  --out-dir /tudelft.net/staff-umbrella/MscThesisjverhoog/nuscenes_shadow_root \
  --extra-tag nuscenes
'

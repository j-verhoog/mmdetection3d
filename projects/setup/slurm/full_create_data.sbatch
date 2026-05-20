#!/bin/bash
#SBATCH --partition=general # request partition
#SBATCH --qos=medium # Quality of service, short default
#SBATCH --job-name=cmt_data_prep
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/cmt_data_prep/shadow_cmt.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/cmt_data_prep/shadow_cmt.err
#SBATCH --time=12:00:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

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

echo "===WORK WITH SHADOW ROOT==="
NU_ROOT_REAL="/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets/nuscenes"
NU_ROOT_SHADOW="/tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/nuscenes_cmt_full"

# Remove directory and recreate it
rm -rf "$NU_ROOT_SHADOW"
mkdir -p "$NU_ROOT_SHADOW"

# Copy real to shadow with symlinks only if not yet done
REQUIRED_DIRS=(samples sweeps maps lidarseg panoptic v1.0-trainval v1.0-test v1.0-mini)

for d in "${REQUIRED_DIRS[@]}"; do
  if [ -e "$NU_ROOT_REAL/$d" ]; then
    if [ ! -L "$NU_ROOT_SHADOW/$d" ]; then
      echo "Linking $d..."
      ln -sfn "$NU_ROOT_REAL/$d" "$NU_ROOT_SHADOW/$d"
    else
      echo "$d already linked. Skipping."
    fi
  else
    echo "Warning: Source directory $NU_ROOT_REAL/$d does not exist. Skipping."
  fi
done

echo "===GOING INTO CONTAINER==="
apptainer exec --cleanenv \
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
  --root-path /tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/nuscenes_cmt_full \
  --version v1.0 \
  --out-dir /tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/nuscenes_cmt_full \
  --extra-tag nuscenes \
  --workers 16
'

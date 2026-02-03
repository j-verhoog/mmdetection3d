#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --job-name=pp_test_4subs
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/pointpillars/pp_test_4subs_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/pointpillars/pp_test_4subs_%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# OTHER CONFIG FILE AND CHECKPOINT

set -euo pipefail

# FOR NAMING STICK TO: A–Z  a–z  0–9  .  _  -
export WANDB_PROJECT="pointpillars4x8"
export WANDB_NAME="base_testing"
export WANDB_API_KEY=$(awk '$1=="machine" && $2=="api.wandb.ai" {found=1}found && $1=="password" {print $2; exit}' "$HOME/.netrc")

/usr/bin/scontrol show job -d "$SLURM_JOB_ID"

# Setup random port (keep this, even if single GPU)
export PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))

# define image and output dirs
export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/MscThesisjverhoog/image/mmdet3d_v1rc5.sif
OUT_ROOT_HOST=/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/pointpillars4x8/base_testing_pretrained
mkdir -p "$OUT_ROOT_HOST"

# --- LOCAL TMP SETUP (keep structure you used) ---
TMP_BASE="/tmp/jtverhoog_${SLURM_JOB_ID}"
mkdir -p "$TMP_BASE"/{wandb,torch,xdg,tmp,home}

# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true)

echo "=== LOADING CUDA ==="
module use /opt/insy/modulefiles
module load cuda

# setup ports (keep this)
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$PORT
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# ----------------------------
# What to test
# ----------------------------
# IMPORTANT: set this to the checkpoint you want to evaluate (HOST path)
CKPT_HOST=/home/nfs/jtverhoog/mmdet/mmdetection3d/checkpoints/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth

# 4 subset roots (HOST paths)
declare -A SUBSETS
SUBSETS[full_set]="/tudelft.net/staff-umbrella/MscThesisjverhoog/nuscenes_shadow_root"
SUBSETS[boston_day_clear]="/tudelft.net/staff-umbrella/MscThesisjverhoog/nuscenes_subsets_full/boston_day_clear"
SUBSETS[boston_day_rain]="/tudelft.net/staff-umbrella/MscThesisjverhoog/nuscenes_subsets_full/boston_day_rain"
SUBSETS[singapore_night_clear]="/tudelft.net/staff-umbrella/MscThesisjverhoog/nuscenes_subsets_full/singapore_night_clear"
SUBSETS[singapore_day_clear]="/tudelft.net/staff-umbrella/MscThesisjverhoog/nuscenes_subsets_full/singapore_day_clear"

# ----------------------------
# KEEP-ALL BINDS (from your train script)
# ----------------------------
COMMON_BINDS=(
  --bind "$HOME/mmdet:/workspace/mmdet"
  --bind "$TMP_BASE:/workspace/tmp"
  --bind "$OUT_ROOT_HOST:/workspace/out"
  --bind "/tudelft.net/staff-umbrella/MscThesisjverhoog:/tudelft.net/staff-umbrella/MscThesisjverhoog"
  --bind "/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets:/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets"
)

# Also bind config + ckpt explicitly (extra robust)
COMMON_BINDS+=(
  --bind "$CKPT_HOST:/workspace/ckpt.pth"
)

# ----------------------------
# KEEP-ALL ENVS (from your train script)
# ----------------------------
COMMON_ENVS=(
  --env SLURM_PROCID="${SLURM_PROCID:-0}"
  --env SLURM_NTASKS="${SLURM_NTASKS:-1}"
  --env SLURM_NODELIST="${SLURM_NODELIST:-}"
  --env SLURM_JOB_ID="${SLURM_JOB_ID:-}"
  --env TORCH_HOME="/workspace/tmp/torch/${SLURM_PROCID:-0}"
  --env XDG_CACHE_HOME="/workspace/tmp/xdg/${SLURM_PROCID:-0}"
  --env WANDB_DIR="/workspace/tmp/wandb/${SLURM_PROCID:-0}"
  --env TMPDIR="/workspace/tmp/tmp/${SLURM_PROCID:-0}"
  --env WANDB_PROJECT="$WANDB_PROJECT"
  --env WANDB_NAME="${WANDB_NAME}_${SLURM_PROCID:-0}"
  --env WANDB_API_KEY="$WANDB_API_KEY"
  --env OMP_NUM_THREADS=1
  --env MKL_NUM_THREADS=1
  --env NCCL_DEBUG=INFO
  --env NCCL_ASYNC_ERROR_HANDLING=1
  --env NCCL_BLOCKING_WAIT=1
  --env TORCH_DISTRIBUTED_DEBUG=DETAIL
  --env MASTER_ADDR="$MASTER_ADDR"
  --env MASTER_PORT="$MASTER_PORT"
)

# ----------------------------
# Run evaluation sequentially over subsets
# ----------------------------
for name in full_set boston_day_clear boston_day_rain singapore_night_clear singapore_day_clear; do
  host_root="${SUBSETS[$name]}"
  out_dir_host="${OUT_ROOT_HOST}/${name}"
  mkdir -p "$out_dir_host"

  echo "============================================================"
  echo "Subset: $name"
  echo "Host root: $host_root"
  echo "Out dir: $out_dir_host"
  echo "============================================================"

  apptainer exec --nv --cleanenv \
    "${COMMON_BINDS[@]}" \
    --bind "${host_root}:/workspace/mmdet/mmdetection3d/data/nuscenes" \
    "${COMMON_ENVS[@]}" \
    "$APPTAINER_IMAGE" \
    bash -lc "
      set -euo pipefail
      cd /workspace/mmdet/mmdetection3d

      echo 'CKPT: /workspace/ckpt.pth'           | tee '/workspace/out/${name}/meta.txt'
      echo 'SUBSET: ${name}'                     | tee -a '/workspace/out/${name}/meta.txt'
      echo 'DATA_ROOT(bind): data/nuscenes -> subset root' | tee -a '/workspace/out/${name}/meta.txt'

      python -u tools/test.py \
        /workspace/mmdet/mmdetection3d/configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py \
        /workspace/ckpt.pth \
        --eval bbox \
        --out /workspace/out/${name}/results.pkl \
        --seed 0 \
        --deterministic \
        --cfg-options data.test.data_root='/workspace/mmdet/mmdetection3d/data/nuscenes/' \
        2>&1 | tee /workspace/out/${name}/metrics.txt
    "
done

# Clean up local disk after job
rm -rf "$TMP_BASE"

# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true

echo "DONE. Outputs in: $OUT_ROOT_HOST"

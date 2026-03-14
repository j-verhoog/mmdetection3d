#!/bin/bash
#SBATCH --account=education-me-msc-ro
#SBATCH --partition=gpu-a100
#SBATCH --job-name=centr_base
#SBATCH --output=/scratch/%u/logs/delftblue_paper_config_job%j.out
#SBATCH --error=/scratch/%u/logs/delftblue_paper_config_job%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1              # Amount of machines
#SBATCH --ntasks=4             # Total tasks (4 per node)
#SBATCH --ntasks-per-node=4    # Spread them evenly
#SBATCH --cpus-per-task=8      # 8 cores per task (32 cores total)
#SBATCH --gpus-per-task=1      # 1 GPU per task (4 GPUs total)
#SBATCH --mem-per-cpu=8G       # 8GB per CPU (256GB total)
#SBATCH --tmp=500G            # 500GB local scratch (adjust as needed)

set -e # Good practice to fail fast
echo "=== Starting Job ID: $SLURM_JOB_ID ==="

echo "Printing ip link stuff"
ip link show


PRETRAINED_BACKBONE="ckpts/fcos3d_vovnet_imgbackbone-remapped.pth"
TARGET_EPOCHS=20
SAMPLES_PER_GPU=4       # 4 leads OOM on A40, 2==31GB and takes 11 days on two gpu's

# FOR NAMING STICK TO: A–Z  a–z  0–9  .  _  -
export WANDB_PROJECT="cmt"
export WANDB_NAME="paper_config_4x4_hour_test_delftblue"
export WANDB_API_KEY=$(awk '$1=="machine" && $2=="api.wandb.ai" {found=1}found && $1=="password" {print $2; exit}' "$HOME/.netrc")

/usr/bin/scontrol show job -d "$SLURM_JOB_ID" # check sbatch working

# define image and work dirs
export APPTAINER_IMAGE=/scratch/$USER/image/mmdet3d_v1rc5.sif
export WORK=/scratch/$USER/projects/$WANDB_PROJECT/$WANDB_NAME
mkdir -p "$WORK"

# --- LOCAL DATA SETUP ---
NETWORK_DATA_ROOT="/scratch/$USER/data/nuscenes"
TMP_BASE="${TMPDIR}/jtverhoog_${SLURM_JOB_ID}"

# Create tmp directories and register cleanup trap (runs even on timeout/cancel)
echo "Creating tmp directories..."
/usr/bin/mkdir --parents "$TMP_BASE"/{wandb,torch,xdg,tmp,home,nuscenes} && echo "$TMP_BASE created successfully."

function clean_up {
  /usr/bin/rm --recursive --force "$TMP_BASE" && echo "Clean up of $TMP_BASE completed successfully."
  exit
}
trap 'clean_up' EXIT

# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true)

echo "=== LOADING CUDA ==="
module load 2024r1
module load cuda

# Get the hostname of the first node in the list
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(apptainer exec --cleanenv "$APPTAINER_IMAGE" python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "Master node is: $MASTER_ADDR"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# Copy everything EXCEPT the .tar / .tar.gz files from scratch to tmp
echo "Copying base files to local tmp..."
time rsync -a --exclude='*.tar' "$NETWORK_DATA_ROOT/" "$TMP_BASE/nuscenes/"

# Extract the tarballs directly into that same tmp folder
echo "Extracting GT database and heavy data to local tmp..."
time tar -xf /scratch/$USER/data/nuscenes/nuscenes_gt_database.tar -C "$TMP_BASE/nuscenes"
time tar -xf /scratch/$USER/data/nuscenes/nuscenes_samples_sweeps.tar -C "$TMP_BASE/nuscenes"

echo "=== ENTERING CONTAINER ==="
srun --ntasks=4 --gpus-per-task=1 bash -lc "
  apptainer exec --nv --cleanenv \
  --bind \"$HOME/mmdet:/workspace/mmdet\" \
  --bind \"$TMP_BASE/nuscenes:/workspace/mmdet/mmdetection3d/data/nuscenes\" \
  --bind \"$TMP_BASE:/workspace/tmp\" \
  --bind \"$WORK:/workspace/work_dirs\" \
  --env SLURM_PROCID=\"\$SLURM_PROCID\" \
  --env SLURM_NTASKS=\"\$SLURM_NTASKS\" \
  --env SLURM_NODELIST=\"\$SLURM_NODELIST\" \
  --env SLURM_JOB_ID=\"\$SLURM_JOB_ID\" \
  --env TORCH_HOME=\"/workspace/tmp/torch/\$SLURM_PROCID\" \
  --env XDG_CACHE_HOME=\"/workspace/tmp/xdg/\$SLURM_PROCID\" \
  --env WANDB_DIR=\"/workspace/tmp/wandb/\$SLURM_PROCID\" \
  --env TMPDIR=\"/workspace/tmp/tmp/\$SLURM_PROCID\" \
  --env WANDB_PROJECT=\"$WANDB_PROJECT\" \
  --env WANDB_NAME=\"${WANDB_NAME}\" \
  --env WANDB_API_KEY=\"$WANDB_API_KEY\" \
  --env OMP_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  --env NCCL_DEBUG=INFO \
  --env NCCL_ASYNC_ERROR_HANDLING=1 \
  --env NCCL_BLOCKING_WAIT=1 \
  --env TORCH_DISTRIBUTED_DEBUG=DETAIL \
  --env MASTER_ADDR=\"$MASTER_ADDR\" \
  --env MASTER_PORT=\"$MASTER_PORT\" \
  --env NCCL_SOCKET_IFNAME=\"ib\" \
  \"$APPTAINER_IMAGE\" \
  bash -lc '
    cd /workspace/mmdet/mmdetection3d && \
    export PYTHONPATH=/workspace/mmdet/mmdetection3d:/workspace/mmdet:\$PYTHONPATH && \
    python -u tools/train.py \
      projects/cmt/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs.py \
      --launcher slurm \
      --work-dir /workspace/work_dirs \
      --cfg-options \
        evaluation.interval=1 \
        load_from='"$PRETRAINED_BACKBONE"' \
        total_epochs=$TARGET_EPOCHS \
        data.samples_per_gpu=$SAMPLES_PER_GPU \
        dist_params.port='"$MASTER_PORT"'
  '
"



# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true






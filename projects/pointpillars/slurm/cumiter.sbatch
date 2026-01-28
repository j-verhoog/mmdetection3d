#!/bin/bash
#SBATCH --partition=general,cor,me-cor # request partition, can submit to cor, me-cor but also maybe nto needed.
#SBATCH --qos=short # Quality of service, short default
#SBATCH --job-name=pp_base3
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/pointpillars/baseline3_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/pointpillars/baseline3_%j.err
#SBATCH --time=01:00:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium
#SBATCH --gres=gpu:a40:1 # request A40
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G # A40 clusters have 503GB spread on 3 gpu's
#SBATCH --tmp=50G # Lidar sweeps only is like 150GB, keyframes 45GB, 3GB pkl files -- dowload of keyframes and pkl took 3 hrs. Epoch [6][6300/14065] after 13:43:19

############# NO MORE TOUCH, IT WORKS FOR CUMITER

# changed default_runtime.py and train.py for wandb logging. Also can remove gt_databse in next run for it is unused
# same model to run, different batch size. for previous one resulted in unstable NaN loss

# FOR NAMING STICK TO: A–Z  a–z  0–9  .  _  -
export WANDB_PROJECT="pointpillars"
export WANDB_NAME="baseline3_full_dataset_cumiter"
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
python -u tools/train.py configs/pointpillars/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d.py \
--work-dir /workspace/work_dirs \
--cfg-options \
evaluation.interval=6 \
data.workers_per_gpu=4 \
data.samples_per_gpu=8 \
optimizer_config.cumulative_iters=2 \
model.pts_voxel_encoder.norm_cfg.momentum=0.005 \
model.pts_backbone.norm_cfg.momentum=0.005 \
model.pts_neck.norm_cfg.momentum=0.005
"
# batchsize 16 == OOM on A40 --> accumulate batches across iters for update. Halved momentum so same amount of batches are relevant
# custom hook in mmdet/apis/train to make cumulative iters work

# Clean up local disk after job
rm -rf "$TMP_BASE"

# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true






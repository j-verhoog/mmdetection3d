#!/bin/bash
#SBATCH --account=education-me-msc-ro
#SBATCH --partition=gpu-a100
#SBATCH --job-name=fedselect_full_elastic
#SBATCH --output=/scratch/%u/logs/fedselect_full_elastic_job%j.out
#SBATCH --error=/scratch/%u/logs/fedselect_full_elastic_job%j.err
#SBATCH --time=36:00:00
#SBATCH --ntasks=1             # Total tasks 
#SBATCH --cpus-per-task=8      # 8 cores per task 
#SBATCH --gpus-per-task=1      # 1 GPU per task 
#SBATCH --mem-per-cpu=8000M       # 8GB per CPU

set -e # stop if something fails
echo "=== Starting Job ID: $SLURM_JOB_ID ==="

# --- FEDERATED SETTINGS ---
NUM_ROUNDS=40           # approx 2hour per round of 1 Epoch
EPOCHS_PER_ROUND=1 
SAMPLES_PER_GPU=16
SIZE_A=265              # A MUST BE BIGGEST for logging
SIZE_B=125
SIZE_C=226
SIZE_D=71
SIZE_E=13

START_ROUND=1 
START_MODEL="A"         # If set, runs from that model for the start round, then continues normally. 'A' does nothing

# Paths
DATASET_A="/tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/cmt_subsets/Default_NoFair_SingleClient/boston_day_clear"
DATASET_B="/tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/cmt_subsets/Default_NoFair_SingleClient/boston_day_rain"
DATASET_C="/tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/cmt_subsets/Default_NoFair_SingleClient/singapore_day_clear"
DATASET_D="/tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/cmt_subsets/Default_NoFair_SingleClient/singapore_night_clear"
DATASET_E="/tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/cmt_subsets/Default_NoFair_SingleClient/singapore_night_rain"

# Initial backbone weights (only used in Round 1), this is the correct one for the R50 backbone
PRETRAINED_BACKBONE="ckpts/nuim_r50.pth"

# --- STANDARD SETUP ---
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}
export WANDB_PROJECT="cmt_fedavg_all"
export RUN_NAME="FedSelect_full_elastic"
CONTINUE_WANDB_RUN=false

# Gets API key robustly
export WANDB_API_KEY=$(awk '$1=="machine" && $2=="api.wandb.ai" {found=1}found && $1=="password" {print $2; exit}' "$HOME/.netrc")

export APPTAINER_IMAGE=/scratch/$USER/image/mmdet3d_v1rc5.sif
export WORK=/scratch/$USER/projects/$WANDB_PROJECT/$WANDB_NAME
mkdir -p "$WORK"

# Create tmp directories and register cleanup trap (runs even on timeout/cancel)
echo "Creating tmp directories..."
# Define the TMP_BASE as before
TMP_BASE="${TMPDIR}/jtverhoog_${SLURM_JOB_ID}"

echo "Creating tmp directories and extracting data on ALL allocated nodes..."

# srun with --ntasks=$SLURM_NNODES and --ntasks-per-node=1 ensures this runs exactly once per physical machine. took 20 minutes first
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -c "
  REQUIRED_GB=380
  
  # Get available space in GB (stripping out the 'G' and any spaces)
  AVAIL_GB=\$(df -BG --output=avail \"${TMPDIR}\" | tail -n 1 | tr -dc '0-9')
  
  echo \"=== Disk space on \$(hostname): \${AVAIL_GB}GB available ===\"
  
  if [ \"\$AVAIL_GB\" -lt \"\$REQUIRED_GB\" ]; then
    echo \"FATAL: Not enough local storage on \$(hostname). Needs \${REQUIRED_GB}GB, has \${AVAIL_GB}GB.\"
    exit 1
  fi
  
  echo \"Setting up data on \$(hostname)...\"
  mkdir -p \"$TMP_BASE\"/{wandb,torch,xdg,tmp,home,nuscenes}
  
  echo \"Starting parallel data copy and extraction on \$(hostname)...\"

  # 1. Start rsync in the background
  rsync -a --info=progress2 --exclude='*.tar' \"$NETWORK_DATA_ROOT/\" \"$TMP_BASE/nuscenes/\" &
  
  # 2. Start first tar extraction in the background
  tar -xf \"/scratch/$USER/data/nuscenes/nuscenes_gt_database.tar\" -C \"$TMP_BASE/nuscenes\" --checkpoint=10000 --checkpoint-action=dot &
  
  # 3. Start second tar extraction in the background
  tar -xf \"/scratch/$USER/data/nuscenes/nuscenes_samples_sweeps.tar\" -C \"$TMP_BASE/nuscenes\" --checkpoint=10000 --checkpoint-action=dot &
  
  # Wait for all background tasks to finish before moving on
  wait
  
  echo \"Finished all parallel extractions on \$(hostname).\"
"

function clean_up {
  echo "Cleaning up $TMP_BASE on all nodes..."
  srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 rm -rf "$TMP_BASE"
  echo "Clean up completed."
  exit
}
trap 'clean_up' EXIT


echo "=== LOADING CUDA ==="
module load 2024r1
module load cuda


# master port queries for free ports and uses that per training run, ensuring all processes can communicate properly
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# --- ONE General training function for all rounds: CHECK PATH ---
source /home/jtverhoog/mmdet/mmdetection3d/projects/cmt_delftblue/fed/fedSelect/full_elastic/runner_delftblue.sh


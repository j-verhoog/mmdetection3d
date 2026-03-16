#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --job-name=fedSelect_40
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/cmt/fedselect40_full_ctn_job%j_rank%t.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/cmt/fedselect40_full_ctn_job%j_rank%t.err
#SBATCH --time=80:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

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

START_ROUND=19 
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
export RUN_NAME="FedSelect_40epochs"
CONTINUE_WANDB_RUN=true

# Gets API key robustly
export WANDB_API_KEY=$(awk '$1=="machine" && $2=="api.wandb.ai" {found=1}found && $1=="password" {print $2; exit}' "$HOME/.netrc")

export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/MscThesisjverhoog/image/mmdet3d_v1rc5.sif
export WORK=/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/$WANDB_PROJECT/$RUN_NAME
mkdir -p "$WORK"

TMP_BASE="/tmp/jtverhoog_${SLURM_JOB_ID}"

# --- NODE SETUP ---
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 mkdir -p "$TMP_BASE"/{wandb,torch,xdg,tmp,home}
module use /opt/insy/modulefiles
module load cuda

# master port queries for free ports and uses that per training run, ensuring all processes can communicate properly
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# --- ONE General training function for all rounds: CHECK PATH ---
source /home/nfs/jtverhoog/mmdet/mmdetection3d/projects/cmt/fed/fedSelect/runner_40epochs.sh

# Cleanup
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 rm -rf "$TMP_BASE"
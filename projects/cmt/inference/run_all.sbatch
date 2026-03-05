#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --job-name=val_run_all
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/cmt_fedavg_all_val/logs/run1_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/cmt_fedavg_all_val/logs/run1_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

# --- User Configuration ---
export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/MscThesisjverhoog/image/mmdet3d_v1rc5.sif
export BASE_CONFIG=/workspace/mmdet/mmdetection3d/projects/cmt/fed/all_domains/improved_lightweight_cmt_iterated.py
export FULL_LOG_DIR="/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/cmt_fedavg_all_val/logs/run1"
mkdir -p "$FULL_LOG_DIR"

# --- Run Orchestrator ---
# Make sure the .sh file is executable: chmod +x validate.sh
bash /home/nfs/jtverhoog/mmdet/mmdetection3d/projects/cmt/inference/validate.sh

echo "Full job completed. Logs saved to: $FULL_LOG_DIR"

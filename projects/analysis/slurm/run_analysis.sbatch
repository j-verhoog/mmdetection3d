#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --job-name=pp_analysis
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/pointpillars/analysis_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/pointpillars/analysis_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

# 1. Configuration & Paths
export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/MscThesisjverhoog/image/mmdet3d_v1rc5.sif
OUT_ROOT_HOST=/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/pointpillars/analysis
DATA_HOST=/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets/nuscenes/LIDAR_TOP/

# Base directory for your checkpoints
CP_BASE="/home/nfs/jtverhoog/mmdet/MscThesisjverhoog/projects/pointpillars4x8"

mkdir -p "$OUT_ROOT_HOST"

# 2. Define Subsets 
# Format: "Relative_Path_to_PT1 Relative_Path_to_PT2 Label1 Label2"
declare -A SUBSETS
SUBSETS[night]="day_clear_Boston/latest.pth night_clear_Singapore/latest.pth DayClear_Bost NightClear_Sing"

# 3. Environment & Binds
COMMON_BINDS=(
  --bind "$HOME/mmdet:/workspace/mmdet"
  --bind "$OUT_ROOT_HOST:/workspace/out"
  --bind "$DATA_HOST:/data/LIDAR_TOP"
  --bind "/tudelft.net/staff-umbrella/MscThesisjverhoog:/tudelft.net/staff-umbrella/MscThesisjverhoog"
  --bind "$CP_BASE:/checkpoints"
)

# 4. Execution Loop
for name in "${!SUBSETS[@]}"; do
  read -r rel_pt1 rel_pt2 label1 label2 <<< "${SUBSETS[$name]}"
  
  # Map the relative paths to the internal /checkpoints bind
  pt1="/checkpoints/$rel_pt1"
  pt2="/checkpoints/$rel_pt2"

  echo "------------------------------------------------------------"
  echo "Running analysis for: $name"
  echo "Using weights: $pt1 and $pt2"
  echo "------------------------------------------------------------"

  apptainer exec --cleanenv \
    "${COMMON_BINDS[@]}" \
    "$APPTAINER_IMAGE" \
    bash -c "
      cd /workspace/mmdet/mmdetection3d
      # Force CPU usage via environment variable if the script doesn't have a --cpu flag
      export CUDA_VISIBLE_DEVICES=""
      
      python3 -u projects/analysis/scripts/main.py \
        --config /workspace/mmdet/mmdetection3d/configs/pointpillars/hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d.py \
        --pcd_dir /data/LIDAR_TOP/ \
        --pt $pt1 $pt2 \
        --labels $label1 $label2 \
        --max_samples 10 \
        --output_dir /workspace/out/$name \
        --run_name $name
    "
done

echo "DONE. Results in $OUT_ROOT_HOST"
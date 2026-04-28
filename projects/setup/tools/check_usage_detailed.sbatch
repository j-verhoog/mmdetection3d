#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --job-name=du_scan_max
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/tools/du_job_detailed_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/tools/du_job_detailed_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64   # Requests 64 physical cores
#SBATCH --mem=32G            # Bumped memory slightly to support 128 concurrent 'du' processes safely

TARGET_DIR="/tudelft.net/staff-umbrella/MscThesisjverhoog"
cd "$TARGET_DIR" || exit 1

echo "Starting MAX SPEED parallel disk usage scan at $(date)"
echo "Requested 64 CPUs. Running xargs with 128 threads."
echo "------------------------------------------------------"

# 1. Explicitly list ONLY the folders we want to scan to avoid wasting time
TARGET_FOLDERS=("datasets" "nusc_datasets" "nuscenes_shadow_root" "nuscenes_subsets_full" "yolo_nuscenes")

# 2. Find only the 1st-level subdirectories INSIDE those targets
# 3. Use -P 128 to utilize both hardware threads per requested CPU core
# 4. Sort and format into Human-Readable output on the fly
find "${TARGET_FOLDERS[@]}" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null | \
    xargs -0 -P 128 -n 1 du -sk 2>/dev/null | \
    sort -nr | \
    awk '{
        sum=$1;
        if (sum >= 1073741824) { printf "%.2f TB\t%s\n", sum/1073741824, $2 }
        else if (sum >= 1048576) { printf "%.2f GB\t%s\n", sum/1048576, $2 }
        else if (sum >= 1024) { printf "%.2f MB\t%s\n", sum/1024, $2 }
        else { printf "%d KB\t%s\n", sum, $2 }
    }'

echo "------------------------------------------------------"
echo "Finished scan at $(date)"
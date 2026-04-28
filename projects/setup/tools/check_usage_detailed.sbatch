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
RAW_OUT="~/raw_du_output.txt"
cd "$TARGET_DIR" || exit 1

echo "Starting MAX SPEED parallel disk usage scan at $(date)"
echo "Requested 64 CPUs. Running xargs with 128 threads."
echo "Excluding .pcd, .bin, and .jpg files."
echo "------------------------------------------------------"

# 1. Explicitly list ONLY the folders we want to scan
TARGET_FOLDERS=("datasets" "nusc_datasets" "nuscenes_shadow_root" "nuscenes_subsets_full" "yolo_nuscenes")

# Ensure our raw output file exists
touch "$RAW_OUT"

# 2. Setup Idempotency: Read already processed directories into an associative array
declare -A PROCESSED_MAP
if [[ -s "$RAW_OUT" ]]; then
    while read -r size dir; do
        PROCESSED_MAP["$dir"]=1
    done < "$RAW_OUT"
    echo "Found ${#PROCESSED_MAP[@]} directories already processed. Skipping these..."
fi

# 3. Find subdirectories and filter out the ones we've already processed
unset PENDING_DIRS
declare -a PENDING_DIRS

while IFS= read -r -d '' dir; do
    if [[ -z "${PROCESSED_MAP[$dir]}" ]]; then
        PENDING_DIRS+=("$dir")
    fi
done < <(find "${TARGET_FOLDERS[@]}" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)

# 4. Run the 128-thread scan ONLY on the pending directories
if [[ ${#PENDING_DIRS[@]} -eq 0 ]]; then
    echo "All directories have already been scanned! Moving to format..."
else
    echo "Scanning ${#PENDING_DIRS[@]} remaining directories..."
    # Print the array null-separated into xargs, and APPEND (>>) to the raw output
    printf "%s\0" "${PENDING_DIRS[@]}" | \
        xargs -0 -P 128 -n 1 du -sk --exclude="*.pcd" --exclude="*.bin" --exclude="*.jpg" 2>/dev/null >> "$RAW_OUT"
fi

echo "------------------------------------------------------"
echo "Generating human-readable summary from $RAW_OUT..."

# 5. Sort and format the accumulated data into Human-Readable output
cat "$RAW_OUT" | sort -nr | \
    awk '{
        sum=$1;
        if (sum >= 1073741824) { printf "%.2f TB\t%s\n", sum/1073741824, $2 }
        else if (sum >= 1048576) { printf "%.2f GB\t%s\n", sum/1048576, $2 }
        else if (sum >= 1024) { printf "%.2f MB\t%s\n", sum/1024, $2 }
        else { printf "%d KB\t%s\n", sum, $2 }
    }'

echo "------------------------------------------------------"
echo "Finished scan at $(date)"
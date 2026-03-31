#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --job-name=du_scan_part
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/tools/du_job_%A_%a.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/tools/du_job_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-13  # <--- UPDATE THIS: If you have 14 subfolders, use 0-13

# 1. Setup Directories
TARGET_DIR="/tudelft.net/staff-umbrella/MscThesisjverhoog"
LOG_DIR="/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/tools/parts"
mkdir -p "$LOG_DIR"

# 2. Get the list of top-level subdirectories into an array
# We use 'find' to get only directories at depth 1
mapfile -t SUBDIRS < <(find "$TARGET_DIR" -maxdepth 1 -mindepth 1 -type d)

# 3. Identify the specific folder for THIS array task
CURRENT_FOLDER="${SUBDIRS[$SLURM_ARRAY_TASK_ID]}"

# Safety check: if the array index is higher than the number of folders, exit
if [ -z "$CURRENT_FOLDER" ]; then
    echo "No folder assigned for index $SLURM_ARRAY_TASK_ID. Exiting."
    exit 0
fi

FOLDER_NAME=$(basename "$CURRENT_FOLDER")
PART_REPORT="$LOG_DIR/usage_${FOLDER_NAME}.txt"

echo "Starting scan for: $CURRENT_FOLDER"
echo "Output will be saved to: $PART_REPORT"

# 4. Run the scan
# We use -b (bytes) for more accurate sorting later, then convert to human-readable
du -b --max-depth=2 "$CURRENT_FOLDER" | sort -nr > "$PART_REPORT"

echo "Finished scan for $FOLDER_NAME at $(date)"


# combine with: cat /tudelft.net/staff-umbrella/MscThesisjverhoog/logs/tools/parts/usage_*.txt | sort -nr > final_disk_usage.txt
# readable with: head -n 20 final_disk_usage.txt | numfmt --to=iec --field=1
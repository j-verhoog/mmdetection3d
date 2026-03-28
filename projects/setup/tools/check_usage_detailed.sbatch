#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --job-name=disk_usage_scan
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/tools/disk_usage_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/tools/disk_usage_%j.err
#SBATCH --time=4:00:00 # Kept at 4 hours. If there are truly billions of files, this might need the 36hr medium queue.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 # du is single-threaded; more CPUs won't speed it up.
#SBATCH --mem=4G # 4GB is plenty for caching the directory trees during the scan.

# Define the target directory and where you want the final report saved
TARGET_DIR="/tudelft.net/staff-umbrella/MscThesisjverhoog"
REPORT_FILE="/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/tools/folder_sizes_${SLURM_JOB_ID}.txt"

echo "Starting massive disk usage scan at: $(date)"
echo "Target directory: $TARGET_DIR"
echo "This may take a long time depending on filesystem metadata limits..."

# The core command:
# du -h: Human readable formats (K, M, G, T)
# --max-depth=3: Only go down 3 levels for the summary
# sort -hr: Sorts the output from largest to smallest, understanding human-readable suffixes

du -h --max-depth=3 "$TARGET_DIR" | sort -hr > "$REPORT_FILE"

echo "Scan finished at: $(date)"
echo "Top 20 largest folders:"
head -n 20 "$REPORT_FILE"

echo "Full report saved to: $REPORT_FILE"
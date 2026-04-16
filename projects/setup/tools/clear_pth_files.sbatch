#!/bin/bash
#SBATCH --partition=general # request partition
#SBATCH --qos=short # Quality of service, short default
#SBATCH --job-name=remove_stuff
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/tools/remove_stuff_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/tools/remove_stuff_%j.err
#SBATCH --time=4:00:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# Define the target directory
TARGET_DIR="/home/nfs/jtverhoog/mmdet/MscThesisjverhoog/projects/cmt_fedavg_all/"
# TARGET_DIR="/home/nfs/jtverhoog/mmdet/MscThesisjverhoog/projects/cmt_40_epoch/"

echo "Starting checkpoint cleanup at: $(date)"
echo "Current Disk usage staff-umbrella:"
df -h /tudelft.net/staff-umbrella/MscThesisjverhoog

echo "Target directory: $TARGET_DIR"

# Use 'find' with specific regex/globbing to target exactly the files you want.
# -maxdepth 5: Restricts search depth
# -type f: Ensures we only look at files (not directories)
# -print0 + read -d '': Handles file paths safely


# make sure to exclude the centralized model runs!!
removed_count=0
while IFS= read -r -d '' file; do
    if rm -v "$file"; then
        ((removed_count++))
    fi
done < <(find "$TARGET_DIR" -maxdepth 8 -type f \( \
    -name "latest.pth" -o \
    -name "epoch_[0-9].pth" -o \
    -name "epoch_1[0-8].pth" \
\) -print0)

echo "Total files removed: $removed_count"

echo "Cleanup finished at: $(date)"
echo "Current Disk usage staff-umbrella:"
df -h /tudelft.net/staff-umbrella/MscThesisjverhoog
#!/bin/bash
#SBATCH --partition=general # request partition
#SBATCH --qos=short # Quality of service, short default
#SBATCH --job-name=copy_stuff
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/tools/copy_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/tools/copy_%j.err
#SBATCH --time=4:00:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

echo "Current Disk usage:"
du -h '/tudelft.net/staff-umbrella/MscThesisjverhoog/' | sort -h | tail

DEST="/tudelft.net/staff-umbrella/MscThesisjverhoog/"

echo "Starting rsync at $(date)"

# --exclude='MscThesisjverhoog' tells rsync to skip that specific entry inside ~/mmdet
# -a ensures other symlinks (like datasets) stay as links and aren't expanded
rsync -ahv \
    --exclude='MscThesisjverhoog' \
    ~/mmdet "$DEST"

echo "Rsync completed at $(date)"

echo "Now disk usage after rsync:"
du -h '/tudelft.net/staff-umbrella/MscThesisjverhoog/' | sort -h | tail

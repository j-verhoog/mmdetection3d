#!/bin/bash
#SBATCH --partition=general # request partition
#SBATCH --qos=short # Quality of service, short default
#SBATCH --job-name=rm_dir
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/nuscenes_data_prep/rm_dir.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/nuscenes_data_prep/rm_dir.err
#SBATCH --time=01:30:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# remove all old directories
TARGET="/home/nfs/jtverhoog/mmdet/MscThesisjverhoog/nusc_datasets"

if [ -d "$TARGET" ]; then
    echo "Removing $TARGET at $(date)"
    rm -rf "$TARGET"
    echo "Done at $(date)"
else
    echo "Target not found."
fi

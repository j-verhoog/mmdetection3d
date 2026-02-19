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

echo "Starting rsync at $(date)"
# Use -a to preserve symlinks and -h for human readable stats
# Ensure there is NO trailing slash on the source folder
rsync -ahv ~/mmdet /tudelft.net/staff-umbrella/MscThesisjverhoog/

echo "Rsync completed at $(date)"

echo "Now disk usage after rsync:"
du -h '/tudelft.net/staff-umbrella/MscThesisjverhoog/' | sort -h | tail

#!/bin/bash
#SBATCH --partition=general # request partition
#SBATCH --qos=short # Quality of service, short default
#SBATCH --job-name=du
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/tools/du_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/tools/du_%j.err
#SBATCH --time=00:01:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

echo "Current Disk usage staff-umbrella:"
df -h /tudelft.net/staff-umbrella/MscThesisjverhoog


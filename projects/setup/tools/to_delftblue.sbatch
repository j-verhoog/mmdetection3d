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

DEST="/tudelft.net/staff-umbrella/MscThesisjverhoog/homedir"

echo "Starting rsync at $(date)"
rsync -ahv \
    --exclude='.ssh' \
    --exclude='.conda' \
    --exclude='.cache' \
    --exclude='.local' \
    --exclude='.apptainer' \
    --exclude='.vscode-server' \
    --exclude='.nv' \
    --exclude='.bash_history' \
    ~/ "$DEST"

echo "Rsync completed at $(date)"

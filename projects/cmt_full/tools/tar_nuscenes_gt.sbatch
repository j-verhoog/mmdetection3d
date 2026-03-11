#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --job-name=tar_nusc
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/nuscenes_data_prep/tar_nusc_gt_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/nuscenes_data_prep/tar_nusc_gt_%j.err
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2    # Reduced from 8: uncompressed tar is single-threaded
#SBATCH --mem=8G

set -e # Fail fast on errors
echo "=== Starting Uncompressed Tar Job: $SLURM_JOB_ID ==="

# Define paths clearly
BASE_DIR="/tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/nuscenes_cmt_full"
TARGET_DIR="nuscenes_gt_database"
OUTPUT_TAR="$BASE_DIR/nuscenes_gt_database_2.tar"

# Move into the base directory so the tarball uses relative paths!
cd "$BASE_DIR"
echo "Working directory is now: $(pwd)"

echo "Archiving '$TARGET_DIR' into uncompressed file..."
# 'time' will print exactly how long the network I/O took
time tar -cf "$OUTPUT_TAR" "$TARGET_DIR"

echo "=== Archiving Complete ==="
ls -lh "$OUTPUT_TAR"
#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --job-name=tar_nusc_samples_sweeps
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/nuscenes_data_prep/tar_nusc_samples_sweeps_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/nuscenes_data_prep/tar_nusc_samples_sweeps_%j.err
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -e # Fail fast on errors
echo "=== Starting Heavy NuScenes Tar Job: $SLURM_JOB_ID ==="

# Define paths
BASE_DIR="/tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/nuscenes_cmt_full"
OUTPUT_TAR="$BASE_DIR/nuscenes_samples_sweeps.tar"

# Move into the base directory so the tarball uses clean relative paths
cd "$BASE_DIR"
echo "Working directory is now: $(pwd)"

echo "Archiving ONLY the requested samples and sweeps into $OUTPUT_TAR..."

# Feed the strict inclusion list into the tar command
time cat << 'EOF' | grep -v '^$' | tar -cf "$OUTPUT_TAR" -T -
samples/CAM_FRONT
samples/CAM_FRONT_RIGHT
samples/CAM_BACK_RIGHT
samples/CAM_BACK
samples/CAM_BACK_LEFT
samples/CAM_FRONT_LEFT
samples/LIDAR_TOP
sweeps/LIDAR_TOP
EOF

echo "=== Archiving Complete ==="
ls -lh "$OUTPUT_TAR"
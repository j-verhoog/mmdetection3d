#!/bin/bash
#SBATCH --partition=general # request partition
#SBATCH --qos=medium # Quality of service, short default
#SBATCH --job-name=cmt-splits
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/nuscenes_data_prep/cmt_splits_pt1.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/nuscenes_data_prep/cmt_splits_pt1.err
#SBATCH --time=01:30:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium. One takes 1:04hrs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G

# define image and work dirs
export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/MscThesisjverhoog/image/mmdet3d_v1rc5.sif

# temporary image dir to install openpyxl
mkdir -p /home/nfs/jtverhoog/mmdet/mmdetection3d/tmpfs

# REMEMBER TO EDIT HERE THE PYTHON FILE IF MOVING PLACES
echo "===GOING INTO CONTAINER==="
apptainer exec --cleanenv --overlay tmpfs \
--bind "$HOME/mmdet:/workspace/mmdet"   \
--bind "/tudelft.net/staff-umbrella/MscThesisjverhoog:/tudelft.net/staff-umbrella/MscThesisjverhoog"   \
--bind "/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets:/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets"   \
"$APPTAINER_IMAGE" \
bash -lc "
pip install openpyxl && 
python -c 'import openpyxl' &&
cd /workspace/mmdet/mmdetection3d && 
python projects/subsets_creation/full_subsets_nuscenes.py
"

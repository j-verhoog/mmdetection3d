#!/bin/bash
#SBATCH --partition=general # request partition
#SBATCH --qos=short # Quality of service, short default
#SBATCH --job-name=uniformsplits
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/nuscenes_data_prep/uniformsplits%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/nuscenes_data_prep/uniformsplits_%j.err
#SBATCH --time=4:00:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G


# define image and work dirs
export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/MscThesisjverhoog/image/mmdet3d_v1rc5.sif

# REMEMBER TO EDIT HERE THE PYTHON FILE IF MOVING PLACES
echo "===GOING INTO CONTAINER==="
apptainer exec --cleanenv --overlay tmpfs \
--bind "$HOME/mmdet:/workspace/mmdet"   \
--bind "/tudelft.net/staff-umbrella/MscThesisjverhoog:/tudelft.net/staff-umbrella/MscThesisjverhoog"   \
--bind "/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets:/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets"   \
"$APPTAINER_IMAGE" \
bash -lc "
python /workspace/mmdet/mmdetection3d/projects/cmt_40_epoch/fedavg/fairsplit/create_pkl_files.py
"

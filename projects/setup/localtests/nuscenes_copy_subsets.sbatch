#!/bin/bash

# define image and work dirs
export APPTAINER_IMAGE=/home/jolle/mmdet/image/mmdet3d_v1rc5.sif

# temporary image dir to install openpyxl
mkdir -p /home/jolle/mmdet/mmdetection3d/tmpfs

echo "===GOING INTO CONTAINER==="
apptainer exec --cleanenv \
--overlay /home/jolle/mmdet/mmdetection3d/tmpfs \
--bind "$HOME/mmdet:/workspace/mmdet"   \
--bind "/home/jolle/nuscenes_img_class/nuscenes/:/home/jolle/nuscenes_img_class/nuscenes/" \
"$APPTAINER_IMAGE" \
bash -lc "pip install openpyxl && python -c 'import openpyxl'
cd /workspace/mmdet/mmdetection3d && \
python projects/setup/full_subsets_nuscenes.py
"

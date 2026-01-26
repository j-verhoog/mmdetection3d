#!/bin/bash

# define image and work dirs
export APPTAINER_IMAGE=/home/jolle/mmdet/image/mmdet3d_v1rc5.sif
export WORK=/home/jolle/mmdet/scratch
mkdir -p "$WORK"

DATA_ROOT="/home/jolle/mmdet/nuscenes_subsets_full/singapore_night_clear"

echo "===GOING INTO CONTAINER==="
apptainer exec --cleanenv \
--bind "$HOME/mmdet:/workspace/mmdet"   \
--bind "/home/jolle/nuscenes_img_class/nuscenes/:/home/jolle/nuscenes_img_class/nuscenes/" \
--bind "/home/jolle/mmdet/nuscenes_subsets_full:/home/jolle/mmdet/nuscenes_subsets_full" \
"$APPTAINER_IMAGE" \
bash -lc "cd /workspace/mmdet/mmdetection3d && \
python tools/create_data.py nuscenes \
  --root-path $DATA_ROOT \
  --version v1.0 \
  --out-dir $DATA_ROOT \
  --extra-tag nuscenes \
  --workers 4
"

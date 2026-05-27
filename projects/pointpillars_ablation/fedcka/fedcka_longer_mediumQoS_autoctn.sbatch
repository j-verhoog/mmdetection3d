#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --job-name=2x_fedcka
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/pointpillars_ablation/fedcka_48epochs_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/pointpillars_ablation/fedcka_48epochs_%j.err
#SBATCH --time=36:00:00
#SBATCH --nodes=1            
#SBATCH --ntasks=2             # Total tasks (2 per node)
#SBATCH --ntasks-per-node=2    # Spread them evenly
#SBATCH --gres=gpu:a40:2       # Request 2 GPUs per node 
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -e # stop if something fails

# --- FEDERATED SETTINGS ---
NUM_ROUNDS=48       # approx 2hour per round of 1 Epoch
EPOCHS_PER_ROUND=1 
SAMPLES_PER_GPU=8
SIZE_A=265  # A MUST BE BIGGEST for logging
SIZE_B=125
SIZE_C=226
SIZE_D=71
SIZE_E=13

START_ROUND=1  # <--- NEW: Set to 1 for new run, or X to start loop at round X, round 16 mains load from global 15.pth
START_MODEL='A'

# Paths
DAY_CLEAR_DATAROOT="/tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/cmt_subsets/Default_NoFair_SingleClient/boston_day_clear"
DAY_RAIN_DATAROOT="/tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/cmt_subsets/Default_NoFair_SingleClient/boston_day_rain"
DAY_CLEAR_DATAROOT_SING="/tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/cmt_subsets/Default_NoFair_SingleClient/singapore_day_clear"
NIGHT_CLEAR_DATAROOT="/tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/cmt_subsets/Default_NoFair_SingleClient/singapore_night_clear"
NIGHT_RAIN_DATAROOT="/tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/cmt_subsets/Default_NoFair_SingleClient/singapore_night_rain"

CONFIG_A=$DAY_CLEAR_DATAROOT
CONFIG_B=$DAY_RAIN_DATAROOT
CONFIG_C=$DAY_CLEAR_DATAROOT_SING
CONFIG_D=$NIGHT_CLEAR_DATAROOT
CONFIG_E=$NIGHT_RAIN_DATAROOT

# --- STANDARD SETUP ---
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}
export WANDB_PROJECT="pointpillars_ablation"
export RUN_NAME="FedCKA_48epochs"

# Gets API key robustly
export WANDB_API_KEY=$(awk '$1=="machine" && $2=="api.wandb.ai" {found=1}found && $1=="password" {print $2; exit}' "$HOME/.netrc")

export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/MscThesisjverhoog/image/mmdet3d_v1rc5.sif
export WORK=/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/$WANDB_PROJECT/$RUN_NAME
mkdir -p "$WORK"

# fix dependencies by linking the _base_ configs to the project folder
[ -e ~/mmdet/mmdetection3d/projects/pointpillars_ablation/_base_ ] || \
ln -s ~/mmdet/mmdetection3d/configs/_base_ \
      ~/mmdet/mmdetection3d/projects/pointpillars_ablation/_base_
      
# ==========================================
# --- AUTO-DETECT RESUME STATE ---
# ==========================================
LATEST_ROUND=0
for (( i=1; i<=NUM_ROUNDS; i++ )); do
    if [ -d "$WORK/round_$i" ]; then
        LATEST_ROUND=$i
    fi
done

if [ "$LATEST_ROUND" -gt 0 ]; then
    STOP_EPOCH=$((LATEST_ROUND * EPOCHS_PER_ROUND))
    
    if [ -f "$WORK/round_${LATEST_ROUND}/merged_E.pth" ]; then
        # The round completely finished merging
        START_ROUND=$((LATEST_ROUND + 1))
        START_MODEL="A"
    elif [ -f "$WORK/round_${LATEST_ROUND}/ModelE/epoch_${STOP_EPOCH}.pth" ]; then
        # Model E finished, but merging timed out. 
        # Setting START_MODEL="merge" and bumping the round forces runner.sh to merge the previous round first.
        START_ROUND=$((LATEST_ROUND + 1))
        START_MODEL="merge"
    elif [ -f "$WORK/round_${LATEST_ROUND}/ModelD/epoch_${STOP_EPOCH}.pth" ]; then
        START_ROUND=$LATEST_ROUND
        START_MODEL="E"
    elif [ -f "$WORK/round_${LATEST_ROUND}/ModelC/epoch_${STOP_EPOCH}.pth" ]; then
        START_ROUND=$LATEST_ROUND
        START_MODEL="D"
    elif [ -f "$WORK/round_${LATEST_ROUND}/ModelB/epoch_${STOP_EPOCH}.pth" ]; then
        START_ROUND=$LATEST_ROUND
        START_MODEL="C"
    elif [ -f "$WORK/round_${LATEST_ROUND}/ModelA/epoch_${STOP_EPOCH}.pth" ]; then
        START_ROUND=$LATEST_ROUND
        START_MODEL="B"
    else
        START_ROUND=$LATEST_ROUND
        START_MODEL="A"
    fi
fi

echo "=== Auto-detected resume state: START_ROUND=$START_ROUND | START_MODEL=$START_MODEL ==="
# ==========================================

TMP_BASE="/tmp/jtverhoog_${SLURM_JOB_ID}"

# --- NODE SETUP ---
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 mkdir -p "$TMP_BASE"/{wandb,torch,xdg,tmp,home}
module use /opt/insy/modulefiles
module load cuda
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)


# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_instance_id,gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true)

# --- TRAINING FUNCTION ---
run_training () {
    local MODEL_NAME=$1
    local DATASET_PATH=$2
    local CHECKPOINT=$3
    local ROUND=$4
    local GLOBAL_EPOCHS=$5    # Changed from TARGET_EPOCHS
    local STOP_EPOCH=$6       # NEW ARGUMENT
    local IS_RESUME=$7
    local WANDB_RUN_ID=$8       # get run consistency
    
    local WORK_DIR="$WORK/round_${ROUND}/${MODEL_NAME}"
        
    # Configure Load vs Resume arguments
    local CFG_LOAD_ARG=""
    local RESUME_ARG=""
    
    if [ "$IS_RESUME" == "true" ]; then
        # In rounds > 1, we RESUME. This loads optimizer state and epoch count.
        RESUME_ARG="--resume-from ${CHECKPOINT}"
    else
        # In round 1, we LOAD weights only (backbone). Optimizer starts fresh.
        CFG_LOAD_ARG="load_from='${CHECKPOINT}'"
    fi

    echo "=== ROUND $ROUND | Training $MODEL_NAME | Stop Epoch: $STOP_EPOCH (Global: $GLOBAL_EPOCHS) ==="
    local MASTER_PORT=$(apptainer exec --cleanenv "$APPTAINER_IMAGE" python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    echo "Using Master Port: $MASTER_PORT"
    
    srun --ntasks=2 --gpus-per-task=1 bash -lc "
      apptainer exec --nv --cleanenv \
      --bind \"$HOME/mmdet:/workspace/mmdet\" \
      --bind \"$TMP_BASE:/workspace/tmp\" \
      --bind \"$WORK:/workspace/work_dirs\" \
      --bind \"/tudelft.net/staff-umbrella/MscThesisjverhoog:/tudelft.net/staff-umbrella/MscThesisjverhoog\" \
      --bind \"/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets:/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets\" \
      --bind \"$DATASET_PATH:/workspace/mmdet/mmdetection3d/data/nuscenes\" \
      --env SLURM_PROCID=\"\$SLURM_PROCID\" \
      --env SLURM_NTASKS=\"\$SLURM_NTASKS\" \
      --env SLURM_NODELIST=\"\$SLURM_NODELIST\" \
      --env SLURM_JOB_ID=\"\$SLURM_JOB_ID\" \
      --env TORCH_HOME=\"/workspace/tmp/torch/\$SLURM_PROCID\" \
      --env XDG_CACHE_HOME=\"/workspace/tmp/xdg/\$SLURM_PROCID\" \
      --env WANDB_DIR=\"/workspace/tmp/wandb/\$SLURM_PROCID\" \
      --env TMPDIR=\"/workspace/tmp/tmp/\$SLURM_PROCID\" \
      --env WANDB_PROJECT=\"$WANDB_PROJECT\" \
      --env WANDB_NAME=\"${RUN_NAME}_${MODEL_NAME}\" \
      --env WANDB_API_KEY=\"$WANDB_API_KEY\" \
      --env OMP_NUM_THREADS=1 \
      --env MKL_NUM_THREADS=1 \
      --env NCCL_DEBUG=INFO \
      --env NCCL_ASYNC_ERROR_HANDLING=1 \
      --env NCCL_BLOCKING_WAIT=1 \
      --env TORCH_DISTRIBUTED_DEBUG=DETAIL \
      --env MASTER_ADDR=\"$MASTER_ADDR\" \
      --env MASTER_PORT=\"$MASTER_PORT\" \
      --env NCCL_SOCKET_IFNAME=\"campus\" \
      --env WANDB_RUN_ID=\"$WANDB_RUN_ID\" \
      --env WANDB_RESUME=\"allow\" \
      \"$APPTAINER_IMAGE\" \
      bash -lc '
        cd /workspace/mmdet/mmdetection3d && \
        export PYTHONPATH=/workspace/mmdet/mmdetection3d:/workspace/mmdet:\$PYTHONPATH && \
        python -u tools/train.py \
        projects/pointpillars_ablation/fedcka/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py \
          --launcher slurm \
          --work-dir $WORK_DIR \
          $RESUME_ARG \
          --cfg-options ${CFG_LOAD_ARG} dist_params.port=$MASTER_PORT \
            data.samples_per_gpu=$SAMPLES_PER_GPU \
            evaluation.interval=1 \
            lr_config.warmup_iters=2000 \
            lr_config.warmup_ratio=0.0005 \
            lr_config.step=[40,46] \
            runner.max_epochs=$GLOBAL_EPOCHS \
            total_epochs=$GLOBAL_EPOCHS \
            custom_hooks.1.stop_epoch=$STOP_EPOCH \
            custom_hooks.2.mask_path=\"/workspace/work_dirs/fedselect_masks/${MODEL_NAME}_mask.pth\"
      '
    "
}

# --- MERGE FUNCTION ---
run_merge () {
    local CKPT_A=$1
    local CKPT_B=$2
    local CKPT_C=$3
    local CKPT_D=$4
    local CKPT_E=$5
    local OUT_A=$6
    local OUT_B=$7
    local OUT_C=$8
    local OUT_D=$9
    local OUT_E=${10}
    local CURRENT_RATIO=${11}
    local CURRENT_METHOD=${12}

    echo "=== Merging Models using $CURRENT_METHOD ==="
    
    # Base command string
    local MERGE_CMD="python projects/pointpillars_ablation/tools/modular_merging.py \
            --inputs $CKPT_A $CKPT_B $CKPT_C $CKPT_D $CKPT_E \
            --outputs $OUT_A $OUT_B $OUT_C $OUT_D $OUT_E \
            --weight-a $SIZE_A --weight-b $SIZE_B --weight-c $SIZE_C --weight-d $SIZE_D --weight-e $SIZE_E \
            --select-ratio $CURRENT_RATIO \
            --method $CURRENT_METHOD"
            
    # Append CKA specific arguments if the method is fedselect_cka
    if [ "$CURRENT_METHOD" = "fedselect_cka" ]; then
        MERGE_CMD="$MERGE_CMD \
            --runner-path projects/pointpillars_ablation/tools/scripts_cmt_copy_copy/runner.py \
            --data-dir /tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets/nuscenes \
            --modality lidar"
    fi

    apptainer exec --nv --cleanenv \
      --bind "$HOME/mmdet:/workspace/mmdet" \
      --bind "$WORK:/workspace/work_dirs" \
      --bind "/tudelft.net/staff-umbrella/MscThesisjverhoog:/tudelft.net/staff-umbrella/MscThesisjverhoog" \
      --bind "/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets:/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets" \
      "$APPTAINER_IMAGE" \
      bash -lc "
        cd /workspace/mmdet/mmdetection3d && \
        export PYTHONPATH=/workspace/mmdet/mmdetection3d:/workspace/mmdet:\$PYTHONPATH && \
        $MERGE_CMD
      "
}

# ==========================================
# MAIN LOOP
# ==========================================

# Create persistent WandB Run IDs so each model logs to the same run across rounds
export WANDB_ID_A="pp_fedcka_A_48epochs"
export WANDB_ID_B="pp_fedcka_B_48epochs"
export WANDB_ID_C="pp_fedcka_C_48epochs"
export WANDB_ID_D="pp_fedcka_D_48epochs"
export WANDB_ID_E="pp_fedcka_E_48epochs"


if [ "$START_ROUND" -eq 1 ]; then
    echo "Starting fresh run from Round 1"
    
    # Create the new initialization directory and define the path
    INIT_DIR="$WORK/shared_init_round_1"
    mkdir -p "$INIT_DIR"
    INIT_CKPT="$INIT_DIR/initial_random_weights.pth"
    
    echo "Generating shared random initialization weights..."
    apptainer exec --nv --cleanenv \
      --bind "$HOME/mmdet:/workspace/mmdet" \
      --bind "$WORK:/workspace/work_dirs" \
      --bind "/tudelft.net/staff-umbrella/MscThesisjverhoog:/tudelft.net/staff-umbrella/MscThesisjverhoog" \
      "$APPTAINER_IMAGE" \
      bash -lc "
        cd /workspace/mmdet/mmdetection3d && \
        export PYTHONPATH=/workspace/mmdet/mmdetection3d:/workspace/mmdet:\$PYTHONPATH && \
        python3 -c \"
from mmcv import Config
from mmdet3d.models import build_model
import torch

# 1. Load your exact config
cfg = Config.fromfile('projects/pointpillars_ablation/fedavg/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py')

# 2. Build the model using the older v1.0.0rc5 method
model = build_model(
    cfg.model,
    train_cfg=cfg.get('train_cfg'),
    test_cfg=cfg.get('test_cfg')
)

# 3. Trigger standard PyTorch weight initialization
model.init_weights()

# 4. Save just the state_dict (which is all mmcv needs for load_from)
torch.save({'state_dict': model.state_dict()}, '${INIT_CKPT}')
print('Shared initialization successfully saved to: ${INIT_CKPT}')
\"
      "

    # Assign this exact same starting checkpoint to all 5 clients
    MODEL_A="$INIT_CKPT"
    MODEL_B="$INIT_CKPT"
    MODEL_C="$INIT_CKPT"
    MODEL_D="$INIT_CKPT"
    MODEL_E="$INIT_CKPT"
else
    echo "Resuming run from Round $START_ROUND"
    PREV_ROUND=$((START_ROUND - 1))
    MODEL_A="$WORK/round_${PREV_ROUND}/merged_A.pth"
    MODEL_B="$WORK/round_${PREV_ROUND}/merged_B.pth"
    MODEL_C="$WORK/round_${PREV_ROUND}/merged_C.pth"
    MODEL_D="$WORK/round_${PREV_ROUND}/merged_D.pth"
    MODEL_E="$WORK/round_${PREV_ROUND}/merged_E.pth"
fi
# CALCULATE GLOBAL TOTAL ONCE
# This ensures the LR scheduler sees the full picture (e.g., 20 epochs) from Round 1
GLOBAL_MAX_EPOCHS=$((NUM_ROUNDS * EPOCHS_PER_ROUND))

for ((i=START_ROUND; i<=NUM_ROUNDS; i++)); do
    # This is where we want to STOP in this specific round
    CURRENT_STOP_EPOCH=$((i * EPOCHS_PER_ROUND))
    
    echo "-------------------------------------"
    echo "       STARTING FEDCKA ROUND $i      "
    echo "       Scheduler Horizon: $GLOBAL_MAX_EPOCHS epochs"
    echo "       Stop Training At:  $CURRENT_STOP_EPOCH epochs"
    echo "-------------------------------------"

    # Calculate the dynamic select_ratio for THIS round
    # Rounds 1-12: 0.00 (Pure FedAvg)
    # Rounds 13-36: Incrementally grows from >0.00 up to 0.40
    # Rounds >36: Capped at 0.40
    CURRENT_RATIO=$(awk -v r="$i" -v max_s="0.40" 'BEGIN { 
        if (r <= 12) {
            ratio = 0.0;
        } else if (r >= 37) {
            ratio = max_s;
        } else {
            # Calculate linear growth over the 24 active rounds (36 - 12 = 24)
            ratio = ((r - 12) / 24.0) * max_s;
        }
        printf "%.4f", ratio 
    }')

    echo "-------------------------------------"
    echo "       STARTING ROUND $i      "
    echo "       Target Cumulative Sparsity: $CURRENT_RATIO"
    echo "-------------------------------------"

    # --- PHASE merging DETERMINATION ---
    if [ "$i" -le 12 ]; then
        CURRENT_METHOD="fedavg"
    else
        CURRENT_METHOD="fedselect_cka"
    fi
    


    if [ "$i" -eq 1 ]; then
        IS_RESUME="false"
    else
        IS_RESUME="true"
    fi


    if [ "$START_MODEL" != "A" ]; then
        echo "Starting Round $i from Model $START_MODEL , then resuming normal training after that."
        if [ "$START_MODEL" = "B" ]; then
            CKPT_A="$WORK/round_${i}/ModelA/epoch_${CURRENT_STOP_EPOCH}.pth"
            run_training "ModelB" "$CONFIG_B" "$MODEL_B" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_B"
            CKPT_B="$WORK/round_${i}/ModelB/epoch_${CURRENT_STOP_EPOCH}.pth"
            run_training "ModelC" "$CONFIG_C" "$MODEL_C" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_C"
            CKPT_C="$WORK/round_${i}/ModelC/epoch_${CURRENT_STOP_EPOCH}.pth"
            run_training "ModelD" "$CONFIG_D" "$MODEL_D" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_D"
            CKPT_D="$WORK/round_${i}/ModelD/epoch_${CURRENT_STOP_EPOCH}.pth"
            run_training "ModelE" "$CONFIG_E" "$MODEL_E" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_E"
            CKPT_E="$WORK/round_${i}/ModelE/epoch_${CURRENT_STOP_EPOCH}.pth"  
        elif [ "$START_MODEL" = "C" ]; then
            CKPT_A="$WORK/round_${i}/ModelA/epoch_${CURRENT_STOP_EPOCH}.pth"
            CKPT_B="$WORK/round_${i}/ModelB/epoch_${CURRENT_STOP_EPOCH}.pth"    
            run_training "ModelC" "$CONFIG_C" "$MODEL_C" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_C"
            CKPT_C="$WORK/round_${i}/ModelC/epoch_${CURRENT_STOP_EPOCH}.pth"
            run_training "ModelD" "$CONFIG_D" "$MODEL_D" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_D"
            CKPT_D="$WORK/round_${i}/ModelD/epoch_${CURRENT_STOP_EPOCH}.pth"
            run_training "ModelE" "$CONFIG_E" "$MODEL_E" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_E"
            CKPT_E="$WORK/round_${i}/ModelE/epoch_${CURRENT_STOP_EPOCH}.pth"
        elif [ "$START_MODEL" = "D" ]; then
            CKPT_A="$WORK/round_${i}/ModelA/epoch_${CURRENT_STOP_EPOCH}.pth"
            CKPT_B="$WORK/round_${i}/ModelB/epoch_${CURRENT_STOP_EPOCH}.pth"    
            CKPT_C="$WORK/round_${i}/ModelC/epoch_${CURRENT_STOP_EPOCH}.pth"
            run_training "ModelD" "$CONFIG_D" "$MODEL_D" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_D"
            CKPT_D="$WORK/round_${i}/ModelD/epoch_${CURRENT_STOP_EPOCH}.pth"
            run_training "ModelE" "$CONFIG_E" "$MODEL_E" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_E"
            CKPT_E="$WORK/round_${i}/ModelE/epoch_${CURRENT_STOP_EPOCH}.pth"
        elif [ "$START_MODEL" = "E" ]; then
            CKPT_A="$WORK/round_${i}/ModelA/epoch_${CURRENT_STOP_EPOCH}.pth"
            CKPT_B="$WORK/round_${i}/ModelB/epoch_${CURRENT_STOP_EPOCH}.pth"    
            CKPT_C="$WORK/round_${i}/ModelC/epoch_${CURRENT_STOP_EPOCH}.pth"
            CKPT_D="$WORK/round_${i}/ModelD/epoch_${CURRENT_STOP_EPOCH}.pth"
            run_training "ModelE" "$CONFIG_E" "$MODEL_E" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_E"
            CKPT_E="$WORK/round_${i}/ModelE/epoch_${CURRENT_STOP_EPOCH}.pth"
       elif [ "$START_MODEL" = "merge" ]; then
            # Calculate correct previous round and its stop epoch
            PREV_RND=$((i - 1))
            PREV_STOP_EPOCH=$((CURRENT_STOP_EPOCH - EPOCHS_PER_ROUND))
            
            # --- ADDED: Calculate the ratio for the PREVIOUS round ---
            PREV_RATIO=$(awk -v r="$PREV_RND" -v max_s="0.40" 'BEGIN { 
                if (r <= 12) { ratio = 0.0; } 
                else if (r >= 37) { ratio = max_s; } 
                else { ratio = ((r - 12) / 24.0) * max_s; }
                printf "%.4f", ratio 
            }')

            if [ "$PREV_RND" -le 12 ]; then
                PREV_METHOD="fedavg"
            else
                PREV_METHOD="fedselect_cka"
            fi

            CKPT_A="$WORK/round_${PREV_RND}/ModelA/epoch_${PREV_STOP_EPOCH}.pth"
            CKPT_B="$WORK/round_${PREV_RND}/ModelB/epoch_${PREV_STOP_EPOCH}.pth"
            CKPT_C="$WORK/round_${PREV_RND}/ModelC/epoch_${PREV_STOP_EPOCH}.pth"
            CKPT_D="$WORK/round_${PREV_RND}/ModelD/epoch_${PREV_STOP_EPOCH}.pth"
            CKPT_E="$WORK/round_${PREV_RND}/ModelE/epoch_${PREV_STOP_EPOCH}.pth"
            
            # 3. Aggregate and Save 5 Distinct Models for the PREVIOUS round
            MERGED_A="$WORK/round_${PREV_RND}/merged_A.pth"
            MERGED_B="$WORK/round_${PREV_RND}/merged_B.pth"
            MERGED_C="$WORK/round_${PREV_RND}/merged_C.pth"
            MERGED_D="$WORK/round_${PREV_RND}/merged_D.pth"
            MERGED_E="$WORK/round_${PREV_RND}/merged_E.pth"
            
            run_merge "$CKPT_A" "$CKPT_B" "$CKPT_C" "$CKPT_D" "$CKPT_E" \
                    "$MERGED_A" "$MERGED_B" "$MERGED_C" "$MERGED_D" "$MERGED_E" "$PREV_RATIO" "$PREV_METHOD"
            
            # Now run Round 2 training 
            run_training "ModelA" "$CONFIG_A" "$MODEL_A" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_A"
            CKPT_A="$WORK/round_${i}/ModelA/epoch_${CURRENT_STOP_EPOCH}.pth"

            run_training "ModelB" "$CONFIG_B" "$MODEL_B" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_B"
            CKPT_B="$WORK/round_${i}/ModelB/epoch_${CURRENT_STOP_EPOCH}.pth"

            run_training "ModelC" "$CONFIG_C" "$MODEL_C" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_C"
            CKPT_C="$WORK/round_${i}/ModelC/epoch_${CURRENT_STOP_EPOCH}.pth"

            run_training "ModelD" "$CONFIG_D" "$MODEL_D" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_D"
            CKPT_D="$WORK/round_${i}/ModelD/epoch_${CURRENT_STOP_EPOCH}.pth"

            run_training "ModelE" "$CONFIG_E" "$MODEL_E" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_E"
            CKPT_E="$WORK/round_${i}/ModelE/epoch_${CURRENT_STOP_EPOCH}.pth"

            if [ "$PREV_RND" -eq 12 ]; then
                echo "Transitioning to FedSelect after resumed Round 12 merge. Saving baseline global model..."
                mkdir -p "$WORK/fedselect_states"
                cp "$MERGED_A" "$WORK/fedselect_states/global_model.pth"
            fi

        else
            echo "Invalid START_MODEL specified: $START_MODEL. Must be one of A, B, C, D, E."
            exit 1
        fi
        START_MODEL="A" # Reset to normal training for next rounds
    else
        # UPDATE: Pass both GLOBAL total and LOCAL stop to the training function
        # Note: I am updating the function call arguments below
        run_training "ModelA" "$CONFIG_A" "$MODEL_A" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_A"
        CKPT_A="$WORK/round_${i}/ModelA/epoch_${CURRENT_STOP_EPOCH}.pth" # Use specific epoch

        run_training "ModelB" "$CONFIG_B" "$MODEL_B" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_B"
        CKPT_B="$WORK/round_${i}/ModelB/epoch_${CURRENT_STOP_EPOCH}.pth" # Use specific epoch

        run_training "ModelC" "$CONFIG_C" "$MODEL_C" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_C"
        CKPT_C="$WORK/round_${i}/ModelC/epoch_${CURRENT_STOP_EPOCH}.pth" # Use specific epoch

        run_training "ModelD" "$CONFIG_D" "$MODEL_D" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_D"
        CKPT_D="$WORK/round_${i}/ModelD/epoch_${CURRENT_STOP_EPOCH}.pth" # Use specific epoch

        run_training "ModelE" "$CONFIG_E" "$MODEL_E" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_E"
        CKPT_E="$WORK/round_${i}/ModelE/epoch_${CURRENT_STOP_EPOCH}.pth" # Use specific epoch

    fi



    # 3. Aggregate and Save 5 Distinct Models
    MERGED_A="$WORK/round_${i}/merged_A.pth"
    MERGED_B="$WORK/round_${i}/merged_B.pth"
    MERGED_C="$WORK/round_${i}/merged_C.pth"
    MERGED_D="$WORK/round_${i}/merged_D.pth"
    MERGED_E="$WORK/round_${i}/merged_E.pth"
    
    run_merge "$CKPT_A" "$CKPT_B" "$CKPT_C" "$CKPT_D" "$CKPT_E" \
              "$MERGED_A" "$MERGED_B" "$MERGED_C" "$MERGED_D" "$MERGED_E" "$CURRENT_RATIO" "$CURRENT_METHOD"
    
    # Setup for next round
    MODEL_A="$MERGED_A"
    MODEL_B="$MERGED_B"
    MODEL_C="$MERGED_C"
    MODEL_D="$MERGED_D"
    MODEL_E="$MERGED_E"
    
    # --- FEDSELECT HANDOVER LOGIC ---
    # If we just finished Round 12, save Model A (since all are identical after FedAvg) 
    # as the baseline global model for FedSelect in Round 13
    if [ "$i" -eq 12 ]; then
        echo "Transitioning to FedSelect next round. Saving baseline global model..."
        mkdir -p "$WORK/fedselect_states"
        cp "$MERGED_A" "$WORK/fedselect_states/global_model.pth"
    fi

    echo "Round $i complete. 5 distinct merged models saved."
done

# Measure GPU usage of your job (result) for all allocated GPUs
/usr/bin/nvidia-smi --query-accounted-apps='gpu_instance_id,gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true

# Cleanup
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 rm -rf "$TMP_BASE"
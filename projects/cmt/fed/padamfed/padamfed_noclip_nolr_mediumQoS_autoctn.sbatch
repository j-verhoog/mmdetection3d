#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --job-name=3PAdaMFed
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/cmt/padamfed_noclip_nolr_mediumQoS_autoctn_job_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/cmt/padamfed_noclip_nolr_mediumQoS_autoctn_job_%j.err
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -e # stop if something fails

# --- FEDERATED SETTINGS ---
NUM_ROUNDS=20       # approx 2hour per round of 1 Epoch
EPOCHS_PER_ROUND=1 
SAMPLES_PER_GPU=16
SIZE_A=265  # A MUST BE BIGGEST for logging
SIZE_B=125
SIZE_C=226
SIZE_D=71
SIZE_E=13

START_ROUND=1
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

# Initial backbone weights (only used in Round 1)
PRETRAINED_BACKBONE="ckpts/nuim_r50.pth"

# --- STANDARD SETUP ---
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}
export WANDB_PROJECT="cmt_fedavg_all"
export RUN_NAME="PAdaMFed_test_noclip_nolr"

# Gets API key robustly
export WANDB_API_KEY=$(awk '$1=="machine" && $2=="api.wandb.ai" {found=1}found && $1=="password" {print $2; exit}' "$HOME/.netrc")

export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/MscThesisjverhoog/image/mmdet3d_v1rc5.sif
export WORK=/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/$WANDB_PROJECT/$RUN_NAME
mkdir -p "$WORK"

CTRL_A="$WORK/PAdaMFed_VR/client_states/ModelA_control_variate.pth"
CTRL_B="$WORK/PAdaMFed_VR/client_states/ModelB_control_variate.pth"
CTRL_C="$WORK/PAdaMFed_VR/client_states/ModelC_control_variate.pth"
CTRL_D="$WORK/PAdaMFed_VR/client_states/ModelD_control_variate.pth"
CTRL_E="$WORK/PAdaMFed_VR/client_states/ModelE_control_variate.pth"
STEPS_ModelA=887
STEPS_ModelB=337
STEPS_ModelC=587
STEPS_ModelD=116
STEPS_ModelE=25      # at least 20, for at 20 it logs (has logged 17 things, and logs 6 after)

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
    local STEP_VAR="STEPS_${MODEL_NAME}"
    local LOCAL_STEPS_VALUE="${!STEP_VAR}"

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
      --env WANDB_NAME=\"PAdaMFed_noClipLR_${MODEL_NAME}\" \
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
        projects/cmt/fed/padamfed/improved_lightweight_cmt_iterated.py \
          --launcher slurm \
          --work-dir $WORK_DIR \
          $RESUME_ARG \
          --cfg-options ${CFG_LOAD_ARG} dist_params.port=$MASTER_PORT \
            data.samples_per_gpu=$SAMPLES_PER_GPU \
            total_epochs=$GLOBAL_EPOCHS \
            custom_hooks.1.stop_epoch=$STOP_EPOCH \
            optimizer_config.client_id=\"${MODEL_NAME}\" \
            optimizer_config.local_steps_per_round=${LOCAL_STEPS_VALUE}
      '
    "
}

# --- MERGE FUNCTION ---
run_merge () {
    local ROUND_IDX=$1

    local CKPT_A=$2
    local CKPT_B=$3
    local CKPT_C=$4
    local CKPT_D=$5
    local CKPT_E=$6

    local CTRL_A=$7
    local CTRL_B=$8
    local CTRL_C=$9
    local CTRL_D=${10}
    local CTRL_E=${11}

    local OUT_A=${12}
    local OUT_B=${13}
    local OUT_C=${14}
    local OUT_D=${15}
    local OUT_E=${16}

    local STEPS_A=${17}
    local STEPS_B=${18}
    local STEPS_C=${19}
    local STEPS_D=${20}
    local STEPS_E=${21}

    echo "=== Merging Models for round ${ROUND_IDX} ==="
    echo "Local steps: A=${STEPS_A}, B=${STEPS_B}, C=${STEPS_C}, D=${STEPS_D}, E=${STEPS_E}"

    apptainer exec --nv --cleanenv \
      --bind "$HOME/mmdet:/workspace/mmdet" \
      --bind "$WORK:/workspace/work_dirs" \
      --bind "/tudelft.net/staff-umbrella/MscThesisjverhoog:/tudelft.net/staff-umbrella/MscThesisjverhoog" \
      "$APPTAINER_IMAGE" \
      bash -lc "
        cd /workspace/mmdet/mmdetection3d && \
        export PYTHONPATH=/workspace/mmdet/mmdetection3d:/workspace/mmdet:\$PYTHONPATH && \
        python projects/cmt/fed/tools/PAdaMFed.py \
          --inputs \"$CKPT_A\" \"$CKPT_B\" \"$CKPT_C\" \"$CKPT_D\" \"$CKPT_E\" \
          --outputs \"$OUT_A\" \"$OUT_B\" \"$OUT_C\" \"$OUT_D\" \"$OUT_E\" \
          --client-control-paths \"$CTRL_A\" \"$CTRL_B\" \"$CTRL_C\" \"$CTRL_D\" \"$CTRL_E\" \
          --client-ids ModelA ModelB ModelC ModelD ModelE \
          --round-idx \"$ROUND_IDX\" \
          --total-rounds \"$NUM_ROUNDS\" \
          --local-steps-a \"$STEPS_A\" \
          --local-steps-b \"$STEPS_B\" \
          --local-steps-c \"$STEPS_C\" \
          --local-steps-d \"$STEPS_D\" \
          --local-steps-e \"$STEPS_E\" \
          --total-num-clients 5 \
          --num-sampled-clients 5 \
          --weight-a \"$SIZE_A\" \
          --weight-b \"$SIZE_B\" \
          --weight-c \"$SIZE_C\" \
          --weight-d \"$SIZE_D\" \
          --weight-e \"$SIZE_E\"
      "
}

# ==========================================
# MAIN LOOP
# ==========================================

# Create persistent WandB Run IDs so each model logs to the same run across rounds
export WANDB_ID_A="padamfed_A_test_noclip_nolr"
export WANDB_ID_B="padamfed_B_test_noclip_nolr"
export WANDB_ID_C="padamfed_C_test_noclip_nolr"
export WANDB_ID_D="padamfed_D_test_noclip_nolr"
export WANDB_ID_E="padamfed_E_test_noclip_nolr"

if [ "$START_ROUND" -eq 1 ]; then
    echo "Starting fresh run from Round 1"
    MODEL_A="$PRETRAINED_BACKBONE"
    MODEL_B="$PRETRAINED_BACKBONE"
    MODEL_C="$PRETRAINED_BACKBONE"
    MODEL_D="$PRETRAINED_BACKBONE"
    MODEL_E="$PRETRAINED_BACKBONE"
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
      echo "       STARTING ROUND $i      "
      echo "       Scheduler Horizon: $GLOBAL_MAX_EPOCHS epochs"
      echo "       Stop Training At:  $CURRENT_STOP_EPOCH epochs"
      echo "-------------------------------------"

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
            
            run_merge "$PREV_RND" \
            "$CKPT_A" "$CKPT_B" "$CKPT_C" "$CKPT_D" "$CKPT_E" \
            "$CTRL_A" "$CTRL_B" "$CTRL_C" "$CTRL_D" "$CTRL_E" \
            "$MERGED_A" "$MERGED_B" "$MERGED_C" "$MERGED_D" "$MERGED_E" \
            "$STEPS_ModelA" "$STEPS_ModelB" "$STEPS_ModelC" "$STEPS_ModelD" "$STEPS_ModelE"
            
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

        else
            echo "Invalid START_MODEL specified: $START_MODEL. Must be one of A, B, C, D, E, merge."
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
    
    run_merge "$i" \
    "$CKPT_A" "$CKPT_B" "$CKPT_C" "$CKPT_D" "$CKPT_E" \
    "$CTRL_A" "$CTRL_B" "$CTRL_C" "$CTRL_D" "$CTRL_E" \
    "$MERGED_A" "$MERGED_B" "$MERGED_C" "$MERGED_D" "$MERGED_E" \
    "$STEPS_ModelA" "$STEPS_ModelB" "$STEPS_ModelC" "$STEPS_ModelD" "$STEPS_ModelE"
    
    # Setup for next round
    MODEL_A="$MERGED_A"
    MODEL_B="$MERGED_B"
    MODEL_C="$MERGED_C"
    MODEL_D="$MERGED_D"
    MODEL_E="$MERGED_E"
    
    echo "Round $i complete. 5 distinct merged models saved."
done

# Measure GPU usage of your job (result) for all allocated GPUs
/usr/bin/nvidia-smi --query-accounted-apps='gpu_instance_id,gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true

# Cleanup
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 rm -rf "$TMP_BASE"
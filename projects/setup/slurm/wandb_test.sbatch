#!/bin/bash
#SBATCH --partition=general # request partition
#SBATCH --qos=short # Quality of service, short default
#SBATCH --job-name=wandb_test
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/wandb_test%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/wandb_test%j.err
#SBATCH --time=00:10:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

export WANDB_PROJECT="test"
export WANDB_NAME="wandb_test"
export WANDB_API_KEY=$(awk '$1=="machine" && $2=="api.wandb.ai" {found=1}found && $1=="password" {print $2; exit}' "$HOME/.netrc")

/usr/bin/scontrol show job -d "$SLURM_JOB_ID" # check sbatch working

# define image and work dirs
export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/MscThesisjverhoog/image/mmdet3d_v1rc5.sif
export WORK=/tudelft.net/staff-umbrella/MscThesisjverhoog/scratch/testing_stuff
mkdir -p "$WORK"

TMP_BASE="/tmp/jtverhoog_${SLURM_JOB_ID}"
mkdir -p "$TMP_BASE"/{wandb,torch,xdg,tmp,home}

apptainer exec --cleanenv --contain \
--home /workspace/tmp/home \
--bind "$HOME/mmdet:/workspace/mmdet"   \
--bind "$WORK:/workspace/work_dirs" \
--bind "$TMP_BASE:/workspace/tmp" \
--env TMPDIR=/workspace/tmp/tmp \
--env TORCH_HOME=/workspace/tmp/torch   \
--env XDG_CACHE_HOME=/workspace/tmp/xdg   \
--env WANDB_DIR=/workspace/tmp/wandb   \
--env WANDB_NAME="$WANDB_NAME" \
--env WANDB_API_KEY="$WANDB_API_KEY" \
"$APPTAINER_IMAGE" \
bash -lc 'python -c "import wandb; wandb.init(project=\"'$WANDB_PROJECT'\", name=\"'$WANDB_NAME'\"); wandb.log({\"ok\":1}); wandb.finish()"'

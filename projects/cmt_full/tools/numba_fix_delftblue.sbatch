#!/bin/bash
#SBATCH --job-name=mmdet_cmt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1              # Request 1 GPU (change to gpu:2 if you need more)
#SBATCH --time=04:00:00           # Adjust time limit as needed
#SBATCH --cpus-per-task=4         # Request CPU cores for data loading
#SBATCH --mem=32GB                # Request system memory
#SBATCH --output=slurm-%j.out

# --- 1. DelftBlue Best Practices: Use /scratch, not /tmp ---
export BASE=/scratch/$USER/cmt
export APPTAINER_CACHEDIR=/scratch/$USER/.apptainer/cache

# Ensure the cache directory exists
mkdir -p $APPTAINER_CACHEDIR
mkdir -p $BASE/{CMT,ckpts,workdirs,datasets,code}

# --- 2. Load DelftBlue Modules ---
# Load the current default stack, then CUDA
module load 2024r1  # or 'module load 2025' depending on the current active default
module load cuda
module list

# --- 3. Keep new numpy/numba/llvmlite; reinstall mmdet3d without pins ---
export APPTAINERENV_PYTHONFAULTHANDLER=1
export APPTAINERENV_TORCH_SHOW_CPP_STACKTRACES=1
export APPTAINERENV_CUDA_LAUNCH_BLOCKING=1

apptainer exec --nv \
  -B $BASE/CMT:/opt/src/mmdetection3d/projects/CMT \
  -B $BASE/ckpts:/opt/src/mmdetection3d/ckpts \
  -B $BASE/workdirs:/opt/src/mmdetection3d/work_dirs \
  -B $BASE/datasets:/opt/src/mmdetection3d/data \
  -B $BASE/code:/opt/src/code \
  $BASE/cmt_mmdet3d.sif bash -lc '

echo "=== Reinstall mmdet3d without enforcing old numba pins ==="
pip install --no-cache-dir --force-reinstall --no-deps mmdet3d==1.0.0rc5

echo "=== Verify numpy / numba / llvmlite ==="
python - <<PY
import importlib, numpy, numba, llvmlite
print("numpy:", numpy.__version__)
print("numba:", numba.__version__, "llvmlite:", llvmlite.__version__)
importlib.import_module("numba.np.ufunc._internal")
print("numba._internal import OK")
PY

echo "=== Test CMT plugin import ==="
python - <<PY
import importlib, traceback
mod = "projects.CMT.projects.mmdet3d_plugin"
print("Trying:", mod)
try:
    importlib.import_module(mod)
    print("CMT plugin import OK")
except Exception as e:
    print("FAIL:", mod, "->", type(e).__name__, e)
    traceback.print_exc()
PY
'

# (Add your actual training command here once the checks pass)
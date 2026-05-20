"""
Common utilities for model comparison:
- Hooks for extracting layer outputs
- Data loading and inference
- Visualization helpers
"""

import os
import re
import sys
import torch
import torch.nn as nn
from typing import Dict, Tuple, Any, List, Callable
from torch.nn.modules.batchnorm import _BatchNorm
from tqdm import tqdm

# Import from parent directory
try:
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmdet3d.apis import init_model 
    from mmdet3d.datasets.pipelines import Compose
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
except ImportError as e:
    print(f"FATAL IMPORT ERROR: {e}")
    sys.exit(1)


class ForwardHook:
    """
    Generic hook for capturing layer outputs during forward pass.
    Automatically extracts tensor from complex structures.
    """
    
    def __init__(self):
        self.output = None
        self.module_ref = None
    
    def __call__(self, module, input, output):
        """Called during forward pass."""
        self.module_ref = module
        
        if output is None:
            self.output = None
            return
        
        # Unwrap structures
        if isinstance(output, (list, tuple)):
            output = output[0] if len(output) > 0 else None
        
        if output is None:
            self.output = None
            return
        
        # Handle custom tensor wrappers
        if hasattr(output, "features"):
            output = output.features
        
        # Store as tensor
        if torch.is_tensor(output):
            self.output = output.detach().float().cpu()
        else:
            self.output = None
    
    def get_output(self) -> torch.Tensor:
        """Get captured output (or None)."""
        return self.output


def load_model_and_pipeline(config_path: str, checkpoint_path: str):
    """
    Load a model and its data pipeline.
    
    Sets model to eval mode and disables BatchNorm running stats updates
    to ensure no model modification during comparison.
    
    Args:
        config_path: Path to config file
        checkpoint_path: Path to checkpoint
        
    Returns:
        Tuple of (model, pipeline)
    """
    print(f"Loading weights: {os.path.basename(checkpoint_path)} ...")
    model = init_model(config_path, checkpoint_path, device='cuda:0')
    model.eval()
    
    cfg = Config.fromfile(config_path)
    test_pipeline = Compose(cfg.data.test.pipeline)
    return model, test_pipeline


def register_hooks(
    model: nn.Module,
    filter_func: Callable[[str, nn.Module], bool]
) -> Tuple[Dict[str, ForwardHook], List[torch.utils.hooks.RemovableHandle]]:
    """
    Register hooks on model layers matching filter criteria.
    
    Args:
        model: PyTorch model
        filter_func: Function(layer_name, module) -> bool to select layers
        
    Returns:
        Tuple of (hooks_dict, handles_list)
    """
    hooks = {}
    handles = []
    
    for name, module in model.named_modules():
        if filter_func(name, module):
            h = ForwardHook()
            handles.append(module.register_forward_hook(h))
            hooks[name] = h
    
    return hooks, handles

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Any, List, Callable
from torch.nn.modules.batchnorm import _BatchNorm
from tqdm import tqdm

# --- NEW IMPORTS FOR NUSCENES ---
try:
    from nuscenes.nuscenes import NuScenes
except ImportError:
    print("Warning: 'nuscenes-devkit' not found. Camera comparison will require dummy intrinsics.")

# --- LAZY LOADING HELPERS TO AVOID RELOADING DB ---
_NUSC_INSTANCE = None
_NUSC_FILE_INDEX = None

def _get_nuscenes_metadata(file_path: str):
    """
    Singleton helper to load NuScenes DB once and fetch intrinsics for a file.
    Automatically finds the v1.0-mini root directory from the file path.
    """
    global _NUSC_INSTANCE, _NUSC_FILE_INDEX

    # 1. Try to find the v1.0-mini root directory from the input file path
    if _NUSC_INSTANCE is None:
        # Walk up directories until we find 'v1.0-mini'
        current_dir = os.path.dirname(file_path)
        dataroot = None
        for _ in range(10): # Look up to 10 levels
            if os.path.basename(current_dir) == 'v1.0-mini':
                dataroot = current_dir
                break
            if os.path.exists(os.path.join(current_dir, 'v1.0-mini')):
                dataroot = os.path.join(current_dir, 'v1.0-mini')
                break
            current_dir = os.path.dirname(current_dir)
        
        if dataroot is None:
            print(f"Warning: Could not locate v1.0-mini root for {file_path}. Using dummy intrinsics.")
            return None

        print(f"Initializing NuScenes DB from: {dataroot} ...")
        _NUSC_INSTANCE = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=False)
        
        # 2. Build a quick lookup index (filename suffix -> sample_data record)
        # NuScenes stores paths like 'samples/CAM_FRONT/image.jpg'
        # We index by the filename only (e.g., 'image.jpg') to match easily
        _NUSC_FILE_INDEX = {}
        for record in _NUSC_INSTANCE.sample_data:
            fname = os.path.basename(record['filename'])
            _NUSC_FILE_INDEX[fname] = record

    # 3. Look up the specific file
    filename_only = os.path.basename(file_path)
    if _NUSC_FILE_INDEX and filename_only in _NUSC_FILE_INDEX:
        sd_record = _NUSC_FILE_INDEX[filename_only]
        cs_record = _NUSC_INSTANCE.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        return np.array(cs_record['camera_intrinsic'], dtype=np.float32)
    
    return None

# --- MAIN FUNCTION ---

def process_single_sample(
    model: nn.Module,
    pipeline: Any, # Typed as Any to avoid import issues if mmdet not present
    sample_path: str,
    hooks: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    """
    Run inference on a single sample and collect hooked layer outputs.
    Automatically fetches real intrinsics from NuScenes DB if available.
    """
    from mmdet3d.core.bbox import LiDARInstance3DBoxes, CameraInstance3DBoxes
    from mmcv.parallel import collate, scatter

    # 1. Detect Modality
    ext = os.path.splitext(sample_path)[-1].lower()
    is_lidar = ext in ['.bin', '.pcd', '.ply']
    is_camera = ext in ['.jpg', '.jpeg', '.png']

    if not (is_lidar or is_camera):
        return {}

    # 2. Initialize Data Dict
    data = dict(
        timestamp=0.0,
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[]
    )

    if is_lidar:
        data['pts_filename'] = sample_path
        data['sweeps'] = []
    else:
        # Camera Setup
        data['img_filename'] = sample_path
        data['img_prefix'] = ''
        data['filename'] = sample_path
        data['ori_filename'] = sample_path
        
        # --- FETCH REAL INTRINSICS ---
        real_intrinsic = _get_nuscenes_metadata(sample_path)
        
        if real_intrinsic is not None:
            # Use REAL nuScenes data
            intrinsic_matrix = real_intrinsic
        else:
            # Fallback to dummy if lookup fails (prevents crash)
            intrinsic_matrix = np.array([
                [1266.4, 0.0, 816.2],
                [0.0, 1266.4, 491.5],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)

        data['img_info'] = dict(
            filename=sample_path,
            cam_intrinsic=intrinsic_matrix,
            width=1600,
            height=900,
        )

    # 3. Run Pipeline
    data = pipeline(data)
    if data is None: 
        return {}
        
    data = collate([data], samples_per_gpu=1)
    
    # Handle GPU placement
    device = next(model.parameters()).device
    if device.type == 'cuda':
        scattered = scatter(data, [device.index])[0]
    else:
        scattered = scatter(data, [-1])[0]

    if isinstance(scattered, list):
        final_data = scattered[0]
    else:
        final_data = scattered

    # 4. Inject Box Type Metadata
    if 'img_metas' in final_data:
        metas = final_data['img_metas']
        if hasattr(metas, 'data'): metas = metas.data
        if isinstance(metas, list) and len(metas) > 0:
            target = metas[0][0] if isinstance(metas[0], list) else metas[0]
            if isinstance(target, dict) and 'box_type_3d' not in target:
                target['box_type_3d'] = LiDARInstance3DBoxes if is_lidar else CameraInstance3DBoxes

    # 5. Inference
    with torch.no_grad():
        model(return_loss=False, rescale=True, **final_data)
    
    return {name: h.get_output() for name, h in hooks.items() if h.get_output() is not None}

# def process_single_sample(
#     model: nn.Module,
#     pipeline: Compose,
#     sample_path: str,
#     hooks: Dict[str, ForwardHook]
# ) -> Dict[str, torch.Tensor]:
#     """
#     Run inference on a single sample (LiDAR or Camera) and collect hooked layer outputs.
#     Automatically detects modality based on file extension.
    
#     Args:
#         model: Model to run inference
#         pipeline: Data pipeline
#         sample_path: Path to sample file (.bin for LiDAR, .jpg/.png for Camera)
#         hooks: Dict of hooks to collect outputs
        
#     Returns:
#         Dict mapping layer names to output tensors (only non-None outputs)
#     """
#     import os
#     from mmdet3d.core.bbox import LiDARInstance3DBoxes, CameraInstance3DBoxes

#     # 1. Detect Modality and Prepare Data Dictionary
#     ext = os.path.splitext(sample_path)[-1].lower()
#     is_lidar = ext in ['.bin', '.pcd', '.ply']
#     is_camera = ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

#     if not (is_lidar or is_camera):
#         print(f"Skipping unsupported file: {sample_path}")
#         return {}

#     # Initialize base keys required by most MM pipelines
#     data = dict(
#         timestamp=0.0,
#         img_fields=[],
#         bbox3d_fields=[],
#         pts_mask_fields=[],
#         pts_seg_fields=[],
#         bbox_fields=[],
#         mask_fields=[],
#         seg_fields=[]
#     )

#     if is_lidar:
#         # LiDAR-specific keys
#         data['pts_filename'] = sample_path
#         data['sweeps'] = []  # Required by LoadPointsFromMultiSweeps
#     else:
#         # Camera-specific keys (Required by LoadImageFromFile)
#         data['img_filename'] = sample_path
#         data['img_prefix'] = ''
#         data['img_info'] = dict(filename=sample_path)
#         data['filename'] = sample_path
#         data['ori_filename'] = sample_path

#     # 2. Run Pipeline & Collate
#     data = pipeline(data)
#     if data is None:
#         print(f"Pipeline failed for {sample_path}")
#         return {}

#     data = collate([data], samples_per_gpu=1)
    
#     # Handle GPU placement
#     if next(model.parameters()).is_cuda:
#         scattered = scatter(data, [next(model.parameters()).device.index])[0]
#     else:
#         scattered = scatter(data, [-1])[0]

#     # Robust unpacking
#     if isinstance(scattered, list):
#         final_data = scattered[0]
#     else:
#         final_data = scattered

#     # 3. Robust Metadata Injection
#     # Different modalities require different 3D box types in img_metas
#     if 'img_metas' in final_data:
#         metas = final_data['img_metas']
#         # Unwrap DataContainer
#         if hasattr(metas, 'data'):
#             metas = metas.data
            
#         if isinstance(metas, list) and len(metas) > 0:
#             target = metas[0][0] if isinstance(metas[0], list) else metas[0]
            
#             if isinstance(target, dict) and 'box_type_3d' not in target:
#                 if is_lidar:
#                     target['box_type_3d'] = LiDARInstance3DBoxes
#                 else:
#                     target['box_type_3d'] = CameraInstance3DBoxes

#     # 4. Inference
#     with torch.no_grad():
#         # # --- SANITY CHECK START ---
#         # if 'points' in final_data:
#         #     pts = final_data['points']
#         #     # Recursively unwrap lists/tuples until we find the Tensor
#         #     while isinstance(pts, (list, tuple)):
#         #         pts = pts[0] if len(pts) > 0 else torch.tensor(0)
            
#         #     # Now safely print the sum
#         #     if hasattr(pts, 'sum'):
#         #         print(f"DEBUG: Input Sum for {os.path.basename(pcd_path)}: {pts.sum().item():.4f}")
#         # # --- SANITY CHECK END ---
#         model(return_loss=False, rescale=True, **final_data)
    
#     # 5. Collect Results from hooks
#     return {name: h.get_output() for name, h in hooks.items() if h.get_output() is not None}


def get_bn_module_refs(
    model: nn.Module
) -> Dict[str, _BatchNorm]:
    """
    Get reference to all BatchNorm modules in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dict mapping module names to _BatchNorm modules
    """
    bn_modules = {}
    
    for name, module in model.named_modules():
        if isinstance(module, _BatchNorm):
            bn_modules[name] = module
    
    return bn_modules


def default_hook_filter(name: str, module: nn.Module) -> bool:
    """
    Default filter for selecting layers to hook.
    Includes convolutions, linear layers, and normalization layers.
    
    Args:
        name: Layer name
        module: Module instance
        
    Returns:
        bool: Whether to hook this layer
    """
    cls = module.__class__.__name__.lower()
    
    # Skip activations
    skip_types = (
        nn.Dropout, nn.Dropout2d, nn.Dropout3d,
        nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU, nn.ELU,
        nn.Sigmoid, nn.Softmax
    )
    
    if isinstance(module, skip_types):
        return False
    
    # Include main layer types
    include_types = (
        nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.Linear,
        _BatchNorm, nn.SyncBatchNorm,
        nn.GroupNorm, nn.LayerNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d
    )
    
    if isinstance(module, include_types):
        return True
    
    # Check by name for norm-like layers
    is_normish = any(norm_str in cls for norm_str in 
                     ["batchnorm", "syncbn", "norm", "layernorm", "groupnorm", "instancenorm"])
    
    return is_normish


import re
from typing import List, Tuple

def sort_layers_by_network_order(layers: List[str]) -> List[str]:
    """
    Sort layers by typical network order for 3D Detection (FCOS3D/PointPillars).
    Ensures numerical layers (10 vs 2) and architectural stages follow data flow.
    """
    # 1. Strict chronological prefix order
    prefix_order = [
        "pts_voxel_layer", 
        "pts_voxel_encoder", 
        "pts_middle_encoder",
        "pts_backbone", 
        "backbone", 
        "pts_neck", 
        "neck", 
        "pts_bbox_head", 
        "bbox_head", 
        "head"
    ]
    
    # 2. Sub-layer priority to mimic data flow within the Neck
    # Lateral connections happen before FPN/Upsample/Extra convs
    sub_priority = {
        "lateral_convs": 0,
        "fpn_convs": 1,
        "upsample_layers": 2,
        "extra_convs": 3
    }

    def natural_keys(text: str):
        """Helper to sort 'layer10' after 'layer2'."""
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

    def sort_key(layer_name: str) -> Tuple:
        # Determine Main Stage
        prefix_idx = 999
        for i, prefix in enumerate(prefix_order):
            if layer_name.startswith(prefix):
                prefix_idx = i
                break
        
        # Determine Sub-Stage priority (mostly for the Neck)
        internal_priority = 99
        for sub_prefix, priority in sub_priority.items():
            if sub_prefix in layer_name:
                internal_priority = priority
                break
                
        # Return hierarchy: (Stage, Sub-Stage, Natural Alphanumeric Order)
        return (prefix_idx, internal_priority, natural_keys(layer_name))

    return sorted(layers, key=sort_key)
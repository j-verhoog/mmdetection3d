"""
Common utilities for model comparison:
- Hooks for extracting layer outputs
- Data loading and inference
- Visualization helpers
"""

import os
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
    
    # Disable BatchNorm running stats updates - ensure model is never modified
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            module.track_running_stats = False
    
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


def process_single_sample(
    model: nn.Module,
    pipeline: Compose,
    pcd_path: str,
    hooks: Dict[str, ForwardHook]
) -> Dict[str, torch.Tensor]:
    """
    Run inference on a single sample and collect hooked layer outputs.
    
    Args:
        model: Model to run inference
        pipeline: Data pipeline
        pcd_path: Path to point cloud file
        hooks: Dict of hooks to collect outputs
        
    Returns:
        Dict mapping layer names to output tensors (only non-None outputs)
    """
    # 1. Prepare Data
    data = dict(
        pts_filename=pcd_path,
        timestamp=0.0,
        sweeps=[],
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[]
    )
    data = pipeline(data)
    data = collate([data], samples_per_gpu=1)
    scattered = scatter(data, [0])[0]

    # Robust unpacking
    if isinstance(scattered, list):
        final_data = scattered[0]
    else:
        final_data = scattered

    # Robust Metadata Injection
    if 'img_metas' in final_data:
        metas = final_data['img_metas']
        if isinstance(metas, list) and len(metas) > 0:
            target = metas[0][0] if isinstance(metas[0], list) else metas[0]
            if isinstance(target, dict) and 'box_type_3d' not in target:
                target['box_type_3d'] = LiDARInstance3DBoxes

    # 2. Inference
    with torch.no_grad():
        model(return_loss=False, rescale=True, **final_data)
    
    # 3. Collect Results from hooks
    return {name: h.get_output() for name, h in hooks.items() if h.get_output() is not None}


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


def sort_layers_by_network_order(layers: List[str]) -> List[str]:
    """
    Sort layers by typical network order: backbone -> neck -> head.
    
    Args:
        layers: List of layer names
        
    Returns:
        Sorted list maintaining architectural order
    """
    prefix_order = [
        "pts_voxel_layer", "pts_voxel_encoder", "pts_middle_encoder",
        "pts_backbone", "pts_neck", "pts_bbox_head", "backbone", "neck", "head"
    ]
    
    def sort_key(layer_name: str) -> Tuple[int, str]:
        # Find matching prefix index
        prefix_idx = next(
            (i for i, prefix in enumerate(prefix_order) if layer_name.startswith(prefix)),
            999
        )
        return (prefix_idx, layer_name)
    
    return sorted(layers, key=sort_key)
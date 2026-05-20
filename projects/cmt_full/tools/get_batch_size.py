import os
import argparse
import importlib
import torch
from mmcv import Config, DictAction
from mmdet3d.models import build_model 

# does not work at all... 
# full cmt: python projects/cmt_full/tools/get_batch_size.py     projects/cmt/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs.py     --batch-size 2
# current config: python projects/cmt_full/tools/get_batch_size.py     projects/cmt/fed/all_domains/improved_lightweight_cmt_iterated.py     --batch-size 16

def parse_args():
    parser = argparse.ArgumentParser(description='Estimate VRAM for mmdet3d model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override config options')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size to estimate for')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    print(f"Loading config {args.config}...")
    
    # ==========================================
    # --- START OF EXACT USER LOGIC ---
    # ==========================================
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    import importlib
    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
                
    plg_lib_base = importlib.import_module('mmdetection3d.mmdet3d')
    # ==========================================
    # --- END OF EXACT USER LOGIC ---
    # ==========================================

    print("Building model on CPU to estimate parameters...")
    # Build the model on CPU. This uses 0 GB of GPU VRAM.
    model = build_model(cfg.model)
    
    # 1. Calculate Static Memory (Weights)
    # Assuming Mixed Precision/FP16 (2 bytes per param)
    param_count = sum(p.numel() for p in model.parameters())
    param_memory_gb = (param_count * 2) / 1024**3
    
    # 2. Calculate Optimizer States (Adam/AdamW)
    # Adam stores master weights (FP32) + momentum + variance = ~8 bytes per param
    optimizer_memory_gb = (param_count * 8) / 1024**3
    
    print(f"\n--- Analysis for Batch Size: {args.batch_size} ---")
    print(f"Model Parameters: {param_count / 1e6:.2f} Million")
    print(f"Weights Memory (FP16):   {param_memory_gb:.2f} GB")
    print(f"Optimizer Memory (Adam): {optimizer_memory_gb:.2f} GB")
    
    # 3. Estimate Activations
    # For 3D voxel models (like CMT) with high res (1600x640), 
    # activations take roughly 3x to 4x the weight memory per batch item.
    est_activations = param_memory_gb * 3 * args.batch_size 
    
    # 4. Total estimation + 1GB CUDA context overhead
    total_est = param_memory_gb + optimizer_memory_gb + est_activations + 1.0 
    
    print(f"Estimated Activations:   {est_activations:.2f} GB")
    print(f"CUDA Overhead:           1.00 GB")
    print("-" * 35)
    print(f"Estimated Total VRAM:    {total_est:.2f} GB")

if __name__ == "__main__":
    main()
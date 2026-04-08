import argparse
import torch
import os
import string


def parse_args():
    parser = argparse.ArgumentParser(description='Merge N models via FedAvg/FedBN and preserve state')
    
    parser.add_argument('--inputs', nargs='+', required=True, help='Paths to input model checkpoints')
    parser.add_argument('--outputs', nargs='+', required=True, help='Paths to save the distinct merged models')
    
    # Optional method flag. Defaults to fedavg to preserve original behavior.
    parser.add_argument('--method', type=str, default='fedavg', 
                    choices=['fedavg', 'fedmc'],
                    help='Aggregation method. Default is fedavg.')
    parser.add_argument('--config', type=str, default=None, 
                        help='Path to the MMDet3D config file (e.g., improved_lightweight_cmt_iterated.py). Required for FedBN.')
    parser.add_argument(
        '--select-ratio',
        type=float,
        default=0.05,
        help='fraction of total params selected per round (default: 0.05).')
    parser.add_argument(
        '--max-sparsity',
        type=float,
        default=0.4,
        help='max personalized parameter fraction (default: 0.4).')

    parser.add_argument('--fisher_paths', nargs='+', type=str, default=None, 
                    help='Paths to the saved Fisher Information tensors (Required ONLY for FedMC)')
    
    for i, char in enumerate(string.ascii_lowercase):
        help_text = f'Weight for model {char.upper()}' if i < 5 else argparse.SUPPRESS
        parser.add_argument(f'--weight-{char}', type=float, default=None, help=help_text)
        
    args = parser.parse_args()
    return args

def fedavg(models, output_paths, norm_weights):
    """
    Original FedAvg implementation. Averages all valid floating-point keys.
    Maintains exact original functionality.
    """
    print("Running standard FedAvg...")
    
    # 1. Setup accumulators for the N-way average using the first model's structure
    temp_ckpt = torch.load(models[0], map_location='cpu')

    # 2. Setup accumulators for the N-way average
    running_sum = {}
    presence_weights = {}
    
    for k, v in temp_ckpt['state_dict'].items():
        if not v.is_floating_point() or 'num_batches_tracked' in k:
             continue
             
        running_sum[k] = torch.zeros_like(v)
        presence_weights[k] = 0.0
        
    del temp_ckpt
    
    # 3. Iterate sequentially through remaining models
    print("Averaging weights...")
    for i in range(len(models)):
        m_path = models[i]
        w_i = norm_weights[i]
        
        ckpt_i = torch.load(m_path, map_location='cpu')
        state_dict_i = ckpt_i['state_dict']
        
        for k in running_sum.keys():
            if k in state_dict_i:
                running_sum[k] += state_dict_i[k] * w_i
                presence_weights[k] += w_i
                
        # Free memory after processing each model
        del ckpt_i 
        
    # Calculate final averaged weights
    averaged_weights = {k: running_sum[k] / presence_weights[k] for k in running_sum.keys()}

    # 4. Inject Averaged Weights into EACH Model and Save
    for in_path, out_path in zip(models, output_paths):
        print(f"Updating and saving individual state for: {out_path}")
        ckpt = torch.load(in_path, map_location='cpu')
        
        for k in averaged_weights.keys():
            ckpt['state_dict'][k] = averaged_weights[k]
            
        if 'optimizer' in ckpt and 'state' in ckpt['optimizer']:
            for param_id in ckpt['optimizer']['state']:
                for key in ckpt['optimizer']['state'][param_id]:
                    if torch.is_tensor(ckpt['optimizer']['state'][param_id][key]):
                        ckpt['optimizer']['state'][param_id][key].zero_()
                        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(ckpt, out_path)
        print(f"Saved merged model with optimizer state to {out_path}")

def fedbn(models, output_paths, norm_weights, model_instance):
    """
    FedBN implementation. Uses the provided PyTorch model instance to map out
    BatchNorm layers and excludes their weights, biases, and running stats from averaging.
    """
    print("Running FedBN. Extracting BatchNorm topology from model class...")
    
    # Identify all base names of BatchNorm layers using the actual PyTorch classes
    bn_prefixes = set()
    total_layers = 0
    for name, module in model_instance.named_modules():
        total_layers += 1
        # This catches nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, and SyncBatchNorm
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            clean_name = name.replace('module.', '')
            bn_prefixes.add(clean_name)
            
    print(f"Total layers in model: {total_layers}")
    print(f"Identified {len(bn_prefixes)} BatchNorm layers to keep local.")

    # 1. Setup accumulators using the first model's structure
    temp_ckpt = torch.load(models[0], map_location='cpu')

    running_sum = {}
    presence_weights = {}
    
    for k, v in temp_ckpt['state_dict'].items():
        if not v.is_floating_point() or 'num_batches_tracked' in k:
             continue
        
        clean_k = k.replace('module.', '')
             
        # Skip this key if it belongs to any identified BatchNorm layer
        if any(clean_k.startswith(prefix + '.') for prefix in bn_prefixes):
            continue
             
        running_sum[k] = torch.zeros_like(v)
        presence_weights[k] = 0.0
        
    del temp_ckpt
    
    # 2. Iterate sequentially through models
    print("Averaging non-BN weights...")
    for i in range(len(models)):
        m_path = models[i]
        w_i = norm_weights[i]
        
        ckpt_i = torch.load(m_path, map_location='cpu')
        state_dict_i = ckpt_i['state_dict']
        
        for k in running_sum.keys():
            if k in state_dict_i:
                running_sum[k] += state_dict_i[k] * w_i
                presence_weights[k] += w_i
                
        del ckpt_i 
        
    averaged_weights = {k: running_sum[k] / presence_weights[k] for k in running_sum.keys()}

    # 3. Inject Averaged Weights into EACH Model and Save
    for in_path, out_path in zip(models, output_paths):
        print(f"Updating and saving individual state for: {out_path}")
        ckpt = torch.load(in_path, map_location='cpu')
        
        # BN keys aren't in averaged_weights, so their local state remains untouched
        for k in averaged_weights.keys():
            ckpt['state_dict'][k] = averaged_weights[k]
            
        if 'optimizer' in ckpt and 'state' in ckpt['optimizer']:
            for param_id in ckpt['optimizer']['state']:
                for key in ckpt['optimizer']['state'][param_id]:
                    if torch.is_tensor(ckpt['optimizer']['state'][param_id][key]):
                        ckpt['optimizer']['state'][param_id][key].zero_()
                        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(ckpt, out_path)
        print(f"Saved merged model with optimizer state to {out_path}")

def fedmc(models, fisher_paths, output_paths, norm_weights, client_ids, 
          prev_global_path="/workspace/work_dirs/fedmc_states/global_model.pth", 
          mask_dir="/workspace/work_dirs/fedmc_masks", 
          select_ratio=0.05, max_sparsity=0.5):
    """
    FedMC implementation (Information Content Model Customization). 
    Automatically discovers and freezes personalized subnetworks for each client 
    based on Fisher Information (parameter importance), then aggregates the shared parameters.
    """
    print(f"Running FedMC Aggregation (select_ratio={select_ratio}, max_sparsity={max_sparsity})...")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(os.path.dirname(prev_global_path), exist_ok=True)

    if not fisher_paths:
        raise ValueError("Error: 'fisher_paths' must be provided when running FedMC. Check your argparse inputs.")

    # 1. Load Pre-Training Global Weights
    if not os.path.exists(prev_global_path):
        print(f"No previous global model found at {prev_global_path}. Exiting FedMC since we need a baseline for aggregation. Please run one round of standard FedAvg first.")
        exit(1)
        
    prev_global_ckpt = torch.load(prev_global_path, map_location='cpu')
    prev_state = prev_global_ckpt['state_dict']
    
    # Identify valid floating-point keys
    valid_keys = [k for k, v in prev_state.items() if v.is_floating_point() and 'num_batches_tracked' not in k]
    total_params = sum(prev_state[k].numel() for k in valid_keys)
    print(f"Total valid parameters for FedMC: {total_params:,}")

    # ---------------------------------------------------------
    # Layer Mapping for Visualization (Created Once)
    # ---------------------------------------------------------
    mapping_file = os.path.join(mask_dir, "layer_mapping.pth")
    if not os.path.exists(mapping_file):
        layer_info = {}
        current_idx = 0
        for k in valid_keys:
            numel = prev_state[k].numel()
            layer_info[k] = {
                "shape": list(prev_state[k].shape),
                "numel": numel,
                "start_idx": current_idx,
                "end_idx": current_idx + numel
            }
            current_idx += numel
        torch.save(layer_info, mapping_file)
        print(f"Created layer mapping file at {mapping_file}")

    # ---------------------------------------------------------
    # Auto-Detect Current Round Directory
    # ---------------------------------------------------------
    existing_rounds = []
    for d in os.listdir(mask_dir):
        if d.startswith("round_") and os.path.isdir(os.path.join(mask_dir, d)):
            try:
                existing_rounds.append(int(d.split("_")[1]))
            except ValueError:
                pass
    current_round = max(existing_rounds) + 1 if existing_rounds else 0
    round_mask_dir = os.path.join(mask_dir, f"round_{current_round}")
    os.makedirs(round_mask_dir, exist_ok=True)
    print(f"Saving historical Fisher masks for round {current_round} to {round_mask_dir}")

    # ---------------------------------------------------------
    # Phase 1: Client Subnetwork Discovery using Fisher Information
    # ---------------------------------------------------------
    print("\nPhase 1: Discovering informative client subnetworks via Fisher Information...")
    for m_path, f_path, cid in zip(models, fisher_paths, client_ids):
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        
        # Load or initialize client mask (0 = Global, 1 = Personalized)
        if os.path.exists(mask_path):
            client_mask = torch.load(mask_path)
        else:
            client_mask = {k: torch.zeros_like(prev_state[k], dtype=torch.bool) for k in valid_keys}

        ckpt_i = torch.load(m_path, map_location='cpu')
        state_i = ckpt_i['state_dict']
        
        # Load Fisher Information diagonal
        fisher_i = torch.load(f_path, map_location='cpu')
        if 'state_dict' in fisher_i: # Handle case if saved as a dict like checkpoints
            fisher_i = fisher_i['state_dict']
        
        all_importance = []
        current_personalized = 0
        
        for k in valid_keys:
            if k in state_i and k in fisher_i:
                # Use Fisher Information as the importance metric (using abs as a safety net)
                importance = torch.abs(fisher_i[k])
                
                # Only evaluate parameters that are currently shared (mask == 0)
                global_mask = ~client_mask[k]
                all_importance.append(importance[global_mask].flatten())
                current_personalized += client_mask[k].sum().item()
                
        # Determine how many new parameters to select this round
        cat_importance = torch.cat(all_importance)
        k_to_select = int(total_params * select_ratio)
        max_allowed = int(total_params * max_sparsity)
        k_to_select = min(k_to_select, max_allowed - current_personalized)
        
        if k_to_select > 0 and len(cat_importance) > 0:
            k_to_select = min(k_to_select, len(cat_importance))
            
            # --- FISHER LOGGING ---
            avg_overall_fisher = cat_importance.mean().item()
            top_values = torch.topk(cat_importance, k_to_select).values
            threshold = top_values[-1].item()
            avg_selected_fisher = top_values.mean().item()
            
            print(f"\n[{cid}] FISHER INFORMATIVENESS:")
            print(f"  -> Avg Fisher (all shared params): {avg_overall_fisher:.6e}")
            print(f"  -> Avg Fisher (selected top-K):    {avg_selected_fisher:.6e}")
            print(f"  -> Fisher Threshold Cutoff:        {threshold:.6e}")
            
            # Update the mask permanently
            new_personalized = 0
            layer_counts = {}
            
            for k in valid_keys:
                if k in state_i and k in fisher_i:
                    importance = torch.abs(fisher_i[k])
                    
                    # Flip 0 to 1 if it exceeds the Fisher threshold and is currently 0
                    new_ones = (~client_mask[k]) & (importance >= threshold)
                    client_mask[k][new_ones] = True
                    
                    selected_in_layer = new_ones.sum().item()
                    new_personalized += selected_in_layer
                    
                    if selected_in_layer > 0:
                        layer_counts[k] = selected_in_layer
            
            # Print top 3 layers with the most highly informative parameters
            top_layers = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  -> Most informative layers modified: {', '.join([f'{k} (+{v} params)' for k, v in top_layers])}")
            print(f"  -> Total Sparsity: {(current_personalized + new_personalized) / total_params * 100:.2f}% (+{new_personalized:,} new)")
        else:
            print(f"\n[{cid}] Reached max sparsity or no params to select. Total sparsity: {current_personalized / total_params * 100:.2f}%")
            
        torch.save(client_mask, mask_path)

        # Save historical visualization mask (0=Global, 1=Personal)
        vis_mask = {k: v.to(torch.int8) for k, v in client_mask.items()}
        hist_mask_path = os.path.join(round_mask_dir, f"{cid}_mask.pt")
        torch.save(vis_mask, hist_mask_path)

        del ckpt_i, fisher_i
        
    # ---------------------------------------------------------
    # Phase 2: Masked Server Aggregation
    # ---------------------------------------------------------
    print("\nPhase 2: Aggregating non-informative (shared) parameters on server...")
    running_sum = {k: torch.zeros_like(v) for k, v in prev_state.items() if k in valid_keys}
    presence_weights = {k: torch.zeros_like(v) for k, v in prev_state.items() if k in valid_keys}
    
    for m_path, cid, w_i in zip(models, client_ids, norm_weights):
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        client_mask = torch.load(mask_path)
        
        ckpt_i = torch.load(m_path, map_location='cpu')
        state_i = ckpt_i['state_dict']
        
        for k in valid_keys:
            if k in state_i:
                # Active mask = 1 where parameter is shared (mask == 0)
                active_mask = (~client_mask[k]).float()
                running_sum[k] += state_i[k] * active_mask * w_i
                presence_weights[k] += active_mask * w_i
                
        del ckpt_i
        
    # Finalize averaged weights
    averaged_weights = {}
    for k in valid_keys:
        valid_mask = presence_weights[k] > 0
        averaged_weights[k] = torch.where(
            valid_mask,
            running_sum[k] / presence_weights[k].clamp(min=1e-9),
            prev_state[k] # Fallback if all clients personalized it
        )
        
    # Save the new global model
    for k in valid_keys:
        prev_global_ckpt['state_dict'][k] = averaged_weights[k]
    torch.save(prev_global_ckpt, prev_global_path)
    del prev_global_ckpt

    # ---------------------------------------------------------
    # Phase 3: Client Subnetwork Injection & Saving
    # ---------------------------------------------------------
    print("Phase 3: Injecting shared weights back into client models...")
    for in_path, out_path, cid in zip(models, output_paths, client_ids):
        ckpt = torch.load(in_path, map_location='cpu')
        state = ckpt['state_dict']
        
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        client_mask = torch.load(mask_path)
        
        for k in valid_keys:
            if k in state:
                # Final weight = Mask * Local + (1 - Mask) * Global
                m = client_mask[k].float()
                state[k] = (m * state[k]) + ((1.0 - m) * averaged_weights[k])
                        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(ckpt, out_path)
        print(f"Saved personalized FedMC model for {cid} to {out_path}")

def main():
    args = parse_args()
    
    if len(args.inputs) != len(args.outputs):
        raise ValueError("Number of input paths must match number of output paths.")
        
    model_paths = args.inputs
    output_paths = args.outputs
    
    # Extract and normalize weights
    raw_weights = []
    for i in range(len(model_paths)):
        char = string.ascii_lowercase[i]
        w = getattr(args, f'weight_{char}', None)
        raw_weights.append(w if w is not None else 1.0)
        
    total_weight = sum(raw_weights)
    norm_weights = [w / total_weight for w in raw_weights]

    print(f"Loading {len(model_paths)} models for merge via {args.method.upper()}...")
    for i, (m_path, w) in enumerate(zip(model_paths, norm_weights)):
        print(f" Model {string.ascii_uppercase[i]}: {m_path} (normalized w={w:.4f})")
        
    # Route to the appropriate modular function
    if args.method == 'fedavg':
        fedavg(model_paths, output_paths, norm_weights)

    elif args.method == 'fedbn':
        if args.config is None:
            raise ValueError("You must provide a --config file to use FedBN so the model architecture can be built.")
            
        print(f"Building model from config: {args.config}")
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(['projects.mmdet3d_plugin.models.detectors.cmt'])
        
        from mmcv import Config
        cfg = Config.fromfile(args.config)
        if cfg.get('custom_imports', None):
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

        from mmdet3d.models import build_model
        model_instance = build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))

        fedbn(model_paths, output_paths, norm_weights, model_instance)

    elif args.method == 'fedmc':
        client_ids = [f"Model{string.ascii_uppercase[i]}" for i in range(len(model_paths))]
        fedmc(
            models=model_paths, 
            output_paths=output_paths, 
            norm_weights=norm_weights, 
            client_ids=client_ids,
            fisher_paths=args.fisher_paths,
            prev_global_path="/workspace/work_dirs/fedmc_states/global_model.pth", 
            mask_dir="/workspace/work_dirs/fedmc_masks",
            select_ratio=args.select_ratio,
            max_sparsity=args.max_sparsity
        )

if __name__ == '__main__':
    main()
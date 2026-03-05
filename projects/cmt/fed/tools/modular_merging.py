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
                    choices=['fedavg', 'fedbn', 'fednorm', 'fedper', 'fed_bn_and_per', 'fedmedian', 'feddyn', 'fed_dyn_bn_and_per'],
                    help='Aggregation method. Default is fedavg.')
    parser.add_argument('--config', type=str, default=None, 
                        help='Path to the MMDet3D config file (e.g., improved_lightweight_cmt_iterated.py). Required for FedBN.')

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

def fedper(models, output_paths, norm_weights):
    """
    FedPer implementation. Averages the backbone and neck, but keeps the 
    task head subsets completely local and personalized.
    """
    print("Running FedPer. Keeping task head local...")
    
    # 1. Setup accumulators using the first model's structure
    temp_ckpt = torch.load(models[0], map_location='cpu')

    running_sum = {}
    presence_weights = {}
    
    personalized_layers = 0
    global_layers = 0
    # Define specifically which sub-components of the head to keep local
    local_prefixes = (
        'pts_bbox_head.common_heads',
        'pts_bbox_head.separate_head',
        'pts_bbox_head.tasks',
        'pts_bbox_head.task_heads'       # <--- FIXED PREFIX
    )
    
    for k, v in temp_ckpt['state_dict'].items():
        if not v.is_floating_point() or 'num_batches_tracked' in k:
             continue
             
        # Only skip the specific task-prediction heads, not the whole transformer
        if any(k.startswith(prefix) for prefix in local_prefixes):
            personalized_layers += 1
            continue
             
        running_sum[k] = torch.zeros_like(v)
        presence_weights[k] = 0.0
        global_layers += 1
        
    del temp_ckpt
    
    print(f"Identified {personalized_layers} personalized head layers to keep local.")
    print(f"Identified {global_layers} global layers to average across models.")
    
    # 2. Iterate sequentially through models
    print("Averaging backbone and neck weights...")
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

    # 3. Inject Averaged Weights and Save
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
        print(f"Saved merged model to {out_path}")

def fed_bn_and_per(models, output_paths, norm_weights, model_instance):
    """
    Combined FedBN + FedPer. Excludes both BatchNorm layers and task head
    subsets from averaging, keeping them local and personalized.
    """
    print("Running FedBN+FedPer. Keeping BatchNorm layers and task heads local...")
    
    # --- FedBN: Identify BatchNorm layer prefixes ---
    bn_prefixes = set()
    total_layers = 0
    for name, module in model_instance.named_modules():
        total_layers += 1
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            clean_name = name.replace('module.', '')
            bn_prefixes.add(clean_name)
            
    print(f"Total layers in model: {total_layers}")
    print(f"Identified {len(bn_prefixes)} BatchNorm layers to keep local.")
    
    # --- FedPer: Define task head prefixes to keep local ---
    local_prefixes = (
        'pts_bbox_head.common_heads',
        'pts_bbox_head.separate_head',
        'pts_bbox_head.tasks',
        'pts_bbox_head.task_heads'       # <--- FIXED PREFIX
    )
    
    # 1. Setup accumulators, excluding both BN and task head keys
    temp_ckpt = torch.load(models[0], map_location='cpu')

    running_sum = {}
    presence_weights = {}
    shared_layers = 0
    personal_layers =0
    private_bn_layers = 0
    for k, v in temp_ckpt['state_dict'].items():
        if not v.is_floating_point() or 'num_batches_tracked' in k:
             continue
        
        clean_k = k.replace('module.', '')
             
        # Skip BatchNorm keys
        if any(clean_k.startswith(prefix + '.') for prefix in bn_prefixes):
            private_bn_layers += 1
            continue
            
        # Skip task head keys
        if any(k.startswith(prefix) for prefix in local_prefixes):
            personal_layers += 1
            continue
             
        running_sum[k] = torch.zeros_like(v)
        presence_weights[k] = 0.0
        shared_layers += 1
        
    del temp_ckpt
    
    print(f"Identified {private_bn_layers} private BatchNorm layers to keep local.")
    print(f"Identified {personal_layers} personalized head layers to keep local.")
    print(f"Identified {shared_layers} shared layers to average across models.")
    
    # 2. Iterate sequentially through models
    print("Averaging non-BN, non-head weights...")
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
        
        # BN and head keys aren't in averaged_weights, so their local state remains untouched
        for k in averaged_weights.keys():
            ckpt['state_dict'][k] = averaged_weights[k]
            
        if 'optimizer' in ckpt and 'state' in ckpt['optimizer']:
            for param_id in ckpt['optimizer']['state']:
                for key in ckpt['optimizer']['state'][param_id]:
                    if torch.is_tensor(ckpt['optimizer']['state'][param_id][key]):
                        ckpt['optimizer']['state'][param_id][key].zero_()
                        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(ckpt, out_path)
        print(f"Saved merged model to {out_path}")

def fedmedian(models, output_paths):
    """
    FedMedian implementation. Calculates the coordinate-wise median across all models.
    Provides robust aggregation against severe domain shifts and outliers.
    """
    print("Running FedMedian. Calculating coordinate-wise median (ignoring scalar weights)...")
    
    temp_ckpt = torch.load(models[0], map_location='cpu')

    # We need to collect tensors from all models in a list before taking the median
    stacked_weights = {}
    for k, v in temp_ckpt['state_dict'].items():
        if not v.is_floating_point() or 'num_batches_tracked' in k:
             continue
        stacked_weights[k] = []
        
    del temp_ckpt
    
    # 1. Load weights from all models into lists
    print("Loading weights from all models into memory...")
    for i in range(len(models)):
        m_path = models[i]
        ckpt_i = torch.load(m_path, map_location='cpu')
        state_dict_i = ckpt_i['state_dict']
        
        for k in stacked_weights.keys():
            if k in state_dict_i:
                # Append a clone of the tensor to our list
                stacked_weights[k].append(state_dict_i[k].clone())
        del ckpt_i 
        
    # 2. Calculate medians
    print("Stacking tensors and calculating median...")
    averaged_weights = {}
    for k, tensor_list in stacked_weights.items():
        if len(tensor_list) > 0:
            # Stack tensors along dim=0 and compute median
            # torch.median returns a namedtuple (values, indices); we just want values
            stacked_tensor = torch.stack(tensor_list, dim=0)
            averaged_weights[k] = torch.median(stacked_tensor, dim=0).values
            
    # Free up memory immediately
    del stacked_weights

    # 3. Inject and Save
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
        print(f"Saved merged model to {out_path}")

def feddyn(models, output_paths, norm_weights, alpha=0.01, work_dir="work_dirs/feddyn_states"):
    print("Running FedDyn Aggregation...")
    
    # 1. Standard FedAvg of the incoming client weights
    temp_ckpt = torch.load(models[0], map_location='cpu')
    running_sum = {k: torch.zeros_like(v) for k, v in temp_ckpt['state_dict'].items() if v.is_floating_point() and 'num_batches_tracked' not in k}
    del temp_ckpt
    
    for i, m_path in enumerate(models):
        ckpt_i = torch.load(m_path, map_location='cpu')
        for k in running_sum.keys():
            if k in ckpt_i['state_dict']:
                running_sum[k] += ckpt_i['state_dict'][k] * norm_weights[i]
        del ckpt_i
        
    averaged_weights = {k: running_sum[k] for k in running_sum.keys()}

    # 2. Update Global Server State (h_global)
    h_global_path = os.path.join(work_dir, "server_h_state.pth")
    if os.path.exists(h_global_path):
        h_global = torch.load(h_global_path)
    else:
        h_global = {k: torch.zeros_like(v) for k, v in averaged_weights.items()}

    # Sum up all client h_states
    client_ids = ["ModelA", "ModelB", "ModelC", "ModelD", "ModelE"]
    skipped = 0
    updated = 0
    for i, cid in enumerate(client_ids):
        client_h_path = os.path.join(work_dir, f"{cid}_h_state.pth")
        if os.path.exists(client_h_path):
            client_h = torch.load(client_h_path)
            for k in h_global.keys():
                # Safety check: Only update if the client actually tracked this parameter's state
                if k in client_h:
                    h_global[k] -= (alpha * norm_weights[i]) * (averaged_weights[k] - client_h[k])
                    updated += 1
                else:
                    #print(f"Warning: {client_h_path} does not contain state for {k}. Skipping update for this key.")
                    skipped += 1

    print(f"FedDyn: Updated global state with client contributions. Skipped {skipped} keys due to missing client states.")
    print(f"FedDyn: Successfully updated {updated} keys in global state.")
    torch.save(h_global, h_global_path)

    # 3. Apply Global State to Averaged Weights
    for k in averaged_weights.keys():
        averaged_weights[k] += (1.0 / alpha) * h_global[k]

    # 4. Inject and Save
    for in_path, out_path in zip(models, output_paths):
        ckpt = torch.load(in_path, map_location='cpu')
        for k in averaged_weights.keys():
            ckpt['state_dict'][k] = averaged_weights[k]
        
        # Zero out optimizer momentum
        if 'optimizer' in ckpt and 'state' in ckpt['optimizer']:
            for param_id in ckpt['optimizer']['state']:
                for key in ckpt['optimizer']['state'][param_id]:
                    if torch.is_tensor(ckpt['optimizer']['state'][param_id][key]):
                        ckpt['optimizer']['state'][param_id][key].zero_()
                        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(ckpt, out_path)

def fed_dyn_bn_and_per(models, output_paths, norm_weights, model_instance, client_ids, alpha=0.01, work_dir="work_dirs/feddyn_states"):
    """
    Combined FedDyn + FedBN + FedPer. 
    Excludes both BatchNorm layers and task head subsets from averaging and FedDyn updates, 
    keeping them completely local. Applies layer-wise FedDyn math to the shared layers.
    """
    print("Running FedDyn + FedBN + FedPer Aggregation...")
    
    # --- FedBN: Identify BatchNorm layer prefixes ---
    bn_prefixes = set()
    for name, module in model_instance.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            clean_name = name.replace('module.', '')
            bn_prefixes.add(clean_name)
            
    # --- FedPer: Define task head prefixes to keep local ---
    local_prefixes = (
        'pts_bbox_head.common_heads',
        'pts_bbox_head.separate_head',
        'pts_bbox_head.tasks',
        'pts_bbox_head.task_heads'       # <--- FIXED PREFIX
    )
    
    global_layers = 0
    personalized_layers = 0
    bn_layers = 0
    # 1. Setup accumulators for FedDyn, excluding BN and local heads
    temp_ckpt = torch.load(models[0], map_location='cpu')
    running_sum = {}
    
    for k, v in temp_ckpt['state_dict'].items():
        if not v.is_floating_point() or 'num_batches_tracked' in k:
             continue
        
        clean_k = k.replace('module.', '')
        
        # Skip BatchNorm and Task Head keys
        if any(clean_k.startswith(prefix + '.') for prefix in bn_prefixes):
            bn_layers += 1
            continue
        if any(k.startswith(prefix) for prefix in local_prefixes):
            personalized_layers += 1
            continue
             
        running_sum[k] = torch.zeros_like(v)
        global_layers += 1
        
    del temp_ckpt
    
    print(f"Identified {bn_layers} BatchNorm layers to keep local.")
    print(f"Identified {personalized_layers} personalized head layers to keep local.")
    print(f"Identified {global_layers} global layers to average across models.")
    
    # 2. Standard FedAvg of the incoming client weights (only for shared keys)
    for i, m_path in enumerate(models):
        ckpt_i = torch.load(m_path, map_location='cpu')
        for k in running_sum.keys():
            if k in ckpt_i['state_dict']:
                running_sum[k] += ckpt_i['state_dict'][k] * norm_weights[i]
        del ckpt_i
        
    averaged_weights = {k: running_sum[k] for k in running_sum.keys()}

    # 3. Update Global Server State (h_global)
    h_global_path = os.path.join(work_dir, "server_h_state.pth")
    if os.path.exists(h_global_path):
        h_global = torch.load(h_global_path)
    else:
        h_global = {k: torch.zeros_like(v) for k, v in averaged_weights.items()}

    skipped = 0
    updated = 0
    for i, cid in enumerate(client_ids):
        client_h_path = os.path.join(work_dir, f"{cid}_h_state.pth")
        if os.path.exists(client_h_path):
            client_h = torch.load(client_h_path)
            for k in h_global.keys():
                if k in client_h:
                    # Match layer-wise alpha logic from your FedDyn hook
                    layer_alpha = alpha
                    if 'img_backbone' in k:
                        layer_alpha = alpha * 0.01
                    elif 'img_neck' in k:
                        layer_alpha = alpha * 0.1
                        
                    h_global[k] -= (layer_alpha * norm_weights[i]) * (averaged_weights[k] - client_h[k])
                    updated += 1
                else:
                    skipped += 1

    print(f"FedDyn+BN+Per: Updated {updated} shared keys globally. Skipped {skipped} keys.")
    torch.save(h_global, h_global_path)

    # 4. Apply Global State to Averaged Weights
    for k in averaged_weights.keys():
        layer_alpha = alpha
        if 'img_backbone' in k:
            layer_alpha = alpha * 0.01
        elif 'img_neck' in k:
            layer_alpha = alpha * 0.1
            
        averaged_weights[k] += (1.0 / layer_alpha) * h_global[k]

    # 5. Inject Averaged Weights into EACH Model and Save
    for in_path, out_path in zip(models, output_paths):
        ckpt = torch.load(in_path, map_location='cpu')
        
        # BN and head keys aren't in averaged_weights, so their local state remains untouched
        for k in averaged_weights.keys():
            ckpt['state_dict'][k] = averaged_weights[k]
            
        if 'optimizer' in ckpt and 'state' in ckpt['optimizer']:
            for param_id in ckpt['optimizer']['state']:
                for key in ckpt['optimizer']['state'][param_id]:
                    if torch.is_tensor(ckpt['optimizer']['state'][param_id][key]):
                        ckpt['optimizer']['state'][param_id][key].zero_()
                        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(ckpt, out_path)
        print(f"Saved merged model to {out_path}")

def fednorm(models, output_paths, norm_weights, model_instance):
    """
    FedNorm implementation. Uses the provided PyTorch model instance to map out
    all Normalization layers (BatchNorm, LayerNorm, InstanceNorm, GroupNorm) 
    and excludes their weights, biases, and running stats from averaging.
    """
    print("Running FedNorm. Extracting all Normalization topology from model class...")
    
    # Identify all base names of Normalization layers
    norm_prefixes = set()
    total_layers = 0
    
    # Define all standard normalization base classes in PyTorch
    norm_classes = (
        torch.nn.modules.batchnorm._BatchNorm,
        torch.nn.LayerNorm,
        torch.nn.modules.instancenorm._InstanceNorm,
        torch.nn.GroupNorm
    )
    
    for name, module in model_instance.named_modules():
        total_layers += 1
        # Check for any of the normalization classes
        if isinstance(module, norm_classes):
            clean_name = name.replace('module.', '')
            norm_prefixes.add(clean_name)
            
    print(f"Total layers in model: {total_layers}")
    print(f"Identified {len(norm_prefixes)} Normalization layers to keep local.")

    # 1. Setup accumulators using the first model's structure
    temp_ckpt = torch.load(models[0], map_location='cpu')

    running_sum = {}
    presence_weights = {}
    
    for k, v in temp_ckpt['state_dict'].items():
        if not v.is_floating_point() or 'num_batches_tracked' in k:
             continue
        
        clean_k = k.replace('module.', '')
             
        # Skip this key if it belongs to any identified Normalization layer
        if any(clean_k.startswith(prefix + '.') for prefix in norm_prefixes):
            continue
             
        running_sum[k] = torch.zeros_like(v)
        presence_weights[k] = 0.0
        
    del temp_ckpt
    
    # 2. Iterate sequentially through models
    print("Averaging non-Normalization weights...")
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
        
        # Norm keys aren't in averaged_weights, so their local state remains untouched
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
    elif args.method == 'fedper':
        fedper(model_paths, output_paths, norm_weights)
        
    elif args.method == 'fed_bn_and_per':
        if args.config is None:
            raise ValueError("You must provide a --config file to use FedBN+FedPer so the model architecture can be built.")
            
        print(f"Building model from config: {args.config}")
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(['projects.mmdet3d_plugin.models.detectors.cmt'])
        
        from mmcv import Config
        cfg = Config.fromfile(args.config)
        if cfg.get('custom_imports', None):
            import_modules_from_strings(**cfg['custom_imports'])

        import importlib
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

        fed_bn_and_per(model_paths, output_paths, norm_weights, model_instance)

    elif args.method == 'fedmedian':
        # FedMedian ignores scalar norm_weights
        fedmedian(model_paths, output_paths)
        
    elif args.method == 'feddyn':
        feddyn(model_paths, output_paths, norm_weights, alpha=0.01, work_dir="work_dirs/feddyn_states")

    elif args.method == 'fed_dyn_bn_and_per':
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

        # Define the exact client IDs that correspond to the models being merged
        # You may want to pass these as argparse arguments, but hardcoding for now based on your feddyn script
        client_ids = ["ModelA", "ModelB", "ModelC", "ModelD", "ModelE"][:len(model_paths)]
        
        fed_dyn_bn_and_per(
            models=model_paths, 
            output_paths=output_paths, 
            norm_weights=norm_weights, 
            model_instance=model_instance,
            client_ids=client_ids,
            alpha=0.01,
            work_dir="work_dirs/feddyn_states"
        )
    elif args.method == 'fednorm':
        if args.config is None:
            raise ValueError("You must provide a --config file to use fednorm so the model architecture can be built.")
            
        print(f"Building model from config: {args.config}")
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(['projects.mmdet3d_plugin.models.detectors.cmt'])
        
        from mmcv import Config
        cfg = Config.fromfile(args.config)
        if cfg.get('custom_imports', None):
            import_modules_from_strings(**cfg['custom_imports'])

        import importlib
        if hasattr(cfg, 'plugin'):
            if cfg.plugin:
                if hasattr(cfg, 'plugin_dir'):
                    plugin_dir = cfg.plugin_dir
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]

                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    plg_lib = importlib.import_module(_module_path)
                else:
                    _module_dir = os.path.dirname(args.config)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]
                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    plg_lib = importlib.import_module(_module_path)
                    
        plg_lib_base = importlib.import_module('mmdetection3d.mmdet3d')

        from mmdet3d.models import build_model
        model_instance = build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))

        fednorm(model_paths, output_paths, norm_weights, model_instance)

if __name__ == '__main__':
    main()
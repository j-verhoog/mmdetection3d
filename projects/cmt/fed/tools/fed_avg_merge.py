import argparse
import torch
import os
import copy
import string

def parse_args():
    parser = argparse.ArgumentParser(description='Merge N models via FedAvg and preserve state')
    
    # Using nargs='+' allows us to accept any number of positional arguments.
    # The last argument will be treated as the output path, the rest as models.
    parser.add_argument('paths', nargs='+', help='Paths to model checkpoints, with the LAST path being the output destination.')
    
    # Dynamically add weight arguments up to 'z' (26 models max). 
    # We show help for the first few to keep the menu clean, and suppress the rest.
    for i, char in enumerate(string.ascii_lowercase):
        help_text = f'Weight for model {char.upper()}' if i < 4 else argparse.SUPPRESS
        parser.add_argument(f'--weight-{char}', type=float, default=None, help=help_text)
        
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Ensure minimum required arguments are present (2 models + 1 output)
    if len(args.paths) < 3:
        raise ValueError("You must provide at least two model paths and one output path.")
        
    models = args.paths[:-1]
    output_path = args.paths[-1]
    
    # Extract weights dynamically based on how many models were provided
    raw_weights = []
    for i in range(len(models)):
        char = string.ascii_lowercase[i]
        w = getattr(args, f'weight_{char}')
        
        # Default to 1.0 (equal weighting) if weight flag is missing
        if w is None:
            w = 1.0
        raw_weights.append(w)
        
    total_weight = sum(raw_weights)
    norm_weights = [w / total_weight for w in raw_weights]

    print(f"Loading {len(models)} models for merge...")
    for i, (m_path, w) in enumerate(zip(models, norm_weights)):
        print(f" Model {string.ascii_uppercase[i]}: {m_path} (normalized w={w:.4f})")
        
    # 1. Load the first model (Model A) as the base and deepcopy to preserve state/meta
    print("Deepcopying Model A to preserve full state...")
    base_ckpt = torch.load(models[0], map_location='cpu')
    merged_ckpt = copy.deepcopy(base_ckpt)
    base_keys = list(merged_ckpt['state_dict'].keys())
    
    # 2. Setup accumulators for the N-way average
    running_sum = {}
    presence_weights = {}
    
    for k in base_keys:
        # NEW FIX: Skip integer tensors (like num_batches_tracked or step counts)
        # They will remain exactly as they were in Model A via the deepcopy.
        if not base_ckpt['state_dict'][k].is_floating_point() or 'num_batches_tracked' in k:
            continue
            
        running_sum[k] = base_ckpt['state_dict'][k] * norm_weights[0]
        presence_weights[k] = norm_weights[0]
    
    # Free base_ckpt to save RAM, we only need merged_ckpt and accumulators now
    del base_ckpt
    
    # 3. Iterate sequentially through remaining models
    print("Averaging weights...")
    for i in range(1, len(models)):
        m_path = models[i]
        w_i = norm_weights[i]
        
        ckpt_i = torch.load(m_path, map_location='cpu')
        state_dict_i = ckpt_i['state_dict']
        
        # Iterate over running_sum.keys() instead of base_keys to naturally skip integers
        for k in running_sum.keys():
            if k in state_dict_i:
                running_sum[k] += state_dict_i[k] * w_i
                presence_weights[k] += w_i
                
        # Free memory after processing each model to remain robust against large N
        del ckpt_i 
        
    # 4. Finalize the average and assign back to merged_ckpt
    # Again, only update the keys we actually averaged
    for k in running_sum.keys():
        merged_ckpt['state_dict'][k] = running_sum[k] / presence_weights[k]

    # ... [Optimizer zeroing and saving logic remains exactly the same] ...

    # NEW: Zero out optimizer momentum to prevent bleed-over across domains, 
    # while keeping the state structure intact so --resume-from doesn't crash.
    if 'optimizer' in merged_ckpt and 'state' in merged_ckpt['optimizer']:
        for param_id in merged_ckpt['optimizer']['state']:
            for key in merged_ckpt['optimizer']['state'][param_id]:
                # If the state holds a tensor (like momentum buffers), zero it out
                if torch.is_tensor(merged_ckpt['optimizer']['state'][param_id][key]):
                    merged_ckpt['optimizer']['state'][param_id][key].zero_()
                    

    # Optional: Log the epoch we are merging at
    epoch = merged_ckpt.get('meta', {}).get('epoch', 'Unknown')
    print(f"Merging at end of Epoch: {epoch}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(merged_ckpt, output_path)
    print(f"Saved merged model with optimizer state to {output_path}")

if __name__ == '__main__':
    main()
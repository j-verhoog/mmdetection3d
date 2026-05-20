import argparse
import torch
import os
import copy
import string

def parse_args():
    parser = argparse.ArgumentParser(description='Merge N models via FedAvg and preserve state')
    
    # Using nargs='+' allows us to accept any number of positional arguments.

    parser.add_argument('--inputs', nargs='+', required=True, help='Paths to input model checkpoints')
    parser.add_argument('--outputs', nargs='+', required=True, help='Paths to save the distinct merged models')

    # Dynamically add weight arguments up to 'z' (26 models max). 
    # We show help for the first few to keep the menu clean, and suppress the rest.
    for i, char in enumerate(string.ascii_lowercase):
        help_text = f'Weight for model {char.upper()}' if i < 5 else argparse.SUPPRESS
        parser.add_argument(f'--weight-{char}', type=float, default=None, help=help_text)
        
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    if len(args.inputs) != len(args.outputs):
        raise ValueError("Number of input paths must match number of output paths.")
        
    models = args.inputs
    output_paths = args.outputs
    
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
        
    # # 1. Load the first model (Model A) as the base and deepcopy to preserve state/meta
    # print("Deepcopying Model A to preserve full state...")
    # base_ckpt = torch.load(models[0], map_location='cpu')
    
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
        
        # Iterate over running_sum.keys() instead of base_keys to naturally skip integers
        for k in running_sum.keys():
            if k in state_dict_i:
                running_sum[k] += state_dict_i[k] * w_i
                presence_weights[k] += w_i
                
        # Free memory after processing each model to remain robust against large N
        del ckpt_i 
        
    # 3. Calculate final averaged weights
    averaged_weights = {k: running_sum[k] / presence_weights[k] for k in running_sum.keys()}

    # # ... [Optimizer zeroing and saving logic remains exactly the same] ...

    # # NEW: Zero out optimizer momentum to prevent bleed-over across domains, 
    # # while keeping the state structure intact so --resume-from doesn't crash.
    # if 'optimizer' in merged_ckpt and 'state' in merged_ckpt['optimizer']:
    #     for param_id in merged_ckpt['optimizer']['state']:
    #         for key in merged_ckpt['optimizer']['state'][param_id]:
    #             # If the state holds a tensor (like momentum buffers), zero it out
    #             if torch.is_tensor(merged_ckpt['optimizer']['state'][param_id][key]):
    #                 merged_ckpt['optimizer']['state'][param_id][key].zero_()
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

if __name__ == '__main__':
    main()
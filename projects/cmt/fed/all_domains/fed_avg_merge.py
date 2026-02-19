import argparse
import torch
import shutil
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
    # We start with Model A's values scaled by its normalized weight
    running_sum = {k: base_ckpt['state_dict'][k] * norm_weights[0] for k in base_keys}
    presence_weights = {k: norm_weights[0] for k in base_keys}
    
    # Free base_ckpt to save RAM, we only need merged_ckpt and accumulators now
    del base_ckpt
    
    # 3. Iterate sequentially through remaining models
    print("Averaging weights...")
    for i in range(1, len(models)):
        m_path = models[i]
        w_i = norm_weights[i]
        
        ckpt_i = torch.load(m_path, map_location='cpu')
        state_dict_i = ckpt_i['state_dict']
        
        for k in base_keys:
            if k in state_dict_i:
                running_sum[k] += state_dict_i[k] * w_i
                presence_weights[k] += w_i
                
        # Free memory after processing each model to remain robust against large N
        del ckpt_i 
        
    # 4. Finalize the average and assign back to merged_ckpt
    for k in base_keys:
        # Divide by the sum of weights of models that actually contained this key.
        # This mirrors the old logic perfectly: if a key was missing in later models,
        # it normalizes correctly to keep the original un-averaged Model A weights.
        merged_ckpt['state_dict'][k] = running_sum[k] / presence_weights[k]

    # Optional: Log the epoch we are merging at
    epoch = merged_ckpt.get('meta', {}).get('epoch', 'Unknown')
    print(f"Merging at end of Epoch: {epoch}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(merged_ckpt, output_path)
    print(f"Saved merged model with optimizer state to {output_path}")

if __name__ == '__main__':
    main()
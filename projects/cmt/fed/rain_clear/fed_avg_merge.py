import argparse
import torch
import shutil
import os
import copy

def parse_args():
    parser = argparse.ArgumentParser(description='Merge two models via FedAvg and preserve state')
    parser.add_argument('model_a', help='Path to checkpoint A')
    parser.add_argument('model_b', help='Path to checkpoint B')
    parser.add_argument('output', help='Path to save merged model')
    parser.add_argument('--weight-a', type=float, default=0.5, help='Weight for model A (0-1)')
    parser.add_argument('--weight-b', type=float, default=0.5, help='Weight for model B (0-1)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    total_weight = args.weight_a + args.weight_b
    wa = args.weight_a / total_weight
    wb = args.weight_b / total_weight

    print(f"Loading models for merge...\n A: {args.model_a} (w={wa:.2f})\n B: {args.model_b} (w={wb:.2f})")
    
    # Load all data (CPU to save memory)
    ckpt_a = torch.load(args.model_a, map_location='cpu')
    ckpt_b = torch.load(args.model_b, map_location='cpu')

    # --- UPDATED LOGIC: Deepcopy Model A ---
    # This preserves the exact structure of Model A (meta, optimizer, author, etc.)
    print("Deepcopying Model A to preserve full state...")
    merged_ckpt = copy.deepcopy(ckpt_a)

    state_dict_a = ckpt_a['state_dict']
    state_dict_b = ckpt_b['state_dict']
    
    # Iterate over keys in A (which are also in our merged_ckpt)
    print("Averaging weights...")
    for key in state_dict_a.keys():
        if key in state_dict_b:
            # Calculate the weighted average
            new_val = (state_dict_a[key] * wa) + (state_dict_b[key] * wb)
            
            # Update the value in the deepcopied checkpoint
            merged_ckpt['state_dict'][key] = new_val
        else:
            # If the key is missing in B, we keep the value from A.
            # Since merged_ckpt is a clone of A, we don't need to do anything here.
            pass

    # Optional: Log the epoch we are merging at
    epoch = merged_ckpt.get('meta', {}).get('epoch', 'Unknown')
    print(f"Merging at end of Epoch: {epoch}")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(merged_ckpt, args.output)
    print(f"Saved merged model with optimizer state to {args.output}")

if __name__ == '__main__':
    main()
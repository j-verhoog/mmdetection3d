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

    state_dict_a = ckpt_a['state_dict']
    state_dict_b = ckpt_b['state_dict']
    
    new_state_dict = {}
    
    # 1. Average the Model Weights
    for key in state_dict_a.keys():
        if key in state_dict_b:
            new_state_dict[key] = (state_dict_a[key] * wa) + (state_dict_b[key] * wb)
        else:
            new_state_dict[key] = state_dict_a[key]

    # 2. Preserve Optimizer & Meta from Client A
    # This tricks MMDet into thinking we are just continuing Client A's training,
    # but with the new averaged weights.
    new_ckpt = {
        'meta': ckpt_a.get('meta', {}),
        'optimizer': ckpt_a.get('optimizer', {}), # CRITICAL: Keeps momentum/Adam states
        'state_dict': new_state_dict
    }

    # Optional: Log the epoch we are merging at
    epoch = new_ckpt['meta'].get('epoch', 'Unknown')
    print(f"Merging at end of Epoch: {epoch}")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(new_ckpt, args.output)
    print(f"Saved merged model with optimizer state to {args.output}")

if __name__ == '__main__':
    main()
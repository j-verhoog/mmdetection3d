import argparse
import torch
import shutil
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Merge two models via FedAvg')
    parser.add_argument('model_a', help='Path to checkpoint A')
    parser.add_argument('model_b', help='Path to checkpoint B')
    parser.add_argument('output', help='Path to save merged model')
    parser.add_argument('--weight-a', type=float, default=0.5, help='Weight for model A (0-1)')
    parser.add_argument('--weight-b', type=float, default=0.5, help='Weight for model B (0-1)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Calculate normalization in case weights don't sum to 1
    total_weight = args.weight_a + args.weight_b
    wa = args.weight_a / total_weight
    wb = args.weight_b / total_weight

    print(f"Loading models...\n A: {args.model_a} (w={wa:.2f})\n B: {args.model_b} (w={wb:.2f})")
    
    # Load checkpoints (cpu is fine for merging, saves GPU memory)
    ckpt_a = torch.load(args.model_a, map_location='cpu')
    ckpt_b = torch.load(args.model_b, map_location='cpu')

    state_dict_a = ckpt_a['state_dict']
    state_dict_b = ckpt_b['state_dict']
    
    new_state_dict = {}
    
    # Perform weighted averaging
    for key in state_dict_a.keys():
        if key in state_dict_b:
            # Average weights
            new_state_dict[key] = (state_dict_a[key] * wa) + (state_dict_b[key] * wb)
        else:
            print(f"Warning: Key {key} not found in Model B. Keeping Model A value.")
            new_state_dict[key] = state_dict_a[key]
            
    # Create new checkpoint structure
    # We use meta from A, but you might want to reset epoch info if needed
    new_ckpt = {
        'meta': ckpt_a.get('meta', {}),
        'state_dict': new_state_dict
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    torch.save(new_ckpt, args.output)
    print(f"Saved merged model to {args.output}")

if __name__ == '__main__':
    main()
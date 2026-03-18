import os
import re
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BN_WEIGHT_COLOR = [0.90, 0.60, 0.00]
BN_BIAS_COLOR = [0.00, 0.45, 0.70]
BN_RUNNING_MEAN_COLOR = [0.80, 0.47, 0.65]
BN_RUNNING_VAR_COLOR = [0.15, 0.55, 0.25] 
BN_GENERIC_COLOR = [0.60, 0.60, 0.60] 
LAYER_NORM_BIAS_COLOR = [0.00, 0.45, 0.70]
LAYER_NORM_WEIGHT_COLOR = [0.90, 0.60, 0.00]
CONV_COLOR = [0.84, 0.15, 0.16]
ATTENTION_COLOR = [0.00, 0.62, 0.45]
LINEAR_MLP_HEAD_COLOR = [0.55, 0.35, 0.64]

CATEGORY_ORDER = [
    ('BN Weight', 'BN-w', BN_WEIGHT_COLOR),
    ('BN Bias', 'BN-b', BN_BIAS_COLOR),
    ('Linear / MLP / Heads', 'Lin', LINEAR_MLP_HEAD_COLOR),
    ('Layer Norm Weight', 'LN-w', LAYER_NORM_WEIGHT_COLOR),
    ('Layer Norm Bias', 'LN-b', LAYER_NORM_BIAS_COLOR),
    ('BN Running Mean', 'BN-rm', BN_RUNNING_MEAN_COLOR),
    ('BN Running Var', 'BN-rv', BN_RUNNING_VAR_COLOR),
    ('Convolution', 'Conv', CONV_COLOR),
    ('Attention', 'Attn', ATTENTION_COLOR),

]

CATEGORY_TO_COLOR = {label: color for label, _, color in CATEGORY_ORDER}


def _is_bn_indexed_module(parts, key_lower):
    """Detect indexed BN modules like ...deblocks.1.1.* or ...downsample.1.*."""
    if len(parts) < 2:
        return False
    parent = parts[-2]

    if len(parts) >= 3 and parts[-3] == 'downsample':
        return parent == '1'

    if 'pts_backbone' in key_lower or 'pts_neck' in key_lower or 'pts_middle_encoder' in key_lower:
        return parent in {'1', '4', '7', '10', '13', '16'}

    return False


def _is_conv_indexed_module(parts, key_lower):
    """Detect indexed Conv modules like ...deblocks.1.0.* or ...downsample.0.*."""
    if len(parts) < 2:
        return False
    parent = parts[-2]

    if len(parts) >= 3 and parts[-3] == 'downsample':
        return parent == '0'

    if 'pts_backbone' in key_lower or 'pts_neck' in key_lower or 'pts_middle_encoder' in key_lower:
        return parent in {'0', '3', '6', '9', '12', '15'}

    return False


def _natural_tokens(s):
    return [int(t) if t.isdigit() else t for t in re.findall(r'\d+|[^\d]+', s)]


def _suffix_priority(key_lower):
    """Sort params with module-aware suffix ordering."""
    if _is_layer_norm_key(key_lower):
        # Keep LayerNorm pairs together as bias -> weight for readability.
        if key_lower.endswith('.bias'):
            return 0
        if key_lower.endswith('.weight'):
            return 1
        return 9

    # BatchNorm/default ordering.
    if key_lower.endswith('.weight'):
        return 0
    if key_lower.endswith('.bias'):
        return 1
    if key_lower.endswith('.running_mean'):
        return 2
    if key_lower.endswith('.running_var') or key_lower.endswith('.running_vr'):
        return 3
    return 9


def _is_layer_norm_key(key_lower):
    """Detect LayerNorm-like keys while excluding BatchNorm families."""
    if 'running_mean' in key_lower or 'running_var' in key_lower or 'running_vr' in key_lower:
        return False
    if '.bn' in key_lower or 'batchnorm' in key_lower:
        return False
    return ('norms.' in key_lower) or ('.norm.' in key_lower) or key_lower.endswith('.post_norm.bias') or key_lower.endswith('.post_norm.weight')


def sort_layer_keys(keys):
    def logical_sort_key(k):
        # 1. Image Backbone & Pts Encoder: Rename '.bnX' to '.convX_bn'
        # Fixes: ...layer1.0.bn1 -> ...layer1.0.conv1_bn
        # Fixes: img_backbone.bn1 -> img_backbone.conv1_bn
        sort_str = re.sub(r'\.bn(\d+)\b', r'.conv\1_bn', k.lower())
        
        # 2. ResNet Downsamples: Rename downsample.1 (BN) to downsample.0_bn
        sort_str = re.sub(r'downsample\.1\b', r'downsample.0_bn', sort_str)
        
        # 3. Head Shared Conv: Rename shared_conv.bn to shared_conv.conv_bn
        sort_str = re.sub(r'shared_conv\.bn\b', r'shared_conv.conv_bn', sort_str)
        
        return (_natural_tokens(sort_str), _suffix_priority(k.lower()))
        
    return sorted(keys, key=logical_sort_key)

def get_layer_color(key):
    return CATEGORY_TO_COLOR[get_layer_category(key)]


def get_layer_category(key):
    """
    Maps a layer name to its semantic category used by both color and legend.
    Includes advanced parsing for unnamed sequences in PyTorch blocks.
    """
    key_lower = key.lower()
    parts = key_lower.split('.')

    suffix = parts[-1] if parts else ''
    parent = parts[-2] if len(parts) >= 2 else ''

    # BN detection: named BN modules and indexed BN modules.
    is_bn_named = ('.bn' in key_lower) or ('bn' in parent)
    is_bn_indexed = _is_bn_indexed_module(parts, key_lower)
    is_bn = is_bn_named or is_bn_indexed or ('running_' in key_lower)
    
    # 1. Batch Norms with granular BN shades.
    if is_bn:
        if suffix == 'running_mean':
            return 'BN Running Mean'
        if suffix in ('running_var', 'running_vr'):
            return 'BN Running Var'
        if suffix == 'weight':
            return 'BN Weight'
        if suffix == 'bias':
            return 'BN Bias'
        return 'Linear / MLP / Heads'
        
    # 2. Layer Norms (distinct colors for weight and bias)
    if _is_layer_norm_key(key_lower):
        if suffix == 'weight':
            return 'Layer Norm Weight'
        if suffix == 'bias':
            return 'Layer Norm Bias'
        return 'Layer Norm Weight'
        
    # 3. Attention
    if 'attn' in key_lower or 'attention' in key_lower:
        return 'Attention'
        
    # 4. Explicit/indexed Convolutions
    if 'conv' in key_lower or _is_conv_indexed_module(parts, key_lower):
        return 'Convolution'
                
    # 5. Default fallback (MLP / Linear / Heads / Embeddings)
    return 'Linear / MLP / Heads'

def format_params(num):
    """Formats large parameter counts into readable strings (e.g., 1.2M, 450K)."""
    if num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    return str(num)


def create_rgb_grid(mask_tensor, color_rgb, axis_aspect=1.0):
    """
    Reshapes a mask into a 2D RGB image array that perfectly matches 
    the physical aspect ratio of its bounding box. 
    Forces W=1 for 1D tensors (like biases and BN weights).
    """
    # Check if the parameter is natively a 1D tensor (e.g., BN params, biases)
    is_1d = len(mask_tensor.shape) == 1
    
    mask_np = mask_tensor.cpu().numpy().flatten()
    N = mask_np.size
    if N == 0:
        return np.zeros((1, 1, 3))
    
    # CHANGED: Force W=1 if the parameter is natively 1D
    if is_1d:
        W = 1
        H = N
    else:
        # Calculate W and H so the grid matches the axis shape exactly
        W = max(1, math.ceil(math.sqrt(N / axis_aspect)))
        H = math.ceil(N / W)
        
    pad_size = (H * W) - N
    
    # CHANGED: Use 0.96 (faint grey) instead of 1.0 for masked/pruned parameters
    R = np.where(mask_np > 0, color_rgb[0], 0.96)
    G = np.where(mask_np > 0, color_rgb[1], 0.96)
    B = np.where(mask_np > 0, color_rgb[2], 0.96)
    
    # Pad the remainder with 1.0 (pure white) so it blends perfectly with the background
    R = np.pad(R, (0, pad_size), constant_values=1.0).reshape((H, W))
    G = np.pad(G, (0, pad_size), constant_values=1.0).reshape((H, W))
    B = np.pad(B, (0, pad_size), constant_values=1.0).reshape((H, W))
    
    return np.stack([R, G, B], axis=-1)

def plot_module(fig, bounds, mask_dict, layer_keys, title):
    """
    Plots a sequence of layers inside a specific bounding box.
    """
    x0, y0, w, h = bounds
    
    # CHANGED: Filter out keys that don't exist OR have 0 parameters!
    layer_keys = [k for k in layer_keys if k in mask_dict and mask_dict[k].numel() > 0]
    if not layer_keys:
        return

    # Calculate statistics
    total_params = sum(mask_dict[k].numel() for k in layer_keys)
    masked_params = sum((mask_dict[k] > 0).sum().item() for k in layer_keys)
    masked_percentage = (masked_params / total_params) * 100 if total_params > 0 else 0

    # Per-category masked percentage (masked / total within each sub-category).
    category_totals = {label: 0 for label, _, _ in CATEGORY_ORDER}
    category_masked = {label: 0 for label, _, _ in CATEGORY_ORDER}
    for key in layer_keys:
        category = get_layer_category(key)
        if category not in category_totals:
            continue
        curr_mask = mask_dict[key]
        category_totals[category] += curr_mask.numel()
        category_masked[category] += (curr_mask > 0).sum().item()

    # Draw bounding box
    rect = patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor='black', facecolor='none', transform=fig.transFigure)
    fig.add_artist(rect)
    
    # --- PERFECT GIF ALIGNMENT ---
    center_x = x0 + w / 2
    title_y = y0 + h + 0.035
    stats_y = y0 + h + 0.01

    # 1. First Line: Title & Total Params (Split to prevent GIF jitter)
    # 1. First Line: Title & Total Params
    masked_str = format_params(masked_params)
    total_str = format_params(total_params)
    # Pad with figure spaces (\u2007) so the physical width never changes in a GIF
    padded_masked = masked_str.rjust(6, '\u2007') 
    title_text = f"{title} ({padded_masked} / {total_str} params)"
    fig.text(center_x, title_y, title_text, ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 2. Second Line: Pipe-separated category breakdown
    category_parts = []
    for label, short_label, color in CATEGORY_ORDER:
        denom = category_totals[label]
        if denom > 0: # Only include if it actually exists in this subplot
            pct = (category_masked[label] / denom) * 100
            # Clean formatting: "0%" instead of "0.0%", but keep "6.2%"
            pct_str = f"{int(pct)}%" if pct.is_integer() else f"{pct:.1f}%"
            category_parts.append(f"{short_label} {pct_str}")
            
    # Join them all together with a pipe and spaces
    category_line = " | ".join(category_parts)
    
    # Place the pipe-separated string right below the title
    fig.text(center_x, stats_y, category_line, ha='center', va='bottom', fontsize=10)
    # ----------------------------- 

    num_layers = len(layer_keys)
    gap = w * 0.05 / num_layers 
    layer_w = (w - (gap * (num_layers - 1))) / num_layers

    # CHANGED: Calculate the true physical aspect ratio of the sub-axes
    fig_w, fig_h = fig.get_size_inches()
    physical_layer_w = layer_w * fig_w
    physical_layer_h = h * fig_h
    axis_aspect = physical_layer_h / physical_layer_w if physical_layer_w > 0 else 1.0

    for i, key in enumerate(layer_keys):
        mask = mask_dict[key]
        layer_color = get_layer_color(key)
        
        # Pass the dynamic aspect ratio instead of a hardcoded target
        rgb_grid = create_rgb_grid(mask, layer_color, axis_aspect=axis_aspect) 
        
        lx = x0 + i * (layer_w + gap)
        ax = fig.add_axes([lx, y0, layer_w, h])
        
        ax.imshow(rgb_grid, aspect='auto', interpolation='nearest')
        ax.axis('off')

def visualize_model_masks(mask_path, output_image_path):
    print(f"Loading mask from {mask_path}...")
    mask_dict = torch.load(mask_path, map_location='cpu')

    # Strip out num_batches_tracked immediately
    mask_dict = {k: v for k, v in mask_dict.items() if 'num_batches_tracked' not in k}

    # Group layers based on keys. The upgraded sort_layer_keys perfectly orders EVERYTHING now!
    img_backbone_keys = sort_layer_keys([k for k in mask_dict.keys() if 'img_backbone' in k or 'img_neck' in k])
    pts_encoder_keys = sort_layer_keys([k for k in mask_dict.keys() if 'pts_middle_encoder' in k])
    pts_backbone_keys = sort_layer_keys([k for k in mask_dict.keys() if 'pts_backbone' in k or 'pts_neck' in k])
    decoder_keys = sort_layer_keys([k for k in mask_dict.keys() if 'pts_bbox_head.transformer.decoder' in k])
    embedding_keys = sort_layer_keys([k for k in mask_dict.keys() if 'embedding' in k or 'reference_points' in k])
    head_keys = sort_layer_keys([k for k in mask_dict.keys() if 'task_heads' in k or 'shared_conv' in k])

    decoder_hardcoded = [
        'pts_bbox_head.transformer.decoder.layers.0.attentions.0.attn.in_proj_weight',
        'pts_bbox_head.transformer.decoder.layers.0.attentions.0.attn.in_proj_bias',
        'pts_bbox_head.transformer.decoder.layers.0.attentions.0.attn.out_proj.weight',
        'pts_bbox_head.transformer.decoder.layers.0.attentions.0.attn.out_proj.bias',
        'pts_bbox_head.transformer.decoder.layers.0.norms.0.weight',
        'pts_bbox_head.transformer.decoder.layers.0.norms.0.bias',

        'pts_bbox_head.transformer.decoder.layers.0.attentions.1.attn.in_proj_weight',
        'pts_bbox_head.transformer.decoder.layers.0.attentions.1.attn.in_proj_bias',
        'pts_bbox_head.transformer.decoder.layers.0.attentions.1.attn.out_proj.weight',
        'pts_bbox_head.transformer.decoder.layers.0.attentions.1.attn.out_proj.bias',
        'pts_bbox_head.transformer.decoder.layers.0.norms.1.weight',
        'pts_bbox_head.transformer.decoder.layers.0.norms.1.bias',

        'pts_bbox_head.transformer.decoder.layers.0.ffns.0.layers.0.0.weight',
        'pts_bbox_head.transformer.decoder.layers.0.ffns.0.layers.0.0.bias',
        'pts_bbox_head.transformer.decoder.layers.0.ffns.0.layers.1.weight',
        'pts_bbox_head.transformer.decoder.layers.0.ffns.0.layers.1.bias',
        'pts_bbox_head.transformer.decoder.layers.0.norms.2.weight',
        'pts_bbox_head.transformer.decoder.layers.0.norms.2.bias',

        'pts_bbox_head.transformer.decoder.layers.1.attentions.0.attn.in_proj_weight',
        'pts_bbox_head.transformer.decoder.layers.1.attentions.0.attn.in_proj_bias',
        'pts_bbox_head.transformer.decoder.layers.1.attentions.0.attn.out_proj.weight',
        'pts_bbox_head.transformer.decoder.layers.1.attentions.0.attn.out_proj.bias',
        'pts_bbox_head.transformer.decoder.layers.1.norms.0.weight',
        'pts_bbox_head.transformer.decoder.layers.1.norms.0.bias',

        'pts_bbox_head.transformer.decoder.layers.1.attentions.1.attn.in_proj_weight',
        'pts_bbox_head.transformer.decoder.layers.1.attentions.1.attn.in_proj_bias',
        'pts_bbox_head.transformer.decoder.layers.1.attentions.1.attn.out_proj.weight',
        'pts_bbox_head.transformer.decoder.layers.1.attentions.1.attn.out_proj.bias',
        'pts_bbox_head.transformer.decoder.layers.1.norms.1.weight',
        'pts_bbox_head.transformer.decoder.layers.1.norms.1.bias',

        'pts_bbox_head.transformer.decoder.layers.1.ffns.0.layers.0.0.weight',
        'pts_bbox_head.transformer.decoder.layers.1.ffns.0.layers.0.0.bias',
        'pts_bbox_head.transformer.decoder.layers.1.ffns.0.layers.1.weight',
        'pts_bbox_head.transformer.decoder.layers.1.ffns.0.layers.1.bias',
        'pts_bbox_head.transformer.decoder.layers.1.norms.2.weight',
        'pts_bbox_head.transformer.decoder.layers.1.norms.2.bias',

        'pts_bbox_head.transformer.decoder.post_norm.weight',
        'pts_bbox_head.transformer.decoder.post_norm.bias',
    ]

    if all(k in decoder_keys for k in decoder_hardcoded):
        remaining_decoder_keys = [k for k in decoder_keys if k not in decoder_hardcoded]
        decoder_keys = decoder_hardcoded + remaining_decoder_keys
        
    # --- DEBUG PRINT BLOCK ---
    print("\n" + "="*60)
    # ... keep the rest of your debug and plotting logic exactly as it is!
    print("LAYER ALLOCATION & ORDERING DEBUG")
    print("="*60)
    
    # Reverse lookup dictionary to map colors back to category names for debugging
    color_to_name = {
        tuple(BN_WEIGHT_COLOR): "BN Weight",
        tuple(BN_BIAS_COLOR): "BN Bias",
        tuple(LINEAR_MLP_HEAD_COLOR): "Linear / MLP / Heads",
        tuple(LAYER_NORM_WEIGHT_COLOR): "Layer Norm Weight",
        tuple(LAYER_NORM_BIAS_COLOR): "Layer Norm Bias",
        tuple(BN_RUNNING_MEAN_COLOR): "BN Running Mean",
        tuple(BN_RUNNING_VAR_COLOR): "BN Running Var",
        tuple(CONV_COLOR): "Convolution",
        tuple(ATTENTION_COLOR): "Attention",

    }

    debug_groups = [
        ("Image Backbone & Neck", img_backbone_keys),
        ("Pts Middle Encoder", pts_encoder_keys),
        ("Pts Backbone & Neck", pts_backbone_keys),
        ("Transformer Decoder", decoder_keys),
        ("Embeddings & Ref Points", embedding_keys),
        ("Task Heads & Shared Conv", head_keys)
    ]
    
    for module_name, keys in debug_groups:
        print(f"\n>>> {module_name} ({len(keys)} layers):")
        for idx, k in enumerate(keys):
            # Fetch the color to figure out which logic branch this key hit
            assigned_color = tuple(get_layer_color(k))
            cat_name = color_to_name.get(assigned_color, "Unknown")
            print(f"    {idx+1:03d}: [{cat_name.ljust(20)}] {k}")
    print("="*60 + "\n")
    # -------------------------

    # Setup Figure
    fig = plt.figure(figsize=(24, 14))
    fig.patch.set_facecolor('white')

    # Row 1: Image Backbone spans the full width (Height 0.32 -> 0.40, Y-start 0.58 -> 0.50)
    plot_module(fig, [0.05, 0.50, 0.90, 0.40], mask_dict, img_backbone_keys, "Image Backbone & Neck")
    
    # Row 2: Four distinct columns (Heights 0.38 -> 0.285)
    # Column 1
    plot_module(fig, [0.05, 0.10, 0.2, 0.285], mask_dict, pts_encoder_keys, "Pts Middle Encoder")
    
    # Column 2
    plot_module(fig, [0.27, 0.10, 0.25, 0.285], mask_dict, pts_backbone_keys, "Pts Backbone & Neck")
    
    # Column 3 (Split vertically: Decodbuer top, Embeddings bottom)
    # Scaled proportionally to fit the 0.285 total row height
    plot_module(fig, [0.54, 0.22, 0.19, 0.165], mask_dict, decoder_keys, "Transformer Decoder")
    plot_module(fig, [0.54, 0.10, 0.19, 0.05], mask_dict, embedding_keys, "Embeddings & Ref Points")
    
    # Column 4
    plot_module(fig, [0.75, 0.10, 0.20, 0.285], mask_dict, head_keys, "Task Heads & Shared Conv")

    # Add Global Legend
    legend_elements = [patches.Patch(color=color, label=label) for label, _, color in CATEGORY_ORDER]
    fig.legend(handles=legend_elements, loc='upper right', ncol=4, fontsize=12, frameon=False, bbox_to_anchor=(0.98, 0.98))

    plt.text(0.02, 0.96, f"{plot_name}", transform=fig.transFigure, fontsize=16, fontweight='bold', va='top')

    # Save and close
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_image_path}")


if __name__ == "__main__":
    base_sample_mask_path = "/home/jolle/mmdet/visualisations/fedselect_masks/fedselect_masks/"
    base_output_path = "/home/jolle/mmdet/mmdetection3d/projects/analysis/cmt_fedselect_masking/outputs/all_final_no_elastic/"

    # base_sample_mask_path = "/home/jolle/mmdet/visualisations/fedselect_full_elastic/fedselect_masks/"
    # base_output_path = "/home/jolle/mmdet/mmdetection3d/projects/analysis/cmt_fedselect_masking/outputs/all_final_full_elastic/"

    if not os.path.exists(base_output_path):
        print(f"Output directory does not exist at {base_output_path}. Creating the folder.")
        os.makedirs(base_output_path, exist_ok=True)

    for model in ['A', 'B', 'C', 'D', 'E']:
        for curr_round in range(11):
            plot_name = f"Model {model} - Round {curr_round}"
            output_path = os.path.join(base_output_path, f"Round_{curr_round}_model_{model}.png")
            sample_mask_path = os.path.join(base_sample_mask_path, f"round_{curr_round}/Model{model}_mask.pt")

            if os.path.exists(output_path):
                print(f"Output image already exists at {output_path}. Skipping visualization for {plot_name}.")
                continue
            if os.path.exists(sample_mask_path):
                visualize_model_masks(sample_mask_path, output_path)
            else:
                print(f"Mask file not found at {sample_mask_path}. Please update the path.")

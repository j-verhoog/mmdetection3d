import argparse
from html import parser
import torch
import os
import string

def parse_args():
    parser = argparse.ArgumentParser(description='Merge N models via FedAvg/FedBN and preserve state')
    
    parser.add_argument('--inputs', nargs='+', required=True, help='Paths to input model checkpoints')
    parser.add_argument('--outputs', nargs='+', required=True, help='Paths to save the distinct merged models')
    
    # Optional method flag. Defaults to fedavg to preserve original behavior.
    parser.add_argument('--method', type=str, default='fedavg', 
                    choices=['fedavg', 'fedbn', 'fednorm', 'fedper', 'fed_bn_and_per', 'fedmedian', 'feddyn', 
                             'fed_dyn_bn_and_per', 'fedselect', 'fedselect_elastic', 'fedselect_fullelastic', 
                             'fedomg', 'fedomg_better', 'fedmc', 'fedomg_better_better'],
                    help='Aggregation method. Default is fedavg.')
    parser.add_argument('--config', type=str, default=None, 
                        help='Path to the MMDet3D config file (e.g., improved_lightweight_cmt_iterated.py). Required for FedBN.')
    parser.add_argument(
        '--select-ratio',
        type=float,
        default=0.05,
        help='FedSelect-only: fraction of total params selected per round (default: 0.05).')
    parser.add_argument(
        '--max-sparsity',
        type=float,
        default=0.4,
        help='FedSelect-only: max personalized parameter fraction (default: 0.4).')

    parser.add_argument('--fisher_paths', nargs='+', type=str, default=None, 
                    help='Paths to the saved Fisher Information tensors (Required ONLY for FedMC)')
    for i, char in enumerate(string.ascii_lowercase):
        help_text = f'Weight for model {char.upper()}' if i < 5 else argparse.SUPPRESS
        parser.add_argument(f'--weight-{char}', type=float, default=None, help=help_text)
        
    args = parser.parse_args()
    return args

import random # Add this to your imports at the top if not already there
import random

import random

def flatten_tensors(state_dict, valid_keys):
    """Flattens specified keys of a state_dict into a single 1D PyTorch tensor."""
    return torch.cat([state_dict[k].flatten() for k in valid_keys])

def unflatten_tensors(flat_tensor, reference_dict, valid_keys):
    """Restores a 1D tensor back into a state_dict dictionary structure."""
    unflattened = {}
    idx = 0
    for k in valid_keys:
        numel = reference_dict[k].numel()
        shape = reference_dict[k].shape
        unflattened[k] = flat_tensor[idx:idx+numel].view(shape).clone()
        idx += numel
    return unflattened

def fedomg(models, output_paths, norm_weights, client_ids, prev_global_path="/workspace/work_dirs/fedomg_states/global_model.pth"):
    """
FedOMG Implementation (ICLR 2025) - With Configurable Exclusions. NOT WOKRING CORRECLT. NOW IS PCGRADIENT PROJECTION WITHOUT THE MOMENTUM OPTIMIZATION. STILL BETTER THAN FEDAVG BUT NOT AS GOOD AS THE FULL OMG.
    """
    # =====================================================================
    # [CONFIGURATION] EXCLUSION SETTINGS
    # =====================================================================
    # By default, we exclude BatchNorm running stats from gradient matching.
    # You can add any layer prefix or string here to exclude it from FedOMG.
    # Any parameter whose name contains any of these strings will bypass the 
    # projection math and simply be aggregated via standard FedAvg.
    #
    # Example to exclude the detection head and LayerNorms:
    # EXCLUDE_PREFIXES = ['running_mean', 'running_var', 'num_batches_tracked', 'bbox_head', 'LayerNorm']
    # =====================================================================
    
    EXCLUDE_PREFIXES = ['bn','running_mean', 'running_var', 'num_batches_tracked', 'pts_bbox_head.task_heads']
    KEEP_PRIVATE = True     # keeps the excluded prefixes strictly local and does not merge them into the client models

    print("\n" + "="*50)
    print("Running FedOMG (On-Server Matching Gradient) Aggregation...")
    print("="*50)
    
    os.makedirs(os.path.dirname(prev_global_path), exist_ok=True)
    num_clients = len(models)

    if not os.path.exists(prev_global_path):
        print(f"[WARNING] No previous global model found at {prev_global_path}.")
        print("Initializing FedOMG baseline by running a standard FedAvg for Round 0...")
        fedavg(models, [prev_global_path] * num_clients, norm_weights) 
        for in_path, out_path in zip(models, output_paths):
             os.system(f"cp {prev_global_path} {out_path}")
        return

    global_ckpt = torch.load(prev_global_path, map_location='cpu')
    global_state = global_ckpt['state_dict']
    
    # Strictly separate FedOMG keys from standard FedAvg keys based on your config
    fedomg_keys = []
    fedavg_keys = []
    
    for k, v in global_state.items():
        if v.is_floating_point():
            if any(excl in k for excl in EXCLUDE_PREFIXES):
                fedavg_keys.append(k)
            else:
                fedomg_keys.append(k)

    total_params = sum(global_state[k].numel() for k in fedomg_keys)
    print(f"FedOMG: Conflict tracking on {len(fedomg_keys)} layers ({total_params:,} true weights).")
    if not KEEP_PRIVATE:
        print(f"FedAvg: Standard averaging on {len(fedavg_keys)} excluded layers/trackers.")
    else:
        print(f"Not including {len(fedavg_keys)} private layers.")

    # 1. Compute Pseudo-Gradients AND Accumulate FedAvg layers
    print("\n--- Phase 1: Computing Pseudo-Gradients ---")
    flat_global = flatten_tensors(global_state, fedomg_keys)
    
    client_pseudo_grads = []
    fedavg_accumulators = {k: torch.zeros_like(global_state[k]) for k in fedavg_keys}
    
    for i, (m_path, cid, w_i) in enumerate(zip(models, client_ids, norm_weights)):
        ckpt_i = torch.load(m_path, map_location='cpu')
        state_i = ckpt_i['state_dict']
        
        # Extract FedOMG gradient
        flat_client = flatten_tensors(state_i, fedomg_keys)
        pseudo_grad = flat_global - flat_client
        client_pseudo_grads.append(pseudo_grad)
        
        grad_norm = torch.norm(pseudo_grad).item()
        print(f"  [{cid}] Extracted true weight gradient. L2 Norm: {grad_norm:.4f}")
        
        # Accumulate FedAvg layers
        for k in fedavg_keys:
            if k in state_i:
                fedavg_accumulators[k] += state_i[k] * w_i
                
        del ckpt_i, flat_client

    # 2. Conflict Resolution via Gradient Projection
    print("\n--- Phase 2: Resolving Domain Conflicts (Gradient Matching) ---")
    total_conflicts_resolved = 0
    
    for i in range(num_clients):
        check_order = list(range(num_clients))
        random.shuffle(check_order)
        
        for j in check_order:
            if i == j: 
                continue
            
            dot_product = torch.dot(client_pseudo_grads[i], client_pseudo_grads[j]).item()
            cos_sim = dot_product / (torch.norm(client_pseudo_grads[i]).item() * torch.norm(client_pseudo_grads[j]).item() + 1e-8)
            print(f"  [EVAL] {client_ids[i]} vs {client_ids[j]} -> Cosine Sim: {cos_sim:.4f}")
            
            if dot_product < 0:
                total_conflicts_resolved += 1
                norm_sq_j = torch.dot(client_pseudo_grads[j], client_pseudo_grads[j]).item() + 1e-8
                
                print(f"      [!] CONFLICT DETECTED. Projecting {client_ids[i]} away from {client_ids[j]}.")
                
                projection_scalar = dot_product / norm_sq_j
                client_pseudo_grads[i] = client_pseudo_grads[i] - (projection_scalar * client_pseudo_grads[j])

    print(f"\nFedOMG Phase 2 Complete. Total pairwise conflicts resolved: {total_conflicts_resolved}")

    # 3. Aggregate Aligned Gradients
    print("\n--- Phase 3: Aggregating Aligned Gradients ---")
    aggregated_grad = torch.zeros_like(flat_global)
    for i, w_i in enumerate(norm_weights):
        aggregated_grad += client_pseudo_grads[i] * w_i
        
    final_grad_norm = torch.norm(aggregated_grad).item()
    print(f"  Aggregated Global Gradient L2 Norm: {final_grad_norm:.4f}")

    # 4. Update Global Model 
    new_flat_global = flat_global - aggregated_grad
    new_global_weights = unflatten_tensors(new_flat_global, global_state, fedomg_keys)
    
    # Merge both FedOMG weights and FedAvg excluded weights back in
    for k in fedomg_keys:
        global_ckpt['state_dict'][k] = new_global_weights[k]
    for k in fedavg_keys:
        global_ckpt['state_dict'][k] = fedavg_accumulators[k]
        
    print(f"  Saving new conflict-free global model to {prev_global_path}")
    torch.save(global_ckpt, prev_global_path)

    # 5. Distribute
    print("\n--- Phase 4: Distributing Global Model ---")
    for in_path, out_path, cid in zip(models, output_paths, client_ids):
        ckpt = torch.load(in_path, map_location='cpu')
        
        for k in global_ckpt['state_dict'].keys():
            if not k in fedavg_keys:                # Only overwrite FedOMG keys, keep FedAvg keys local
                if k in ckpt['state_dict']:
                    ckpt['state_dict'][k] = global_ckpt['state_dict'][k].clone()
                
        if 'optimizer' in ckpt and 'state' in ckpt['optimizer']:
            for param_id in ckpt['optimizer']['state']:
                for key in ckpt['optimizer']['state'][param_id]:
                    if torch.is_tensor(ckpt['optimizer']['state'][param_id][key]):
                        ckpt['optimizer']['state'][param_id][key].zero_()
                        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(ckpt, out_path)
        print(f"  Saved FedOMG synchronized model for {cid} to {out_path}")
        
    print("="*50)
    print("FedOMG Aggregation Complete.\n")

def fedomg_better(models, output_paths, norm_weights, client_ids, prev_global_path="/workspace/work_dirs/fedomg_states/global_model.pth"):
    EPS = 1e-12

    KAPPA = 0.5
    ETA_G = 1.0
    OMG_INNER_ITERS = 50 
    OMG_LR = 0.1           
    OMG_MOMENTUM = 0.5

    USE_EXCLUSION = True
    EXCLUDE_PREFIXES = ['bn', 'running_mean', 'running_var', 'num_batches_tracked', 'pts_bbox_head.task_heads']
    KEEP_PRIVATE = True

    # =================================================================
    # NEW: Dynamic Min/Max Gamma Bounds
    # =================================================================
    # This factor dynamically computes the floor and ceiling based on dataset size.
    # min_gamma = norm_weight / GAMMA_BOUND_FACTOR
    # max_gamma = norm_weight * GAMMA_BOUND_FACTOR
    GAMMA_BOUND_FACTOR = 5.0 

    print("\n" + "=" * 50)
    print("Running FedOMG (On-Server Matching Gradient) Aggregation...")
    print(f"Dynamic Bounds Active: Max/Min bounded by factor of {GAMMA_BOUND_FACTOR}x")
    print("=" * 50)

    os.makedirs(os.path.dirname(prev_global_path), exist_ok=True)
    num_clients = len(models)

    if not os.path.exists(prev_global_path):
        print(f"[WARNING] No previous global model found at {prev_global_path}.")
        print("Initializing FedOMG baseline by running a standard FedAvg for Round 0...")
        # Assuming you have a standard fedavg() function in scope
        fedavg(models, [prev_global_path] * num_clients, norm_weights)
        for in_path, out_path in zip(models, output_paths):
            os.system(f"cp {prev_global_path} {out_path}")
        return

    global_ckpt = torch.load(prev_global_path, map_location="cpu")
    global_state = global_ckpt["state_dict"]

    fedomg_keys = []
    excluded_keys = []

    for k, v in global_state.items():
        if torch.is_tensor(v) and v.is_floating_point():
            if USE_EXCLUSION and any(excl in k for excl in EXCLUDE_PREFIXES):
                excluded_keys.append(k)
            else:
                fedomg_keys.append(k)

    total_params = sum(global_state[k].numel() for k in fedomg_keys)
    print(f"FedOMG: Tracking {len(fedomg_keys)} floating-point layers ({total_params:,} true weights).")

    if USE_EXCLUSION:
        if KEEP_PRIVATE:
            print(f"Not including {len(excluded_keys)} excluded/private layers.")
        else:
            print(f"FedAvg: Standard averaging on {len(excluded_keys)} excluded layers/trackers.")

    print("\n--- Phase 1: Computing Local Gradients ---")
    flat_global = flatten_tensors(global_state, fedomg_keys)

    client_grads = []
    client_states = []
    excluded_accumulators = {k: torch.zeros_like(global_state[k]) for k in excluded_keys} if (USE_EXCLUSION and not KEEP_PRIVATE) else None

    for m_path, cid, w_i in zip(models, client_ids, norm_weights):
        ckpt_i = torch.load(m_path, map_location="cpu")
        state_i = ckpt_i["state_dict"]
        client_states.append(ckpt_i)

        flat_client = flatten_tensors(state_i, fedomg_keys)
        local_grad = flat_client - flat_global
        client_grads.append(local_grad)

        grad_norm = torch.norm(local_grad).item()
        print(f"  [{cid}] Extracted true local gradient. L2 Norm: {grad_norm:.4f}")

        if USE_EXCLUSION and not KEEP_PRIVATE:
            for k in excluded_keys:
                if k in state_i:
                    excluded_accumulators[k] += state_i[k] * w_i

        del flat_client

    G = torch.stack(client_grads, dim=0)

    print("\n--- Phase 2: Solving FedOMG On-Server Objective ---")
    print("\n HOT UPDATE!!!!!!!!!!!!!!!!! Optimizing Gamma coefficients with normalized grads and then using unscaled for update!!!!!!!!!!!!!!!!!!!!")
    weight_tensor = torch.tensor(norm_weights, dtype=flat_global.dtype, device=flat_global.device)
    weight_tensor = weight_tensor / (weight_tensor.sum() + EPS)

    g_fl = (weight_tensor[:, None] * G).sum(dim=0)

    g_fl_norm = torch.norm(g_fl)
    print(f"  Reference FedAvg Gradient L2 Norm: {g_fl_norm.item():.4f}")

    if g_fl_norm.item() < EPS:
        print("  [WARNING] FedAvg reference gradient is near zero. Falling back to g_FL only.")
        g_igd = g_fl.clone()
        gamma_star = weight_tensor.clone()
        combo = g_fl.clone()
    else:
        # Dynamically calculate the Minimum and Maximum allowed gammas based on dataset size
        min_gamma_tensor = weight_tensor / GAMMA_BOUND_FACTOR
        max_gamma_tensor = torch.clamp(weight_tensor * GAMMA_BOUND_FACTOR, max=1.0)
        
        # Force all gradients to have the exact same magnitude (g_fl_norm)
        # This prevents the solver from "cheating" by picking small-norm gradients.
        G_norms = torch.norm(G, dim=1, keepdim=True) + EPS
        G_normalized = (G / G_norms) * g_fl_norm

        # Safety check: Ensure the minimums don't sum to > 1.0
        sum_min = min_gamma_tensor.sum().item()
        if sum_min >= 1.0:
            print(f"  [WARNING] Sum of min_gammas ({sum_min}) >= 1.0! Scaling down to ensure mathematical stability.")
            min_gamma_tensor = (min_gamma_tensor / sum_min) * 0.99
            sum_min = 0.99
            
        remaining_gamma_pool = 1.0 - sum_min

        logits = torch.log(weight_tensor + EPS).clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([logits], lr=OMG_LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=OMG_INNER_ITERS, eta_min=1e-4)

        best_loss = None
        best_gamma = None
        best_combo = None

        for it in range(OMG_INNER_ITERS):
            optimizer.zero_grad()
            
            # The Affine Softmax Transformation guarantees the floor and sum=1.0
            raw_gamma = torch.softmax(logits, dim=0)
            gamma = min_gamma_tensor + (remaining_gamma_pool * raw_gamma)
            
            # used this untill round 5
            # combo = (gamma[:, None] * G).sum(dim=0)
            # combo_norm = torch.norm(combo) + EPS

            # Use the NORMALIZED gradients to compute the loss objective
            combo_normalized = (gamma[:, None] * G_normalized).sum(dim=0)
            combo_norm_normalized = torch.norm(combo_normalized) + EPS
            combo = combo_normalized
            combo_norm = combo_norm_normalized

            
            # Base optimization objective
            base_loss = torch.dot(combo, g_fl) + KAPPA * g_fl_norm * combo_norm
            
            # NEW: Lagrangian Penalty to strictly enforce the ceiling (max_gamma)
            # If gamma exceeds max_gamma, it multiplies the excess by 10000, slamming the loss upward.
            max_penalty = 10000.0 * torch.relu(gamma - max_gamma_tensor).sum()
            loss = base_loss + max_penalty

            loss.backward()
            optimizer.step()
            scheduler.step()

            current_loss = loss.item()
            if best_loss is None or current_loss < best_loss:
                best_loss = current_loss
                best_gamma = gamma.detach().clone()
                best_combo = combo.detach().clone()

            current_lr = scheduler.get_last_lr()[0]

            print(
                f"  [OMG {it + 1:02d}/{OMG_INNER_ITERS}] "
                f"loss={loss.item():.6f} | "
                f"||Gamma g||={combo_norm.item():.4f} | "
                f"lr={current_lr:.4f}"
            )

        gamma_star = best_gamma

        # =================================================================
        # CRITICAL FIX: Apply the optimal Gamma* back to the RAW true gradients
        # =================================================================
        final_raw_combo = (gamma_star[:, None] * G).sum(dim=0)
        final_raw_combo_norm = torch.norm(final_raw_combo) + EPS

        if final_raw_combo_norm.item() < EPS:
            print("  [WARNING] Optimized Gamma*g is near zero. Using g_FL only.")
            g_igd = g_fl.clone()
        else:
            # Construct g_igd using the TRUE magnitude combination
            g_igd = g_fl + KAPPA * g_fl_norm * (final_raw_combo / final_raw_combo_norm)

        print("  Gamma* coefficients:")
        for cid, gamma_i, max_g, min_g in zip(client_ids, gamma_star.tolist(), max_gamma_tensor.tolist(), min_gamma_tensor.tolist()):
            print(f"    {cid}: {gamma_i:.6f} (Limits: min {min_g:.4f}, max {max_g:.4f})")

    print("\n--- Phase 3: Aggregating Aligned Gradients ---")
    print(f"  Optimized Combined Gradient ||Gamma* g|| (Raw): {final_raw_combo_norm.item():.4f}")
    igd_norm = torch.norm(g_igd).item()
    print(f"  Final Invariant Gradient L2 Norm: {igd_norm:.4f}")

    print("\n--- Phase 3b: Extracting and Saving Domain-Specific (Orthogonal) Gradients ---")
    domain_specific_grads = {}
    igd_norm_sq = torch.dot(g_igd, g_igd) + EPS
    
    for i, cid in enumerate(client_ids):
        g_i = client_grads[i]
        
        proj_scalar = torch.dot(g_i, g_igd) / igd_norm_sq
        g_i_ortho = g_i - (proj_scalar * g_igd)
        domain_specific_grads[cid] = g_i_ortho
        
        ortho_norm = torch.norm(g_i_ortho).item()
        print(f"  [{cid}] Domain-specific (Orthogonal) Gradient L2 Norm: {ortho_norm:.4f}")
        
        ortho_state_dict = unflatten_tensors(g_i_ortho, global_state, fedomg_keys)
        client_out_path = output_paths[i]
        ortho_save_path = client_out_path.replace(".pth", "_ortho.pth")
        
        torch.save({"state_dict": ortho_state_dict}, ortho_save_path)
        print(f"    -> Saved orthogonal component for {cid} to {ortho_save_path}")

    print("\n--- Phase 4: Updating Global Model ---")
    new_flat_global = flat_global + (ETA_G * g_igd)
    new_global_weights = unflatten_tensors(new_flat_global, global_state, fedomg_keys)

    for k in fedomg_keys:
        global_ckpt["state_dict"][k] = new_global_weights[k]

    if USE_EXCLUSION and not KEEP_PRIVATE:
        for k in excluded_keys:
            global_ckpt["state_dict"][k] = excluded_accumulators[k]

    print(f"  Saving new FedOMG global model to {prev_global_path}")
    torch.save(global_ckpt, prev_global_path)

    print("\n--- Phase 5: Distributing Global Model ---")
    for ckpt, out_path, cid in zip(client_states, output_paths, client_ids):
        if KEEP_PRIVATE and USE_EXCLUSION:
            for k in fedomg_keys:
                if k in ckpt["state_dict"]:
                    ckpt["state_dict"][k] = global_ckpt["state_dict"][k].clone()
        else:
            for k in global_ckpt["state_dict"].keys():
                if k in ckpt["state_dict"] and (
                    (k in fedomg_keys) or
                    (USE_EXCLUSION and not KEEP_PRIVATE and k in excluded_keys)
                ):
                    ckpt["state_dict"][k] = global_ckpt["state_dict"][k].clone()

        if "optimizer" in ckpt and "state" in ckpt["optimizer"]:
            for param_id in ckpt["optimizer"]["state"]:
                for key in ckpt["optimizer"]["state"][param_id]:
                    if torch.is_tensor(ckpt["optimizer"]["state"][param_id][key]):
                        ckpt["optimizer"]["state"][param_id][key].zero_()

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(ckpt, out_path)
        print(f"  Saved FedOMG synchronized model for {cid} to {out_path}")

    print("=" * 50)
    print("FedOMG Aggregation Complete.\n")

def fedomg_better_better(models, output_paths, norm_weights, client_ids, prev_global_path="/workspace/work_dirs/fedomg_states/global_model.pth"):
    EPS = 1e-12

    KAPPA = 0.5
    ETA_G = 1.0
    OMG_INNER_ITERS = 50
    OMG_LR = 0.1
    OMG_MOMENTUM = 0.5

    # ================================================================
    # TOGGLES FOR NON-NATIVE FEDOMG FUNCTIONALITY
    # ================================================================
    # Native FedOMG paper defaults would roughly be:
    #   OMG_INNER_ITERS = 21
    #   OMG_LR = 25
    #   KAPPA = 0.5
    #   momentum = 0.5
    #   direct optimization on raw G
    #   no exclusion/private layers
    #   no gamma lower/upper bounds
    #   no ceiling penalty
    #   no orthogonal residual save
    #   global update uses theta_new = theta_old - eta_g * g_igd
    #
    # These toggles let you keep every extra feature without removing logic.
    # ================================================================
    USE_EXCLUSION = True
    EXCLUDE_PREFIXES = ['bn', 'running_mean', 'running_var', 'num_batches_tracked', 'pts_bbox_head.task_heads']
    KEEP_PRIVATE = True

    USE_DYNAMIC_GAMMA_BOUNDS = True
    GAMMA_BOUND_FACTOR = 5.0

    USE_NORMALIZED_GRADS_FOR_GAMMA_SOLVE = True
    APPLY_GAMMA_BACK_TO_RAW_GRADS = True

    USE_MAX_GAMMA_PENALTY = True
    MAX_GAMMA_PENALTY_WEIGHT = 10000.0

    USE_ADAM_SOLVER = True
    USE_COSINE_SCHEDULER = True

    SAVE_ORTHOGONAL_COMPONENTS = True

    USE_PAPER_SIGN_UPDATE = False  # False preserves your current code behavior (+). True switches to paper-faithful (-).

    USE_PAPER_REPORTED_HYPERS = False  # convenience toggle
    if USE_PAPER_REPORTED_HYPERS:
        OMG_INNER_ITERS = 21
        OMG_LR = 25.0
        OMG_MOMENTUM = 0.5
        KAPPA = 0.5

    # ================================================================
    # PAPER-FAITHFUL TOGGLE SETTING BLOCK
    # ================================================================
    USE_PAPER_FAITHFUL_MODE = True

    if USE_PAPER_FAITHFUL_MODE:
        KAPPA = 0.5
        ETA_G = 1.0
        OMG_INNER_ITERS = 50                    # changed
        OMG_LR = 1.0                           # changed
        OMG_MOMENTUM = 0.5

        USE_EXCLUSION = False
        KEEP_PRIVATE = False

        USE_DYNAMIC_GAMMA_BOUNDS = False
        GAMMA_BOUND_FACTOR = 50.0

        USE_NORMALIZED_GRADS_FOR_GAMMA_SOLVE = False
        APPLY_GAMMA_BACK_TO_RAW_GRADS = True

        USE_MAX_GAMMA_PENALTY = False
        MAX_GAMMA_PENALTY_WEIGHT = 10000.0

        USE_ADAM_SOLVER = True                 # changed
        USE_COSINE_SCHEDULER = True            # changed   

        SAVE_ORTHOGONAL_COMPONENTS = False

        USE_PAPER_SIGN_UPDATE = True
    # ================================================================

    PRINT_ALL_SETTINGS = True
    if PRINT_ALL_SETTINGS:
        print("\n" + "=" * 50)
        print("FedOMG Configuration Settings:")
        print(f"  KAPPA: {KAPPA}, ETA_G: {ETA_G}, OMG_INNER_ITERS: {OMG_INNER_ITERS}, OMG_LR: {OMG_LR}, OMG_MOMENTUM: {OMG_MOMENTUM}")
        print(f"  USE_EXCLUSION: {USE_EXCLUSION}, EXCLUDE_PREFIXES: {EXCLUDE_PREFIXES}, KEEP_PRIVATE: {KEEP_PRIVATE} ")
        print(f"  USE_DYNAMIC_GAMMA_BOUNDS: {USE_DYNAMIC_GAMMA_BOUNDS}, GAMMA_BOUND_FACTOR: {GAMMA_BOUND_FACTOR}")
        print(f"  USE_NORMALIZED_GRADS_FOR_GAMMA_SOLVE: {USE_NORMALIZED_GRADS_FOR_GAMMA_SOLVE}, APPLY_GAMMA_BACK_TO_RAW_GRADS: {APPLY_GAMMA_BACK_TO_RAW_GRADS}")
        print(f"  USE_MAX_GAMMA_PENALTY: {USE_MAX_GAMMA_PENALTY}, MAX_GAMMA_PENALTY_WEIGHT: {MAX_GAMMA_PENALTY_WEIGHT}")
        print(f"  USE_ADAM_SOLVER: {USE_ADAM_SOLVER}, USE_COSINE_SCHEDULER: {USE_COSINE_SCHEDULER}")
        print(f"  SAVE_ORTHOGONAL_COMPONENTS: {SAVE_ORTHOGONAL_COMPONENTS}")
        print(f"  USE_PAPER_SIGN_UPDATE: {USE_PAPER_SIGN_UPDATE}")
        print(f"  USE_PAPER_REPORTED_HYPERS: {USE_PAPER_REPORTED_HYPERS}")
        print(f"  USE_PAPER_FAITHFUL_MODE: {USE_PAPER_FAITHFUL_MODE}")

    print("\n" + "=" * 50)
    print("Running FedOMG (On-Server Matching Gradient) Aggregation...")
    if USE_DYNAMIC_GAMMA_BOUNDS:
        print(f"Dynamic Bounds Active: Max/Min bounded by factor of {GAMMA_BOUND_FACTOR}x")
    else:
        print("Dynamic Bounds Active: OFF")
    print("=" * 50)

    os.makedirs(os.path.dirname(prev_global_path), exist_ok=True)
    num_clients = len(models)

    if not os.path.exists(prev_global_path):
        print(f"[WARNING] No previous global model found at {prev_global_path}.")
        print("Initializing FedOMG baseline by running a standard FedAvg for Round 0...")
        fedavg(models, [prev_global_path] * num_clients, norm_weights)
        for in_path, out_path in zip(models, output_paths):
            os.system(f"cp {prev_global_path} {out_path}")
        return

    global_ckpt = torch.load(prev_global_path, map_location="cpu")
    global_state = global_ckpt["state_dict"]

    fedomg_keys = []
    excluded_keys = []

    for k, v in global_state.items():
        if torch.is_tensor(v) and v.is_floating_point():
            if USE_EXCLUSION and any(excl in k for excl in EXCLUDE_PREFIXES):
                excluded_keys.append(k)
            else:
                fedomg_keys.append(k)

    total_params = sum(global_state[k].numel() for k in fedomg_keys)
    print(f"FedOMG: Tracking {len(fedomg_keys)} floating-point layers ({total_params:,} true weights).")

    if USE_EXCLUSION:
        if KEEP_PRIVATE:
            print(f"Not including {len(excluded_keys)} excluded/private layers.")
        else:
            print(f"FedAvg: Standard averaging on {len(excluded_keys)} excluded layers/trackers.")

    print("\n--- Phase 1: Computing Local Gradients ---")
    flat_global = flatten_tensors(global_state, fedomg_keys)

    client_grads = []
    client_states = []
    excluded_accumulators = {k: torch.zeros_like(global_state[k]) for k in excluded_keys} if (USE_EXCLUSION and not KEEP_PRIVATE) else None

    for m_path, cid, w_i in zip(models, client_ids, norm_weights):
        ckpt_i = torch.load(m_path, map_location="cpu")
        state_i = ckpt_i["state_dict"]
        client_states.append(ckpt_i)

        flat_client = flatten_tensors(state_i, fedomg_keys)
        local_grad = flat_client - flat_global
        client_grads.append(local_grad)

        grad_norm = torch.norm(local_grad).item()
        print(f"  [{cid}] Extracted true local gradient. L2 Norm: {grad_norm:.4f}")

        if USE_EXCLUSION and not KEEP_PRIVATE:
            for k in excluded_keys:
                if k in state_i:
                    excluded_accumulators[k] += state_i[k] * w_i

        del flat_client

    G = torch.stack(client_grads, dim=0)

    print("\n--- Phase 2: Solving FedOMG On-Server Objective ---")
    print("\n HOT UPDATE!!!!!!!!!!!!!!!!! Optimizing Gamma coefficients with normalized grads and then using unscaled for update!!!!!!!!!!!!!!!!!!!!")

    weight_tensor = torch.tensor(norm_weights, dtype=flat_global.dtype, device=flat_global.device)
    weight_tensor = weight_tensor / (weight_tensor.sum() + EPS)

    g_fl = (weight_tensor[:, None] * G).sum(dim=0)

    g_fl_norm = torch.norm(g_fl)
    print(f"  Reference FedAvg Gradient L2 Norm: {g_fl_norm.item():.4f}")

    final_raw_combo = g_fl.clone()
    final_raw_combo_norm = torch.norm(final_raw_combo) + EPS

    if g_fl_norm.item() < EPS:
        print("  [WARNING] FedAvg reference gradient is near zero. Falling back to g_FL only.")
        g_igd = g_fl.clone()
        gamma_star = weight_tensor.clone()
        combo = g_fl.clone()
    else:
        if USE_DYNAMIC_GAMMA_BOUNDS:
            min_gamma_tensor = weight_tensor / GAMMA_BOUND_FACTOR
            max_gamma_tensor = torch.clamp(weight_tensor * GAMMA_BOUND_FACTOR, max=1.0)

            sum_min = min_gamma_tensor.sum().item()
            if sum_min >= 1.0:
                print(f"  [WARNING] Sum of min_gammas ({sum_min}) >= 1.0! Scaling down to ensure mathematical stability.")
                min_gamma_tensor = (min_gamma_tensor / sum_min) * 0.99
                sum_min = 0.99

            remaining_gamma_pool = 1.0 - sum_min
        else:
            min_gamma_tensor = torch.zeros_like(weight_tensor)
            max_gamma_tensor = torch.ones_like(weight_tensor)
            remaining_gamma_pool = 1.0

        if USE_NORMALIZED_GRADS_FOR_GAMMA_SOLVE:
            G_norms = torch.norm(G, dim=1, keepdim=True) + EPS
            G_solver = (G / G_norms) * g_fl_norm
        else:
            G_solver = G

        logits = torch.log(weight_tensor + EPS).clone().detach().requires_grad_(True)

        if USE_ADAM_SOLVER:
            optimizer = torch.optim.Adam([logits], lr=OMG_LR)
        else:
            optimizer = torch.optim.SGD([logits], lr=OMG_LR, momentum=OMG_MOMENTUM)

        if USE_COSINE_SCHEDULER:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=OMG_INNER_ITERS, eta_min=1e-4)
        else:
            scheduler = None

        best_loss = None
        best_gamma = None
        best_combo = None

        for it in range(OMG_INNER_ITERS):
            optimizer.zero_grad()

            raw_gamma = torch.softmax(logits, dim=0)

            if USE_DYNAMIC_GAMMA_BOUNDS:
                gamma = min_gamma_tensor + (remaining_gamma_pool * raw_gamma)
            else:
                gamma = raw_gamma

            combo = (gamma[:, None] * G_solver).sum(dim=0)
            combo_norm = torch.norm(combo) + EPS

            base_loss = torch.dot(combo, g_fl) + KAPPA * g_fl_norm * combo_norm

            if USE_MAX_GAMMA_PENALTY:
                max_penalty = MAX_GAMMA_PENALTY_WEIGHT * torch.relu(gamma - max_gamma_tensor).sum()
            else:
                max_penalty = torch.zeros((), dtype=base_loss.dtype, device=base_loss.device)

            loss = base_loss + max_penalty

            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimizer.param_groups[0]["lr"]

            current_loss = loss.item()
            if best_loss is None or current_loss < best_loss:
                best_loss = current_loss
                best_gamma = gamma.detach().clone()
                best_combo = combo.detach().clone()

            print(
                f"  [OMG {it + 1:02d}/{OMG_INNER_ITERS}] "
                f"loss={loss.item():.6f} | "
                f"||Gamma g||={combo_norm.item():.4f} | "
                f"lr={current_lr:.4f}"
            )

        gamma_star = best_gamma

        if APPLY_GAMMA_BACK_TO_RAW_GRADS:
            final_raw_combo = (gamma_star[:, None] * G).sum(dim=0)
            final_raw_combo_norm = torch.norm(final_raw_combo) + EPS
        else:
            final_raw_combo = best_combo
            final_raw_combo_norm = torch.norm(final_raw_combo) + EPS

        if final_raw_combo_norm.item() < EPS:
            print("  [WARNING] Optimized Gamma*g is near zero. Using g_FL only.")
            g_igd = g_fl.clone()
        else:
            g_igd = g_fl + KAPPA * g_fl_norm * (final_raw_combo / final_raw_combo_norm)

        print("  Gamma* coefficients:")
        for cid, gamma_i, max_g, min_g in zip(client_ids, gamma_star.tolist(), max_gamma_tensor.tolist(), min_gamma_tensor.tolist()):
            print(f"    {cid}: {gamma_i:.6f} (Limits: min {min_g:.4f}, max {max_g:.4f})")

    print("\n--- Phase 3: Aggregating Aligned Gradients ---")
    print(f"  Optimized Combined Gradient ||Gamma* g|| (Raw): {final_raw_combo_norm.item():.4f}")
    igd_norm = torch.norm(g_igd).item()
    print(f"  Final Invariant Gradient L2 Norm: {igd_norm:.4f}")

    if SAVE_ORTHOGONAL_COMPONENTS:
        print("\n--- Phase 3b: Extracting and Saving Domain-Specific (Orthogonal) Gradients ---")
        domain_specific_grads = {}
        igd_norm_sq = torch.dot(g_igd, g_igd) + EPS

        for i, cid in enumerate(client_ids):
            g_i = client_grads[i]

            proj_scalar = torch.dot(g_i, g_igd) / igd_norm_sq
            g_i_ortho = g_i - (proj_scalar * g_igd)
            domain_specific_grads[cid] = g_i_ortho

            ortho_norm = torch.norm(g_i_ortho).item()
            print(f"  [{cid}] Domain-specific (Orthogonal) Gradient L2 Norm: {ortho_norm:.4f}")

            ortho_state_dict = unflatten_tensors(g_i_ortho, global_state, fedomg_keys)
            client_out_path = output_paths[i]
            ortho_save_path = client_out_path.replace(".pth", "_ortho.pth")

            torch.save({"state_dict": ortho_state_dict}, ortho_save_path)
            print(f"    -> Saved orthogonal component for {cid} to {ortho_save_path}")

    print("\n--- Phase 4: Updating Global Model ---")
    if USE_PAPER_SIGN_UPDATE:
        new_flat_global = flat_global - (ETA_G * g_igd)
    else:
        new_flat_global = flat_global + (ETA_G * g_igd)

    new_global_weights = unflatten_tensors(new_flat_global, global_state, fedomg_keys)

    for k in fedomg_keys:
        global_ckpt["state_dict"][k] = new_global_weights[k]

    if USE_EXCLUSION and not KEEP_PRIVATE:
        for k in excluded_keys:
            global_ckpt["state_dict"][k] = excluded_accumulators[k]

    print(f"  Saving new FedOMG global model to {prev_global_path}")
    torch.save(global_ckpt, prev_global_path)

    print("\n--- Phase 5: Distributing Global Model ---")
    for ckpt, out_path, cid in zip(client_states, output_paths, client_ids):
        if KEEP_PRIVATE and USE_EXCLUSION:
            for k in fedomg_keys:
                if k in ckpt["state_dict"]:
                    ckpt["state_dict"][k] = global_ckpt["state_dict"][k].clone()
        else:
            for k in global_ckpt["state_dict"].keys():
                if k in ckpt["state_dict"] and (
                    (k in fedomg_keys) or
                    (USE_EXCLUSION and not KEEP_PRIVATE and k in excluded_keys)
                ):
                    ckpt["state_dict"][k] = global_ckpt["state_dict"][k].clone()

        if "optimizer" in ckpt and "state" in ckpt["optimizer"]:
            for param_id in ckpt["optimizer"]["state"]:
                for key in ckpt["optimizer"]["state"][param_id]:
                    if torch.is_tensor(ckpt["optimizer"]["state"][param_id][key]):
                        ckpt["optimizer"]["state"][param_id][key].zero_()

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(ckpt, out_path)
        print(f"  Saved FedOMG synchronized model for {cid} to {out_path}")

    print("=" * 50)
    print("FedOMG Aggregation Complete.\n")

# legacy version, used for round 2&3, but only used ModelD then for the update due to SGD LR overfitting issues. Still better than FedAvg but not as good as the full OMG with the momentum optimization and proper projection math.
# def fedomg_better(models, output_paths, norm_weights, client_ids, prev_global_path="/workspace/work_dirs/fedomg_states/global_model.pth"):
#     EPS = 1e-12

#     KAPPA = 0.5
#     ETA_G = 1.0
#     OMG_INNER_ITERS = 21
#     OMG_LR = 25.0
#     OMG_MOMENTUM = 0.5

#     USE_EXCLUSION = True
#     EXCLUDE_PREFIXES = ['bn', 'running_mean', 'running_var', 'num_batches_tracked', 'pts_bbox_head.task_heads']
#     KEEP_PRIVATE = True

#     print("\n" + "=" * 50)
#     print("Running FedOMG (On-Server Matching Gradient) Aggregation...")
#     print("=" * 50)

#     os.makedirs(os.path.dirname(prev_global_path), exist_ok=True)
#     num_clients = len(models)

#     if not os.path.exists(prev_global_path):
#         print(f"[WARNING] No previous global model found at {prev_global_path}.")
#         print("Initializing FedOMG baseline by running a standard FedAvg for Round 0...")
#         # Assuming you have a standard fedavg() function in scope
#         fedavg(models, [prev_global_path] * num_clients, norm_weights)
#         for in_path, out_path in zip(models, output_paths):
#             os.system(f"cp {prev_global_path} {out_path}")
#         return

#     global_ckpt = torch.load(prev_global_path, map_location="cpu")
#     global_state = global_ckpt["state_dict"]

#     fedomg_keys = []
#     excluded_keys = []

#     for k, v in global_state.items():
#         if torch.is_tensor(v) and v.is_floating_point():
#             if USE_EXCLUSION and any(excl in k for excl in EXCLUDE_PREFIXES):
#                 excluded_keys.append(k)
#             else:
#                 fedomg_keys.append(k)

#     total_params = sum(global_state[k].numel() for k in fedomg_keys)
#     print(f"FedOMG: Tracking {len(fedomg_keys)} floating-point layers ({total_params:,} true weights).")

#     if USE_EXCLUSION:
#         if KEEP_PRIVATE:
#             print(f"Not including {len(excluded_keys)} excluded/private layers.")
#         else:
#             print(f"FedAvg: Standard averaging on {len(excluded_keys)} excluded layers/trackers.")

#     print("\n--- Phase 1: Computing Local Gradients ---")
#     flat_global = flatten_tensors(global_state, fedomg_keys)

#     client_grads = []
#     client_states = []
#     excluded_accumulators = {k: torch.zeros_like(global_state[k]) for k in excluded_keys} if (USE_EXCLUSION and not KEEP_PRIVATE) else None

#     for m_path, cid, w_i in zip(models, client_ids, norm_weights):
#         ckpt_i = torch.load(m_path, map_location="cpu")
#         state_i = ckpt_i["state_dict"]
#         client_states.append(ckpt_i)

#         # local_grad defined as Delta (New - Old) -> Equivalent to -1 * standard gradient
#         flat_client = flatten_tensors(state_i, fedomg_keys)
#         local_grad = flat_client - flat_global
#         client_grads.append(local_grad)

#         grad_norm = torch.norm(local_grad).item()
#         print(f"  [{cid}] Extracted true local gradient. L2 Norm: {grad_norm:.4f}")

#         if USE_EXCLUSION and not KEEP_PRIVATE:
#             for k in excluded_keys:
#                 if k in state_i:
#                     excluded_accumulators[k] += state_i[k] * w_i

#         del flat_client

#     G = torch.stack(client_grads, dim=0)

#     print("\n--- Phase 2: Solving FedOMG On-Server Objective ---")

#     weight_tensor = torch.tensor(norm_weights, dtype=flat_global.dtype, device=flat_global.device)
#     weight_tensor = weight_tensor / (weight_tensor.sum() + EPS)

#     g_fl = (weight_tensor[:, None] * G).sum(dim=0)

#     g_fl_norm = torch.norm(g_fl)
#     print(f"  Reference FedAvg Gradient L2 Norm: {g_fl_norm.item():.4f}")

#     if g_fl_norm.item() < EPS:
#         print("  [WARNING] FedAvg reference gradient is near zero. Falling back to g_FL only.")
#         g_igd = g_fl.clone()
#         gamma_star = weight_tensor.clone()
#         combo = g_fl.clone()
#     else:
#         # Properly track logits with PyTorch Autograd
#         logits = torch.log(weight_tensor + EPS).clone().detach().requires_grad_(True)
        
#         # FIX: Utilize a native PyTorch optimizer for the inner loop to prevent leaf tensor inplace errors
#         optimizer = torch.optim.SGD([logits], lr=OMG_LR, momentum=OMG_MOMENTUM)

#         best_loss = None
#         best_gamma = None
#         best_combo = None

#         for it in range(OMG_INNER_ITERS):
#             optimizer.zero_grad()
            
#             gamma = torch.softmax(logits, dim=0)
#             combo = (gamma[:, None] * G).sum(dim=0)

#             combo_norm = torch.norm(combo) + EPS
#             # Optimization objective: Gamma* = arg min (Gamma g) * g_FL + kappa * ||g_FL|| * ||Gamma g||
#             loss = torch.dot(combo, g_fl) + KAPPA * g_fl_norm * combo_norm

#             loss.backward()
#             optimizer.step()

#             current_loss = loss.item()
#             if best_loss is None or current_loss < best_loss:
#                 best_loss = current_loss
#                 best_gamma = gamma.detach().clone()
#                 best_combo = combo.detach().clone()

#             print(
#                 f"  [OMG {it + 1:02d}/{OMG_INNER_ITERS}] "
#                 f"loss={loss.item():.6f} "
#                 f"||Gamma g||={combo_norm.item():.4f}"
#             )

#         gamma_star = best_gamma
#         combo = best_combo
#         combo_norm = torch.norm(combo)

#         if combo_norm.item() < EPS:
#             print("  [WARNING] Optimized Gamma*g is near zero. Using g_FL only.")
#             g_igd = g_fl.clone()
#         else:
#             # Reconstructing the Invariant Gradient Direction (IGD)
#             g_igd = g_fl + KAPPA * g_fl_norm * (combo / (combo_norm + EPS))

#         print("  Gamma* coefficients:")
#         for cid, gamma_i in zip(client_ids, gamma_star.tolist()):
#             print(f"    {cid}: {gamma_i:.6f}")

#     print("\n--- Phase 3: Aggregating Aligned Gradients ---")
#     combo_norm = torch.norm(combo).item()
#     igd_norm = torch.norm(g_igd).item()

#     print(f"  Optimized Combined Gradient ||Gamma* g||: {combo_norm:.4f}")
#     print(f"  Final Invariant Gradient L2 Norm: {igd_norm:.4f}")

#     print("\n--- Phase 3b: Extracting and Saving Domain-Specific (Orthogonal) Gradients ---")
#     domain_specific_grads = {}
#     igd_norm_sq = torch.dot(g_igd, g_igd) + EPS
    
#     for i, cid in enumerate(client_ids):
#         g_i = client_grads[i]
        
#         # Calculate the scalar for the projection of g_i onto g_igd
#         proj_scalar = torch.dot(g_i, g_igd) / igd_norm_sq
        
#         # Subtract the projected component to get the orthogonal (domain-specific) component
#         g_i_ortho = g_i - (proj_scalar * g_igd)
        
#         # Store it (if you want to use it later)
#         domain_specific_grads[cid] = g_i_ortho
        
#         # Log the magnitude to track how "divergent" this domain is
#         ortho_norm = torch.norm(g_i_ortho).item()
#         print(f"  [{cid}] Domain-specific (Orthogonal) Gradient L2 Norm: {ortho_norm:.4f}")
        
#         # ==========================================
#         # NEW: Unflatten and Save the Orthogonal Component
#         # ==========================================
#         # Map the 1D tensor back to the model's layer shapes
#         ortho_state_dict = unflatten_tensors(g_i_ortho, global_state, fedomg_keys)
        
#         # Construct a save path (e.g., 'merged_A.pth' -> 'merged_A_ortho.pth')
#         client_out_path = output_paths[i]
#         ortho_save_path = client_out_path.replace(".pth", "_ortho.pth")
        
#         # Save as a standard checkpoint dictionary
#         torch.save({"state_dict": ortho_state_dict}, ortho_save_path)
#         print(f"    -> Saved orthogonal component for {cid} to {ortho_save_path}")

#     print("\n--- Phase 4: Updating Global Model ---")
#     # FIX: Because `local_grad` is calculated as (flat_client - flat_global), it represents a positive 
#     # weight step. We must ADD ETA_G * g_igd so the global model learns, avoiding catastrophic unlearning.
#     new_flat_global = flat_global + (ETA_G * g_igd)
#     new_global_weights = unflatten_tensors(new_flat_global, global_state, fedomg_keys)

#     for k in fedomg_keys:
#         global_ckpt["state_dict"][k] = new_global_weights[k]

#     if USE_EXCLUSION and not KEEP_PRIVATE:
#         for k in excluded_keys:
#             global_ckpt["state_dict"][k] = excluded_accumulators[k]

#     print(f"  Saving new FedOMG global model to {prev_global_path}")
#     torch.save(global_ckpt, prev_global_path)

#     print("\n--- Phase 5: Distributing Global Model ---")
#     for ckpt, out_path, cid in zip(client_states, output_paths, client_ids):
#         if KEEP_PRIVATE and USE_EXCLUSION:
#             for k in fedomg_keys:
#                 if k in ckpt["state_dict"]:
#                     ckpt["state_dict"][k] = global_ckpt["state_dict"][k].clone()
#         else:
#             for k in global_ckpt["state_dict"].keys():
#                 if k in ckpt["state_dict"] and (
#                     (k in fedomg_keys) or
#                     (USE_EXCLUSION and not KEEP_PRIVATE and k in excluded_keys)
#                 ):
#                     ckpt["state_dict"][k] = global_ckpt["state_dict"][k].clone()

#         if "optimizer" in ckpt and "state" in ckpt["optimizer"]:
#             for param_id in ckpt["optimizer"]["state"]:
#                 for key in ckpt["optimizer"]["state"][param_id]:
#                     if torch.is_tensor(ckpt["optimizer"]["state"][param_id][key]):
#                         ckpt["optimizer"]["state"][param_id][key].zero_()

#         os.makedirs(os.path.dirname(out_path), exist_ok=True)
#         torch.save(ckpt, out_path)
#         print(f"  Saved FedOMG synchronized model for {cid} to {out_path}")

#     print("=" * 50)
#     print("FedOMG Aggregation Complete.\n")

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

def fedselect(models, output_paths, norm_weights, client_ids, prev_global_path="/workspace/work_dirs/fedselect_states/global_model.pth", mask_dir="/workspace/work_dirs/fedselect_masks", select_ratio=0.05, max_sparsity=0.5):
    """
    FedSelect implementation (CVPR 2024). 
    Automatically discovers and freezes personalized subnetworks for each client 
    based on the magnitude of weight changes, then aggregates only the shared parameters.
    """
    print(f"Running FedSelect Aggregation (select_ratio={select_ratio}, max_sparsity={max_sparsity})...")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(os.path.dirname(prev_global_path), exist_ok=True)

    # 1. Load Pre-Training Global Weights
    if not os.path.exists(prev_global_path):
        print(f"No previous global model found at {prev_global_path}. Exiting FedSelect since we need a baseline for difference calculation. Please run one round of standard FedAvg first to initialize the global model.")
        exit(1)
        
    prev_global_ckpt = torch.load(prev_global_path, map_location='cpu')
    prev_state = prev_global_ckpt['state_dict']
    
    # Identify valid floating-point keys
    valid_keys = [k for k, v in prev_state.items() if v.is_floating_point() and 'num_batches_tracked' not in k]
    total_params = sum(prev_state[k].numel() for k in valid_keys)
    print(f"Total valid parameters for FedSelect: {total_params:,}")

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
    # ### NEW: Scan for existing round directories and increment ###
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
    print(f"Saving historical masks for round {current_round} to {round_mask_dir}")
    # ##############################################################

    # ---------------------------------------------------------
    # Phase 1: Client Subnetwork Discovery (Mask Updating)
    # ---------------------------------------------------------
    print("Phase 1: Expanding personalized client subnetworks...")
    for m_path, cid in zip(models, client_ids):
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        
        # Load or initialize client mask (0 = Global, 1 = Personalized)
        if os.path.exists(mask_path):
            client_mask = torch.load(mask_path)
        else:
            client_mask = {k: torch.zeros_like(prev_state[k], dtype=torch.bool) for k in valid_keys}

        ckpt_i = torch.load(m_path, map_location='cpu')
        state_i = ckpt_i['state_dict']
        
        # Calculate parameter changes and collect valid ones for thresholding
        all_diffs = []
        current_personalized = 0
        
        for k in valid_keys:
            if k in state_i:
                diff = torch.abs(state_i[k] - prev_state[k])
                # Only evaluate parameters that are currently shared (mask == 0)
                global_mask = ~client_mask[k]
                all_diffs.append(diff[global_mask].flatten())
                current_personalized += client_mask[k].sum().item()
                
        # Determine how many new parameters to select this round
        cat_diffs = torch.cat(all_diffs)
        k_to_select = int(total_params * select_ratio)
        max_allowed = int(total_params * max_sparsity)
        k_to_select = min(k_to_select, max_allowed - current_personalized)
        
        if k_to_select > 0 and len(cat_diffs) > 0:
            k_to_select = min(k_to_select, len(cat_diffs))
            
            # --- NEW LOGIC: Calculate averages ---
            avg_overall_diff = cat_diffs.mean().item()
            top_values = torch.topk(cat_diffs, k_to_select).values
            threshold = top_values[-1]
            avg_selected_diff = top_values.mean().item()
            
            print(f"  {cid}: Avg diff of all shared weights: {avg_overall_diff:.6f} | Avg diff of selected {select_ratio*100}%: {avg_selected_diff:.6f}")
            # -------------------------------------
            
            # Update the mask permanently
            new_personalized = 0
            for k in valid_keys:
                if k in state_i:
                    diff = torch.abs(state_i[k] - prev_state[k])
                    # Flip 0 to 1 if it exceeds threshold and is currently 0
                    new_ones = (~client_mask[k]) & (diff >= threshold)
                    client_mask[k][new_ones] = True
                    new_personalized += new_ones.sum().item()
                    
            print(f"  {cid}: Personalized {new_personalized:,} new params. Total sparsity: {(current_personalized + new_personalized) / total_params * 100:.2f}%")
        else:
            print(f"  {cid}: Reached max sparsity or no params to select. Total sparsity: {current_personalized / total_params * 100:.2f}%")
            
        torch.save(client_mask, mask_path)

        # ### NEW: Save historical visualization mask (0=Global, 1=Personal) ###
        vis_mask = {k: v.to(torch.int8) for k, v in client_mask.items()}
        hist_mask_path = os.path.join(round_mask_dir, f"{cid}_mask.pt")
        torch.save(vis_mask, hist_mask_path)
        # ######################################################################

        del ckpt_i
        
    # ---------------------------------------------------------
    # Phase 2: Masked Server Aggregation
    # ---------------------------------------------------------
    print("Phase 2: Aggregating shared parameters on server...")
    running_sum = {k: torch.zeros_like(v) for k, v in prev_state.items() if k in valid_keys}
    presence_weights = {k: torch.zeros_like(v) for k, v in prev_state.items() if k in valid_keys}
    
    for m_path, cid, w_i in zip(models, client_ids, norm_weights):
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        client_mask = torch.load(mask_path)
        
        ckpt_i = torch.load(m_path, map_location='cpu')
        state_i = ckpt_i['state_dict']
        
        for k in valid_keys:
            if k in state_i:
                # Calculate aggregation weight per parameter: w_i * (1 - mask)
                active_mask = (~client_mask[k]).float()
                running_sum[k] += state_i[k] * active_mask * w_i
                presence_weights[k] += active_mask * w_i
                
        del ckpt_i
        
    # Finalize averaged weights (handling division by zero for fully personalized params)
    averaged_weights = {}
    for k in valid_keys:
        valid_mask = presence_weights[k] > 0
        averaged_weights[k] = torch.where(
            valid_mask,
            running_sum[k] / presence_weights[k].clamp(min=1e-9),
            prev_state[k] # Fallback to previous global weight if all clients personalized it
        )
        
    # Save the new global model for the *next* round's difference calculation
    for k in valid_keys:
        prev_global_ckpt['state_dict'][k] = averaged_weights[k]
    torch.save(prev_global_ckpt, prev_global_path)
    del prev_global_ckpt

    # ---------------------------------------------------------
    # Phase 3: Client Subnetwork Injection & Saving
    # ---------------------------------------------------------
    print("Phase 3: Injecting shared weights into client models...")
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
                
        # --- OPTIMIZER PURGING REMOVED FROM HERE ---
                        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(ckpt, out_path)
        print(f"Saved personalized FedSelect model to {out_path}")


def fedselect_elastic(models, output_paths, norm_weights, client_ids, prev_global_path="/workspace/work_dirs/fedselect_states/global_model.pth", mask_dir="/workspace/work_dirs/fedselect_masks", select_ratio=0.05, max_sparsity=0.5):
    """
    FedSelect implementation with Relative Scaling & Mask Reintroduction.
    """
    print(f"Running FedSelect Aggregation (select_ratio={select_ratio}, max_sparsity={max_sparsity})...")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(os.path.dirname(prev_global_path), exist_ok=True)

    if not os.path.exists(prev_global_path):
        print(f"No previous global model found at {prev_global_path}. Exiting. Run one round of standard FedAvg first.")
        exit(1)
        
    prev_global_ckpt = torch.load(prev_global_path, map_location='cpu')
    prev_state = prev_global_ckpt['state_dict']
    
    valid_keys = [k for k, v in prev_state.items() if v.is_floating_point() and 'num_batches_tracked' not in k]
    total_params = sum(prev_state[k].numel() for k in valid_keys)
    print(f"Total valid parameters for FedSelect: {total_params:,}")

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
    # ### NEW: Scan for existing round directories and increment ###
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
    print(f"Saving historical masks for round {current_round} to {round_mask_dir}")
    # ##############################################################

    # ---------------------------------------------------------
    # Phase 1: Client Subnetwork Discovery & Reintroduction
    # ---------------------------------------------------------
    print("Phase 1: Updating personalized client subnetworks...")
    for m_path, cid in zip(models, client_ids):
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        
        if os.path.exists(mask_path):
            client_mask = torch.load(mask_path)
        else:
            client_mask = {k: torch.zeros_like(prev_state[k], dtype=torch.bool) for k in valid_keys}

        ckpt_i = torch.load(m_path, map_location='cpu')
        state_i = ckpt_i['state_dict']
        
        all_shared_rel_diffs = []
        current_personalized = 0
        
        for k in valid_keys:
            if k in state_i:
                # --- FIX 1: Relative Difference ---
                # Add 1e-8 to denominator to prevent division by zero
                rel_diff = torch.abs(state_i[k] - prev_state[k]) / (torch.abs(prev_state[k]) + 1e-8)
                
                global_mask = ~client_mask[k]
                all_shared_rel_diffs.append(rel_diff[global_mask].flatten())
                current_personalized += client_mask[k].sum().item()
                
        cat_diffs = torch.cat(all_shared_rel_diffs)
        
        # Calculate standard mathematical slices (unaffected by sparsity caps)
        base_k = int(total_params * select_ratio)
        fetch_k = min(2 * base_k, len(cat_diffs)) 
        
        if fetch_k > 0:
            top_values = torch.topk(cat_diffs, fetch_k).values
            
            # --- FIX 2: Decouple Thresholds from Sparsity Cap ---
            
            # 1. Reintroduction Threshold (Always calculated if possible)
            next_slice = top_values[base_k:fetch_k]
            reintro_threshold = next_slice.mean().item() if len(next_slice) > 0 else -1.0
            
            # 2. Personalization Threshold (Subject to max_sparsity cap)
            remaining_budget = max(0, int(total_params * max_sparsity) - current_personalized)
            actual_k_to_select = min(base_k, remaining_budget)
            
            if actual_k_to_select > 0:
                personalize_threshold = top_values[actual_k_to_select - 1].item()
            else:
                personalize_threshold = float('inf') # Cap reached, nothing new can exceed infinity
                
            avg_overall_rel_diff = cat_diffs.mean().item()
            print(f"  {cid}: Shared Rel Diff Avg: {avg_overall_rel_diff:.6f} | Reintro Threshold: {reintro_threshold:.6f}")
            if actual_k_to_select == 0:
                print(f"  {cid}: Reached max sparsity. Only evaluating reintroductions.")
            
            # Apply changes
            new_personalized = 0
            num_reintroduced = 0
            
            for k in valid_keys:
                if k in state_i:
                    rel_diff = torch.abs(state_i[k] - prev_state[k]) / (torch.abs(prev_state[k]) + 1e-8)
                    
                    new_ones = (~client_mask[k]) & (rel_diff >= personalize_threshold)
                    
                    if reintro_threshold > 0:
                        # Notice we evaluate personalized weights (client_mask[k] == True) for reintroduction
                        reintro_zeros = client_mask[k] & (rel_diff < reintro_threshold)
                    else:
                        reintro_zeros = torch.zeros_like(client_mask[k], dtype=torch.bool)
                        
                    client_mask[k][new_ones] = True
                    client_mask[k][reintro_zeros] = False
                    
                    new_personalized += new_ones.sum().item()
                    num_reintroduced += reintro_zeros.sum().item()
                    
            new_total = current_personalized + new_personalized - num_reintroduced
            print(f"  {cid}: Personalized +{new_personalized:,} | Reintroduced -{num_reintroduced:,} | Total sparsity: {new_total / total_params * 100:.2f}%")
        else:
            print(f"  {cid}: No shared params left to evaluate. Total sparsity: {current_personalized / total_params * 100:.2f}%")
            
        torch.save(client_mask, mask_path)

        # ### NEW: Save historical visualization mask (0=Global, 1=Personal) ###
        vis_mask = {k: v.to(torch.int8) for k, v in client_mask.items()}
        hist_mask_path = os.path.join(round_mask_dir, f"{cid}_mask.pt")
        torch.save(vis_mask, hist_mask_path)
        # ######################################################################


        del ckpt_i
        
    # ---------------------------------------------------------
    # Phase 2: Masked Server Aggregation
    # ---------------------------------------------------------
    print("Phase 2: Aggregating shared parameters on server...")
    running_sum = {k: torch.zeros_like(v) for k, v in prev_state.items() if k in valid_keys}
    presence_weights = {k: torch.zeros_like(v) for k, v in prev_state.items() if k in valid_keys}
    
    for m_path, cid, w_i in zip(models, client_ids, norm_weights):
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        client_mask = torch.load(mask_path)
        
        ckpt_i = torch.load(m_path, map_location='cpu')
        state_i = ckpt_i['state_dict']
        
        for k in valid_keys:
            if k in state_i:
                active_mask = (~client_mask[k]).float()
                running_sum[k] += state_i[k] * active_mask * w_i
                presence_weights[k] += active_mask * w_i
                
        del ckpt_i
        
    averaged_weights = {}
    for k in valid_keys:
        valid_mask = presence_weights[k] > 0
        averaged_weights[k] = torch.where(
            valid_mask,
            running_sum[k] / presence_weights[k].clamp(min=1e-9),
            prev_state[k] 
        )
        
    for k in valid_keys:
        prev_global_ckpt['state_dict'][k] = averaged_weights[k]
    torch.save(prev_global_ckpt, prev_global_path)
    del prev_global_ckpt

    # ---------------------------------------------------------
    # Phase 3: Client Subnetwork Injection & Saving
    # ---------------------------------------------------------
    print("Phase 3: Injecting shared weights into client models...")
    for in_path, out_path, cid in zip(models, output_paths, client_ids):
        ckpt = torch.load(in_path, map_location='cpu')
        state = ckpt['state_dict']
        
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        client_mask = torch.load(mask_path)
        
        for k in valid_keys:
            if k in state:
                m = client_mask[k].float()
                state[k] = (m * state[k]) + ((1.0 - m) * averaged_weights[k])
                
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(ckpt, out_path)
        print(f"Saved personalized FedSelect model to {out_path}")


def fedselect_fullelastic(models, output_paths, norm_weights, client_ids, prev_global_path="/workspace/work_dirs/fedselect_states/global_model.pth", mask_dir="/workspace/work_dirs/fedselect_masks", select_ratio=0.05):
    """
    FedSelect Full Elastic implementation.
    Evaluates all parameters from scratch every round to find the top `select_ratio` 
    most different parameters. Keeps no mask history across rounds.
    """
    print(f"Running FedSelect Full-Elastic Aggregation (select_ratio={select_ratio})...")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(os.path.dirname(prev_global_path), exist_ok=True)

    if not os.path.exists(prev_global_path):
        print(f"No previous global model found at {prev_global_path}. Exiting. Run one round of standard FedAvg first.")
        exit(1)
        
    prev_global_ckpt = torch.load(prev_global_path, map_location='cpu')
    prev_state = prev_global_ckpt['state_dict']
    
    valid_keys = [k for k, v in prev_state.items() if v.is_floating_point() and 'num_batches_tracked' not in k]
    total_params = sum(prev_state[k].numel() for k in valid_keys)
    print(f"Total valid parameters for FedSelect: {total_params:,}")

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
    # Auto-Detect Current Round Directory for History
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
    print(f"Saving historical masks for round {current_round} to {round_mask_dir}")

    # ---------------------------------------------------------
    # Phase 1: Fresh Client Subnetwork Discovery 
    # ---------------------------------------------------------
    print("Phase 1: Discovering personalized client subnetworks from scratch...")
    for m_path, cid in zip(models, client_ids):
        # Fresh mask initialized to False (0)
        client_mask = {k: torch.zeros_like(prev_state[k], dtype=torch.bool) for k in valid_keys}

        ckpt_i = torch.load(m_path, map_location='cpu')
        state_i = ckpt_i['state_dict']
        
        all_rel_diffs = []
        
        # Calculate relative differences for ALL valid parameters
        for k in valid_keys:
            if k in state_i:
                rel_diff = torch.abs(state_i[k] - prev_state[k]) / (torch.abs(prev_state[k]) + 1e-8)
                all_rel_diffs.append(rel_diff.flatten())
                
        cat_diffs = torch.cat(all_rel_diffs)
        k_to_select = int(total_params * select_ratio)
        
        if k_to_select > 0:
            # Find the threshold for the top 'select_ratio' parameters
            top_values = torch.topk(cat_diffs, k_to_select).values
            personalize_threshold = top_values[-1].item()
            
            avg_overall_rel_diff = cat_diffs.mean().item()
            print(f"  {cid}: Shared Rel Diff Avg: {avg_overall_rel_diff:.6f} | Threshold: {personalize_threshold:.6f}")
            
            new_personalized = 0
            
            # Apply the mask based purely on the new threshold
            for k in valid_keys:
                if k in state_i:
                    rel_diff = torch.abs(state_i[k] - prev_state[k]) / (torch.abs(prev_state[k]) + 1e-8)
                    new_ones = rel_diff >= personalize_threshold
                    client_mask[k] = new_ones
                    new_personalized += new_ones.sum().item()
                    
            print(f"  {cid}: Personalized {new_personalized:,} params | Total sparsity: {new_personalized / total_params * 100:.2f}%")
        else:
            print(f"  {cid}: Select ratio is 0. Total sparsity: 0.00%")
            
        # Overwrite/save the main mask for Phase 2/3
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        torch.save(client_mask, mask_path)

        # Save historical visualization mask (0=Global, 1=Personal)
        vis_mask = {k: v.to(torch.int8) for k, v in client_mask.items()}
        hist_mask_path = os.path.join(round_mask_dir, f"{cid}_mask.pt")
        torch.save(vis_mask, hist_mask_path)

        del ckpt_i
        
    # ---------------------------------------------------------
    # Phase 2: Masked Server Aggregation
    # ---------------------------------------------------------
    print("Phase 2: Aggregating shared parameters on server...")
    running_sum = {k: torch.zeros_like(v) for k, v in prev_state.items() if k in valid_keys}
    presence_weights = {k: torch.zeros_like(v) for k, v in prev_state.items() if k in valid_keys}
    
    for m_path, cid, w_i in zip(models, client_ids, norm_weights):
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        client_mask = torch.load(mask_path)
        
        ckpt_i = torch.load(m_path, map_location='cpu')
        state_i = ckpt_i['state_dict']
        
        for k in valid_keys:
            if k in state_i:
                active_mask = (~client_mask[k]).float()
                running_sum[k] += state_i[k] * active_mask * w_i
                presence_weights[k] += active_mask * w_i
                
        del ckpt_i
        
    averaged_weights = {}
    for k in valid_keys:
        valid_mask = presence_weights[k] > 0
        averaged_weights[k] = torch.where(
            valid_mask,
            running_sum[k] / presence_weights[k].clamp(min=1e-9),
            prev_state[k] 
        )
        
    for k in valid_keys:
        prev_global_ckpt['state_dict'][k] = averaged_weights[k]
    torch.save(prev_global_ckpt, prev_global_path)
    del prev_global_ckpt

    # ---------------------------------------------------------
    # Phase 3: Client Subnetwork Injection & Saving
    # ---------------------------------------------------------
    print("Phase 3: Injecting shared weights into client models...")
    for in_path, out_path, cid in zip(models, output_paths, client_ids):
        ckpt = torch.load(in_path, map_location='cpu')
        state = ckpt['state_dict']
        
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        client_mask = torch.load(mask_path)
        
        for k in valid_keys:
            if k in state:
                m = client_mask[k].float()
                # Personal weights remain (m * state), Global weights injected ((1-m) * averaged)
                state[k] = (m * state[k]) + ((1.0 - m) * averaged_weights[k])
                
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(ckpt, out_path)
        print(f"Saved personalized FedSelect model to {out_path}")



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

    elif args.method == 'fedselect':
        client_ids = [f"Model{string.ascii_uppercase[i]}" for i in range(len(model_paths))]
        fedselect(
            models=model_paths, 
            output_paths=output_paths, 
            norm_weights=norm_weights, 
            client_ids=client_ids,
            prev_global_path="/workspace/work_dirs/fedselect_states/global_model.pth", 
            mask_dir="/workspace/work_dirs/fedselect_masks",
            select_ratio=args.select_ratio,
            max_sparsity=args.max_sparsity
        )
    elif args.method == 'fedselect_elastic':
        client_ids = [f"Model{string.ascii_uppercase[i]}" for i in range(len(model_paths))]
        fedselect_elastic(
            models=model_paths, 
            output_paths=output_paths, 
            norm_weights=norm_weights, 
            client_ids=client_ids,
            prev_global_path="/workspace/work_dirs/fedselect_states/global_model.pth", 
            mask_dir="/workspace/work_dirs/fedselect_masks",
            select_ratio=args.select_ratio,
            max_sparsity=args.max_sparsity
        )
    elif args.method == 'fedselect_fullelastic':
        client_ids = [f"Model{string.ascii_uppercase[i]}" for i in range(len(model_paths))]
        fedselect_fullelastic(
            models=model_paths, 
            output_paths=output_paths, 
            norm_weights=norm_weights, 
            client_ids=client_ids,
            prev_global_path="/workspace/work_dirs/fedselect_states/global_model.pth", 
            mask_dir="/workspace/work_dirs/fedselect_masks",
            select_ratio=args.select_ratio
        )
    elif args.method == 'fedomg':
            client_ids = [f"Model{string.ascii_uppercase[i]}" for i in range(len(model_paths))]
            fedomg(
                models=model_paths, 
                output_paths=output_paths, 
                norm_weights=norm_weights, 
                client_ids=client_ids,
                prev_global_path="/workspace/work_dirs/fedomg_states/global_model.pth"
            )
    elif args.method == 'fedomg_better':
            client_ids = [f"Model{string.ascii_uppercase[i]}" for i in range(len(model_paths))]
            fedomg_better(
                models=model_paths, 
                output_paths=output_paths, 
                norm_weights=norm_weights, 
                client_ids=client_ids,
                prev_global_path="/workspace/work_dirs/fedomg_states/global_model.pth"
            )
    elif args.method == 'fedomg_better_better':
            client_ids = [f"Model{string.ascii_uppercase[i]}" for i in range(len(model_paths))]
            fedomg_better_better(
                models=model_paths, 
                output_paths=output_paths, 
                norm_weights=norm_weights, 
                client_ids=client_ids,
                prev_global_path="/workspace/work_dirs/fedomg_states/global_model.pth"
            )
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
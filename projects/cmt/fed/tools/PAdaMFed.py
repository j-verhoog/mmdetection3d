import argparse
import os
import math
import time
import copy
from typing import Dict, List, Optional

import torch
from modular_merging import fedavg


def PAdaMFed_VR(
    models,
    output_paths,
    norm_weights,
    client_ids,
    prev_global_path="/workspace/work_dirs/PAdaMFed_VR/global_model.pth",
    server_state_path="/workspace/work_dirs/PAdaMFed_VR/server_state.pt",
    client_control_paths=None,
    round_idx=None,
    total_rounds=None,
    local_steps=None,
    local_steps_map=None,
    total_num_clients=None,
    num_sampled_clients=None,
    eta=None,
    gamma=None,
    beta=None,
    exclude_prefixes=None,
    keep_excluded_local=True,
    zero_optimizer_state=True,
    strict_key_check=True,
    eps=1e-12,
):
    # if exclude_prefixes is None:
    #     exclude_prefixes = []

    if not exclude_prefixes:
        exclude_prefixes = ['bn', 'running_mean', 'running_var', 'num_batches_tracked', 'pts_bbox_head.task_heads']

    _validate_inputs(models, output_paths, norm_weights, client_ids)

    num_clients = len(models)

    if total_num_clients is None:
        total_num_clients = len(client_ids)

    if num_sampled_clients is None:
        num_sampled_clients = len(client_ids)

    resolved_local_steps = _resolve_local_steps(
        client_ids=client_ids,
        local_steps=local_steps,
        local_steps_map=local_steps_map,
    )

    if total_rounds is not None and total_rounds <= 0:
        raise ValueError("total_rounds must be positive")

    print("\n" + "=" * 80)
    print("Running PAdaMFed-VR Server Aggregation")
    print("=" * 80)
    print(f"Incoming models          : {num_clients}")
    print(f"Client IDs               : {client_ids}")
    print(f"Previous global path     : {prev_global_path}")
    print(f"Server state path        : {server_state_path}")
    print(f"Client control paths     : {client_control_paths}")
    print(f"Total clients N          : {total_num_clients}")
    print(f"Sampled clients S        : {num_sampled_clients}")
    print(f"Local steps per client   : {resolved_local_steps}")
    print(f"Round index t            : {round_idx}")
    print(f"Total rounds T           : {total_rounds}")
    print(f"Exclude prefixes         : {exclude_prefixes}")
    print(f"Keep excluded local      : {keep_excluded_local}")
    print(f"Zero optimizer state     : {zero_optimizer_state}")
    print(f"Strict key check         : {strict_key_check}")

    if not os.path.exists(prev_global_path):
        print(f"\n[WARNING] No previous global model found at {prev_global_path}.")
        print("Initializing Round 0 baseline by running standard FedAvg...")

        fedavg(models, [prev_global_path] * num_clients, norm_weights)

        global_ckpt = torch.load(prev_global_path, map_location="cpu")
        global_state = global_ckpt["state_dict"]

        for in_path, out_path, cid in zip(models, output_paths, client_ids):
            client_ckpt = torch.load(in_path, map_location="cpu")
            for k, v in global_state.items():
                if k in client_ckpt["state_dict"]:
                    client_ckpt["state_dict"][k] = v.clone()
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(client_ckpt, out_path)
            print(f"  Saved initialized checkpoint for {cid} to {out_path}")

        init_server_state = {
            "method": "PAdaMFed-VR",
            "round": 0,
            "created_at": time.time(),
            "client_ids_all": list(client_ids),
            "N": int(total_num_clients),
            "last_sampled_client_ids": list(client_ids),
            "S": int(num_sampled_clients),
            "local_steps_map": {cid: int(resolved_local_steps[cid]) for cid in client_ids},
            "exclude_prefixes": list(exclude_prefixes),
            "global_control_variate": None,
            "global_momentum": None,
            "client_control_variates": {},
            "last_broadcast_direction": None,
        }
        os.makedirs(os.path.dirname(server_state_path), exist_ok=True)
        torch.save(init_server_state, server_state_path)
        print("Round 0 initialization complete.")
        print("Control variate / momentum will be bootstrapped automatically on the first merge with client control files.")
        print("=" * 80)
        return

    if client_control_paths is None:
        raise ValueError("client_control_paths must be provided for exact PAdaMFed-VR server update")

    if len(client_control_paths) != len(models):
        raise ValueError("client_control_paths length must match models length")

    global_ckpt = torch.load(prev_global_path, map_location="cpu")
    if "state_dict" not in global_ckpt:
        raise KeyError(f"Checkpoint at {prev_global_path} does not contain 'state_dict'")
    global_state = global_ckpt["state_dict"]

    client_ckpts = []
    client_states = []
    for cid, path in zip(client_ids, models):
        ckpt = torch.load(path, map_location="cpu")
        if "state_dict" not in ckpt:
            raise KeyError(f"Checkpoint for client {cid} at {path} does not contain 'state_dict'")
        client_ckpts.append(ckpt)
        client_states.append(ckpt["state_dict"])
        print(f"  Loaded client model for {cid}: {path}")

    if strict_key_check:
        _validate_state_dict_compatibility(global_state, client_states, client_ids)

    mergeable_keys, excluded_keys = _split_keys(global_state, exclude_prefixes)

    print("\n--- Key Partition ---")
    print(f"  Mergeable keys : {len(mergeable_keys)}")
    print(f"  Excluded keys  : {len(excluded_keys)}")

    if os.path.exists(server_state_path):
        server_state = torch.load(server_state_path, map_location="cpu")
        print(f"  Loaded existing server state from {server_state_path}")
    else:
        raise FileNotFoundError(
            f"Expected server_state_path to exist for PAdaMFed-VR round update: {server_state_path}"
        )

    prev_client_controls = server_state.get("client_control_variates", {})
    prev_global_control = server_state.get("global_control_variate", None)
    prev_global_momentum = server_state.get("global_momentum", None)

    current_client_controls = {}
    for cid, control_path in zip(client_ids, client_control_paths):
        control_obj = torch.load(control_path, map_location="cpu")

        if isinstance(control_obj, dict) and "state_dict" in control_obj and isinstance(control_obj["state_dict"], dict):
            control_state = control_obj["state_dict"]
        elif isinstance(control_obj, dict):
            control_state = control_obj
        else:
            raise TypeError(f"Control variate file for client {cid} is not a dict-like object: {control_path}")

        _validate_control_keys(control_state, mergeable_keys, cid)
        current_client_controls[cid] = control_state
        print(f"  Loaded control variate for {cid}: {control_path}")

    bootstrap_control_state = False

    if prev_global_control is None or prev_global_momentum is None:
        bootstrap_control_state = True
        print("\n[WARNING] Server control state is missing.")
        print("Bootstrapping global control variate and momentum from current client control variates.")
    else:
        missing_prev = [cid for cid in client_ids if cid not in prev_client_controls]
        if missing_prev:
            print(f"\n[WARNING] Missing previous control variates for sampled clients: {missing_prev}")
            print("Initializing missing previous client control variates to zeros for this merge.")
            for cid in missing_prev:
                prev_client_controls[cid] = _zero_like_control_state(
                    current_client_controls[cid],
                    mergeable_keys,
                )

    if eta is None or gamma is None or beta is None:
        if total_rounds is None:
            raise ValueError(
                "To auto-compute eta/gamma/beta, provide total_rounds"
            )

        S = float(num_sampled_clients)
        T = float(total_rounds)
        mean_K = sum(float(resolved_local_steps[cid]) for cid in client_ids) / float(len(client_ids))

        if eta is None:
            eta = 1.0 / (mean_K * T)
        if gamma is None:
            gamma = (S * mean_K) ** (1.0 / 3.0) / (T ** (2.0 / 3.0))
        if beta is None:
            beta = (S * mean_K) ** (1.0 / 3.0) / (T ** (2.0 / 3.0))

        print("[INFO] Using unequal-step variant: eta/gamma/beta auto-computed from mean local K across sampled clients.")

    print("\n--- PAdaMFed-VR Scalars ---")
    print(f"  eta   = {eta}")
    print(f"  gamma = {gamma}")
    print(f"  beta  = {beta}")

    print("\n--- Phase 1: Aggregate local model-update direction g_model^t (unequal-step variant) ---")
    g_model = {}
    per_client_step_factors = {}
    for cid in client_ids:
        Ki = float(resolved_local_steps[cid])
        if Ki <= 0:
            raise ValueError(f"Client {cid} has non-positive local steps: {Ki}")
        per_client_step_factors[cid] = 1.0 / (float(eta) * Ki)

    for k in mergeable_keys:
        ref = global_state[k]
        if not torch.is_tensor(ref):
            continue
        acc = torch.zeros_like(ref)
        for cid, client_state in zip(client_ids, client_states):
            acc.add_(global_state[k] - client_state[k], alpha=per_client_step_factors[cid])
        acc.div_(float(num_sampled_clients))
        g_model[k] = acc

    print(f"  Aggregated model direction for {len(g_model)} mergeable tensor keys")
    print(f"  Per-client 1/(eta*K_i): { {cid: per_client_step_factors[cid] for cid in client_ids} }")

    print("\n--- Phase 2: Update global model theta^{t+1} = theta^t - gamma * g_model^t ---")
    new_global_state = _clone_state_dict(global_state)
    for k in mergeable_keys:
        if k in g_model:
            new_global_state[k] = global_state[k] - float(gamma) * g_model[k]

    print("  Global model updated")

    print("\n--- Phase 3: Aggregate control variate c^t ---")

    if bootstrap_control_state:
        new_global_control, new_global_momentum = _bootstrap_global_control_and_momentum(
            current_client_controls=current_client_controls,
            mergeable_keys=mergeable_keys,
            total_num_clients=total_num_clients,
        )
        print("  Bootstrapped global control variate from current client controls")
        print("  Bootstrapped global momentum = global control variate")
    else:
        delta_c_avg_over_N = {}
        for k in mergeable_keys:
            ref = prev_global_control[k]
            if not torch.is_tensor(ref):
                continue
            acc = torch.zeros_like(ref)
            for cid in client_ids:
                curr_ci = current_client_controls[cid][k]
                prev_ci = prev_client_controls[cid][k]
                acc.add_(curr_ci - prev_ci)
            acc.div_(float(total_num_clients))
            delta_c_avg_over_N[k] = acc

        new_global_control = {}
        for k in mergeable_keys:
            if torch.is_tensor(prev_global_control[k]):
                new_global_control[k] = prev_global_control[k] + delta_c_avg_over_N[k]
            else:
                new_global_control[k] = copy.deepcopy(prev_global_control[k])

        print("  Global control variate updated")

        print("\n--- Phase 4: Aggregate momentum g^t ---")
        avg_delta_c_over_S = {}
        for k in mergeable_keys:
            ref = prev_global_control[k]
            if not torch.is_tensor(ref):
                continue
            acc = torch.zeros_like(ref)
            for cid in client_ids:
                curr_ci = current_client_controls[cid][k]
                prev_ci = prev_client_controls[cid][k]
                acc.add_(curr_ci - prev_ci)
            acc.div_(float(num_sampled_clients))
            avg_delta_c_over_S[k] = acc

        new_global_momentum = {}
        for k in mergeable_keys:
            if torch.is_tensor(prev_global_control[k]):
                new_global_momentum[k] = (
                    float(beta) * (avg_delta_c_over_S[k] + prev_global_control[k])
                    + (1.0 - float(beta)) * prev_global_momentum[k]
                )
            else:
                new_global_momentum[k] = copy.deepcopy(prev_global_momentum[k])

        print("  Global momentum updated")

    print("\n--- Phase 5: Build broadcast direction beta*c^t + (1-beta)*g^t ---")
    broadcast_direction = {}
    for k in mergeable_keys:
        if torch.is_tensor(new_global_control[k]):
            broadcast_direction[k] = (
                float(beta) * new_global_control[k]
                + (1.0 - float(beta)) * new_global_momentum[k]
            )

    print("  Broadcast direction built")

    print("\n--- Phase 6: Save updated global checkpoint ---")
    updated_global_ckpt = copy.deepcopy(global_ckpt)
    updated_global_ckpt["state_dict"] = new_global_state
    updated_global_ckpt["padamfed_vr_round"] = round_idx
    updated_global_ckpt["padamfed_vr_eta"] = float(eta)
    updated_global_ckpt["padamfed_vr_gamma"] = float(gamma)
    updated_global_ckpt["padamfed_vr_beta"] = float(beta)
    updated_global_ckpt["padamfed_vr_timestamp"] = time.time()
    updated_global_ckpt["padamfed_vr_local_steps_map"] = {cid: int(resolved_local_steps[cid]) for cid in client_ids}

    os.makedirs(os.path.dirname(prev_global_path), exist_ok=True)
    torch.save(updated_global_ckpt, prev_global_path)
    print(f"  Saved global checkpoint to {prev_global_path}")

    print("\n--- Phase 7: Redistribute next client checkpoints ---")
    for in_path, out_path, cid in zip(models, output_paths, client_ids):
        ckpt = torch.load(in_path, map_location="cpu")
        state = ckpt["state_dict"]

        for k in mergeable_keys:
            if k in new_global_state and k in state:
                state[k] = new_global_state[k].clone()

        if not keep_excluded_local:
            for k in excluded_keys:
                if k in new_global_state and k in state:
                    state[k] = new_global_state[k].clone()

        ckpt["state_dict"] = state

        ckpt["padamfed_vr_server_payload"] = {
            "theta_next_path": prev_global_path,
            "broadcast_direction_included": True,
            "round_idx": round_idx,
            "local_steps_this_round": int(resolved_local_steps[cid]),
        }

        ckpt["padamfed_vr_broadcast_direction"] = _clone_state_dict(broadcast_direction)

        if zero_optimizer_state and "optimizer" in ckpt and "state" in ckpt["optimizer"]:
            for param_id in ckpt["optimizer"]["state"]:
                for key in ckpt["optimizer"]["state"][param_id]:
                    if torch.is_tensor(ckpt["optimizer"]["state"][param_id][key]):
                        ckpt["optimizer"]["state"][param_id][key].zero_()

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(ckpt, out_path)
        print(f"  Saved redistributed model for {cid} to {out_path}")

    print("\n--- Phase 8: Persist server state ---")
    updated_server_state = {
        "method": "PAdaMFed-VR",
        "round": 0 if round_idx is None else int(round_idx) + 1,
        "updated_at": time.time(),
        "client_ids_all": list(server_state.get("client_ids_all", client_ids)),
        "N": int(total_num_clients),
        "last_sampled_client_ids": list(client_ids),
        "S": int(num_sampled_clients),
        "local_steps_map": {cid: int(resolved_local_steps[cid]) for cid in client_ids},
        "exclude_prefixes": list(exclude_prefixes),
        "global_control_variate": _clone_state_dict(new_global_control),
        "global_momentum": _clone_state_dict(new_global_momentum),
        "client_control_variates": {
            **{k: v for k, v in prev_client_controls.items() if k not in client_ids},
            **{cid: _clone_state_dict(current_client_controls[cid]) for cid in client_ids},
        },
        "last_broadcast_direction": _clone_state_dict(broadcast_direction),
        "last_eta": float(eta),
        "last_gamma": float(gamma),
        "last_beta": float(beta),
    }

    os.makedirs(os.path.dirname(server_state_path), exist_ok=True)
    torch.save(updated_server_state, server_state_path)
    print(f"  Saved server state to {server_state_path}")

    print("\nDone.")
    print("=" * 80)


def initialize_padamfed_vr_server_state(
    initial_global_path,
    init_client_control_paths,
    client_ids_all,
    server_state_path,
    exclude_prefixes=None,
    local_steps_map=None,
):
    # if exclude_prefixes is None:
    #     exclude_prefixes = []
    if not exclude_prefixes:
        exclude_prefixes = ['bn', 'running_mean', 'running_var', 'num_batches_tracked', 'pts_bbox_head.task_heads']

    global_ckpt = torch.load(initial_global_path, map_location="cpu")
    global_state = global_ckpt["state_dict"]
    mergeable_keys, _ = _split_keys(global_state, exclude_prefixes)

    client_controls = {}
    for cid, p in zip(client_ids_all, init_client_control_paths):
        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state = obj["state_dict"]
        else:
            state = obj
        _validate_control_keys(state, mergeable_keys, cid)
        client_controls[cid] = state

    global_control = {}
    for k in mergeable_keys:
        ref = client_controls[client_ids_all[0]][k]
        if torch.is_tensor(ref):
            acc = torch.zeros_like(ref)
            for cid in client_ids_all:
                acc.add_(client_controls[cid][k])
            acc.div_(float(len(client_ids_all)))
            global_control[k] = acc
        else:
            global_control[k] = copy.deepcopy(ref)

    global_momentum = _clone_state_dict(global_control)

    state = {
        "method": "PAdaMFed-VR",
        "round": 0,
        "created_at": time.time(),
        "client_ids_all": list(client_ids_all),
        "N": int(len(client_ids_all)),
        "last_sampled_client_ids": [],
        "S": None,
        "local_steps_map": {} if local_steps_map is None else {cid: int(local_steps_map[cid]) for cid in client_ids_all},
        "exclude_prefixes": list(exclude_prefixes),
        "global_control_variate": global_control,
        "global_momentum": global_momentum,
        "client_control_variates": {cid: _clone_state_dict(client_controls[cid]) for cid in client_ids_all},
        "last_broadcast_direction": None,
    }

    os.makedirs(os.path.dirname(server_state_path), exist_ok=True)
    torch.save(state, server_state_path)
    print(f"Initialized PAdaMFed-VR server state at {server_state_path}")


def _resolve_local_steps(client_ids, local_steps=None, local_steps_map=None):
    if local_steps_map is not None:
        missing = [cid for cid in client_ids if cid not in local_steps_map]
        if missing:
            raise ValueError(f"local_steps_map missing client IDs: {missing}")
        resolved = {}
        for cid in client_ids:
            k = int(local_steps_map[cid])
            if k <= 0:
                raise ValueError(f"Client {cid} has invalid local steps: {k}")
            resolved[cid] = k
        return resolved

    if local_steps is None:
        raise ValueError("Provide either local_steps or local_steps_map")

    if isinstance(local_steps, dict):
        missing = [cid for cid in client_ids if cid not in local_steps]
        if missing:
            raise ValueError(f"local_steps dict missing client IDs: {missing}")
        resolved = {}
        for cid in client_ids:
            k = int(local_steps[cid])
            if k <= 0:
                raise ValueError(f"Client {cid} has invalid local steps: {k}")
            resolved[cid] = k
        return resolved

    k = int(local_steps)
    if k <= 0:
        raise ValueError(f"Invalid shared local_steps: {k}")
    return {cid: k for cid in client_ids}


def _validate_inputs(models, output_paths, norm_weights, client_ids):
    n = len(models)
    if len(output_paths) != n:
        raise ValueError("output_paths length mismatch")
    if len(norm_weights) != n:
        raise ValueError("norm_weights length mismatch")
    if len(client_ids) != n:
        raise ValueError("client_ids length mismatch")
    s = sum(float(x) for x in norm_weights)
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"norm_weights must sum to 1.0, got {s}")


def _split_keys(state_dict, exclude_prefixes):
    mergeable_keys = []
    excluded_keys = []
    for k in state_dict.keys():
        if any(p in k for p in exclude_prefixes):
            excluded_keys.append(k)
        else:
            mergeable_keys.append(k)
    return mergeable_keys, excluded_keys


def _validate_state_dict_compatibility(global_state, client_states, client_ids):
    g_keys = set(global_state.keys())
    for cid, st in zip(client_ids, client_states):
        c_keys = set(st.keys())
        if g_keys != c_keys:
            missing = sorted(g_keys - c_keys)
            extra = sorted(c_keys - g_keys)
            raise RuntimeError(
                f"State-dict key mismatch for client {cid}. "
                f"Missing example: {missing[:10]}. Extra example: {extra[:10]}"
            )
        for k in global_state.keys():
            gv = global_state[k]
            cv = st[k]
            if torch.is_tensor(gv) and torch.is_tensor(cv):
                if gv.shape != cv.shape:
                    raise RuntimeError(f"Shape mismatch for key {k} in client {cid}: {gv.shape} vs {cv.shape}")
                if gv.dtype != cv.dtype:
                    raise RuntimeError(f"Dtype mismatch for key {k} in client {cid}: {gv.dtype} vs {cv.dtype}")

def _zero_like_control_state(control_state, mergeable_keys):
    out = {}
    for k in mergeable_keys:
        v = control_state[k]
        if torch.is_tensor(v):
            out[k] = torch.zeros_like(v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _bootstrap_global_control_and_momentum(
    current_client_controls,
    mergeable_keys,
    total_num_clients,
):
    new_global_control = {}

    for k in mergeable_keys:
        ref = None
        for cid in current_client_controls:
            if k in current_client_controls[cid]:
                ref = current_client_controls[cid][k]
                break

        if ref is None:
            raise RuntimeError(f"Could not find key '{k}' in any current client control state during bootstrap")

        if torch.is_tensor(ref):
            acc = torch.zeros_like(ref)
            for cid in current_client_controls:
                acc.add_(current_client_controls[cid][k])
            acc.div_(float(total_num_clients))
            new_global_control[k] = acc
        else:
            new_global_control[k] = copy.deepcopy(ref)

    new_global_momentum = _clone_state_dict(new_global_control)
    return new_global_control, new_global_momentum

def _validate_control_keys(control_state, mergeable_keys, cid):
    missing = [k for k in mergeable_keys if k not in control_state]
    if missing:
        raise RuntimeError(f"Control variate for client {cid} missing keys, example: {missing[:10]}")


def _clone_state_dict(state_dict):
    out = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            out[k] = v.clone()
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run PAdaMFed-VR server aggregation with per-client mapping from CLI arguments."
    )

    parser.add_argument("--inputs", nargs="+", required=True, help="Input client checkpoint paths")
    parser.add_argument("--outputs", nargs="+", required=True, help="Output client checkpoint paths")
    parser.add_argument(
        "--client-control-paths",
        nargs="+",
        required=True,
        help="Per-client control variate paths aligned with --client-ids",
    )
    parser.add_argument("--client-ids", nargs="+", required=True, help="Ordered sampled client IDs")

    parser.add_argument("--round-idx", type=int, default=None, help="Current round index t")
    parser.add_argument("--total-rounds", type=int, default=None, help="Total rounds T")

    parser.add_argument("--local-steps-a", type=int, default=None)
    parser.add_argument("--local-steps-b", type=int, default=None)
    parser.add_argument("--local-steps-c", type=int, default=None)
    parser.add_argument("--local-steps-d", type=int, default=None)
    parser.add_argument("--local-steps-e", type=int, default=None)

    parser.add_argument("--total-num-clients", type=int, default=None)
    parser.add_argument("--num-sampled-clients", type=int, default=None)

    parser.add_argument("--weight-a", type=float, default=None)
    parser.add_argument("--weight-b", type=float, default=None)
    parser.add_argument("--weight-c", type=float, default=None)
    parser.add_argument("--weight-d", type=float, default=None)
    parser.add_argument("--weight-e", type=float, default=None)

    parser.add_argument(
        "--prev-global-path",
        default="/workspace/work_dirs/PAdaMFed_VR/global_model.pth",
        help="Path to previous global checkpoint",
    )
    parser.add_argument(
        "--server-state-path",
        default="/workspace/work_dirs/PAdaMFed_VR/server_state.pt",
        help="Path to server state file",
    )
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument(
        "--exclude-prefixes",
        nargs="*",
        default=['bn', 'running_mean', 'running_var', 'num_batches_tracked', 'pts_bbox_head.task_heads'],
        help="Key substrings to exclude from merge/update",
    )
    parser.add_argument(
        "--keep-excluded-local",
        dest="keep_excluded_local",
        action="store_true",
        default=True,
        help="Keep excluded parameters from local client checkpoints",
    )
    parser.add_argument(
        "--no-keep-excluded-local",
        dest="keep_excluded_local",
        action="store_false",
        help="Overwrite excluded parameters from global checkpoint",
    )
    parser.add_argument(
        "--zero-optimizer-state",
        dest="zero_optimizer_state",
        action="store_true",
        default=True,
        help="Zero optimizer state tensors in redistributed checkpoints",
    )
    parser.add_argument(
        "--no-zero-optimizer-state",
        dest="zero_optimizer_state",
        action="store_false",
        help="Keep optimizer state tensors unchanged",
    )
    parser.add_argument(
        "--strict-key-check",
        dest="strict_key_check",
        action="store_true",
        default=True,
        help="Require exact key/shape/dtype compatibility across client checkpoints",
    )
    parser.add_argument(
        "--no-strict-key-check",
        dest="strict_key_check",
        action="store_false",
        help="Disable strict state-dict compatibility check",
    )
    parser.add_argument("--eps", type=float, default=1e-12)

    return parser.parse_args()


def _collect_per_client_values(args, client_ids):
    letters = ["a", "b", "c", "d", "e"]
    n = len(client_ids)
    if n > len(letters):
        raise ValueError("This CLI mapping currently supports up to 5 clients (A-E)")

    local_steps_raw = [
        args.local_steps_a,
        args.local_steps_b,
        args.local_steps_c,
        args.local_steps_d,
        args.local_steps_e,
    ]
    weights_raw = [
        args.weight_a,
        args.weight_b,
        args.weight_c,
        args.weight_d,
        args.weight_e,
    ]

    local_steps_map = {}
    effective_weights = []
    for i, cid in enumerate(client_ids):
        if local_steps_raw[i] is None:
            raise ValueError(f"Missing --local-steps-{letters[i]} for client {cid}")
        if local_steps_raw[i] <= 0:
            raise ValueError(f"--local-steps-{letters[i]} must be positive")

        if weights_raw[i] is None:
            raise ValueError(f"Missing --weight-{letters[i]} for client {cid}")
        if weights_raw[i] <= 0:
            raise ValueError(f"--weight-{letters[i]} must be positive")

        local_steps_map[cid] = int(local_steps_raw[i])
        effective_weights.append(float(weights_raw[i]))

    total_w = sum(effective_weights)
    if total_w <= 0:
        raise ValueError("Sum of provided weights must be positive")

    norm_weights = [w / total_w for w in effective_weights]
    return local_steps_map, norm_weights


def main():
    args = _parse_args()

    n = len(args.client_ids)
    if len(args.inputs) != n:
        raise ValueError("--inputs length must match --client-ids length")
    if len(args.outputs) != n:
        raise ValueError("--outputs length must match --client-ids length")
    if len(args.client_control_paths) != n:
        raise ValueError("--client-control-paths length must match --client-ids length")

    expected = ["ModelA", "ModelB", "ModelC", "ModelD", "ModelE"][:n]
    if args.client_ids != expected:
        raise ValueError(
            f"For this CLI, --client-ids must be in fixed order {expected}, got {args.client_ids}"
        )

    if args.num_sampled_clients is not None and args.num_sampled_clients != n:
        raise ValueError("--num-sampled-clients must match the number of provided sampled clients")

    local_steps_map, norm_weights = _collect_per_client_values(args, args.client_ids)

    PAdaMFed_VR(
        models=args.inputs,
        output_paths=args.outputs,
        norm_weights=norm_weights,
        client_ids=args.client_ids,
        prev_global_path=args.prev_global_path,
        server_state_path=args.server_state_path,
        client_control_paths=args.client_control_paths,
        round_idx=args.round_idx,
        total_rounds=args.total_rounds,
        local_steps_map=local_steps_map,
        total_num_clients=args.total_num_clients,
        num_sampled_clients=args.num_sampled_clients,
        eta=args.eta,
        gamma=args.gamma,
        beta=args.beta,
        exclude_prefixes=args.exclude_prefixes,
        keep_excluded_local=args.keep_excluded_local,
        zero_optimizer_state=args.zero_optimizer_state,
        strict_key_check=args.strict_key_check,
        eps=args.eps,
    )

if __name__ == "__main__":
    main()
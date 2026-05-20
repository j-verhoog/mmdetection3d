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
    if exclude_prefixes is None:
        exclude_prefixes = []

    _validate_inputs(models, output_paths, norm_weights, client_ids)

    num_clients = len(models)

    if total_num_clients is None:
        total_num_clients = len(client_ids)

    if num_sampled_clients is None:
        num_sampled_clients = len(client_ids)

    if local_steps is None:
        raise ValueError("local_steps must be provided for exact PAdaMFed-VR server update")

    if total_rounds is not None:
        if total_rounds <= 0:
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
    print(f"Local steps K            : {local_steps}")
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
            "K": int(local_steps),
            "exclude_prefixes": list(exclude_prefixes),
            "global_control_variate": None,
            "global_momentum": None,
            "client_control_variates": {},
            "last_broadcast_direction": None,
        }
        os.makedirs(os.path.dirname(server_state_path), exist_ok=True)
        torch.save(init_server_state, server_state_path)
        print("Round 0 initialization complete.")
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
            f"Expected server_state_path to exist for exact PAdaMFed-VR round update: {server_state_path}"
        )

    prev_client_controls = server_state.get("client_control_variates", {})
    prev_global_control = server_state.get("global_control_variate", None)
    prev_global_momentum = server_state.get("global_momentum", None)

    if prev_global_control is None:
        raise RuntimeError(
            "server_state['global_control_variate'] is None. "
            "For exact PAdaMFed-VR, initialize it before first exact round."
        )

    if prev_global_momentum is None:
        raise RuntimeError(
            "server_state['global_momentum'] is None. "
            "For exact PAdaMFed-VR, initialize it before first exact round."
        )

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

    missing_prev = [cid for cid in client_ids if cid not in prev_client_controls]
    if missing_prev:
        raise RuntimeError(
            f"Missing previous control variates for sampled clients in server state: {missing_prev}"
        )

    if eta is None or gamma is None or beta is None:
        if total_rounds is None:
            raise ValueError(
                "To use paper-derived PAdaMFed-VR hyperparameters, provide total_rounds so eta, gamma, beta can be computed"
            )
        S = float(num_sampled_clients)
        K = float(local_steps)
        T = float(total_rounds)
        eta = 1.0 / (K * T)
        gamma = (S * K) ** (1.0 / 3.0) / (T ** (2.0 / 3.0))
        beta = (S * K) ** (1.0 / 3.0) / (T ** (2.0 / 3.0))

    print("\n--- PAdaMFed-VR Scalars ---")
    print(f"  eta   = {eta}")
    print(f"  gamma = {gamma}")
    print(f"  beta  = {beta}")

    print("\n--- Phase 1: Aggregate local model-update direction g_model^t ---")
    g_model = {}
    for k in mergeable_keys:
        ref = global_state[k]
        if not torch.is_tensor(ref):
            continue
        acc = torch.zeros_like(ref)
        for client_state in client_states:
            acc.add_(global_state[k] - client_state[k])
        acc.div_(float(eta) * float(num_sampled_clients) * float(local_steps))
        g_model[k] = acc

    print(f"  Aggregated model direction for {len(g_model)} mergeable tensor keys")

    print("\n--- Phase 2: Update global model theta^{t+1} = theta^t - gamma * g_model^t ---")
    new_global_state = _clone_state_dict(global_state)
    for k in mergeable_keys:
        if k in g_model:
            new_global_state[k] = global_state[k] - float(gamma) * g_model[k]

    print("  Global model updated")

    print("\n--- Phase 3: Aggregate control variate c^t ---")
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

    print("\n--- Phase 8: Persist exact PAdaMFed-VR server state ---")
    updated_server_state = {
        "method": "PAdaMFed-VR",
        "round": 0 if round_idx is None else int(round_idx) + 1,
        "updated_at": time.time(),
        "client_ids_all": list(server_state.get("client_ids_all", client_ids)),
        "N": int(total_num_clients),
        "last_sampled_client_ids": list(client_ids),
        "S": int(num_sampled_clients),
        "K": int(local_steps),
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
):
    if exclude_prefixes is None:
        exclude_prefixes = []

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
        "K": None,
        "exclude_prefixes": list(exclude_prefixes),
        "global_control_variate": global_control,
        "global_momentum": global_momentum,
        "client_control_variates": {cid: _clone_state_dict(client_controls[cid]) for cid in client_ids_all},
        "last_broadcast_direction": None,
    }

    os.makedirs(os.path.dirname(server_state_path), exist_ok=True)
    torch.save(state, server_state_path)
    print(f"Initialized PAdaMFed-VR server state at {server_state_path}")


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
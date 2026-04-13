import os
import copy
import math
from typing import Optional, Union

from mmdetection3d.projects.cmt_40_epoch.fedcka.scripts_cmt_copy import runner
import torch
import torch.nn as nn
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.optimizer import Fp16OptimizerHook
from mmcv.parallel import DataContainer

@HOOKS.register_module()
class PAdaMFedVRFp16OptimizerHook(Fp16OptimizerHook):
    def __init__(
        self,
        client_id,
        work_dir,
        server_state_path,
        prev_global_path,
        local_steps_per_round,
        total_rounds,
        total_num_clients,
        num_sampled_clients,
        grad_clip: Optional[dict] = None,
        coalesce: bool = True,
        bucket_size_mb: int = -1,
        loss_scale: Union[float, str, dict] = 512.,
        distributed: bool = True,
        custom_fp16=None,
        exclude_prefixes=('bn', 'running_mean', 'running_var', 'num_batches_tracked', 'pts_bbox_head.task_heads'),
        eta=None,
        beta=None,
        eps=1e-12,
        strict_state_check=True,
        freeze_prev_global_norm_stats=True,
        log_interval=20,
        use_paramwise_lr_multipliers=False,
        paramwise_lr_multipliers=None,
        use_grad_clip=False,
        train_excluded_locally=True,
        warmup_if_missing_server_state=True,
    ):
        super(PAdaMFedVRFp16OptimizerHook, self).__init__(
            grad_clip=grad_clip,
            coalesce=coalesce,
            bucket_size_mb=bucket_size_mb,
            loss_scale=loss_scale,
            distributed=distributed,
        )

        self.client_id = client_id
        self.work_dir = work_dir
        self.server_state_path = server_state_path
        self.prev_global_path = prev_global_path

        self.local_steps_per_round = int(local_steps_per_round)
        self.total_rounds = int(total_rounds)
        self.total_num_clients = int(total_num_clients)
        self.num_sampled_clients = int(num_sampled_clients)

        self.custom_fp16 = {} if custom_fp16 is None else custom_fp16
        self.exclude_prefixes = tuple(exclude_prefixes)

        self.eta = eta
        self.beta = beta
        self.eps = float(eps)
        self.strict_state_check = bool(strict_state_check)
        self.freeze_prev_global_norm_stats = bool(freeze_prev_global_norm_stats)
        self.log_interval = int(log_interval)

        self.use_paramwise_lr_multipliers = bool(use_paramwise_lr_multipliers)
        self.paramwise_lr_multipliers = (
            {"img_backbone": 0.01, "img_neck": 0.1}
            if paramwise_lr_multipliers is None
            else dict(paramwise_lr_multipliers)
        )
        self.use_grad_clip = bool(use_grad_clip)
        self.train_excluded_locally = bool(train_excluded_locally)
        self.warmup_if_missing_server_state = bool(warmup_if_missing_server_state)

        self.prev_global_model = None
        self.prev_local_control = {}
        self.broadcast_direction = {}
        self.current_control_accum = {}
        self.param_name_to_param = {}
        self.mergeable_param_names = []
        self.excluded_param_names = []
        self.local_iter_count = 0
        self.active_vr_mode = False
        self._last_data_batch = None
        self._root_model = None

    def before_run(self, runner) -> None:
        super().before_run(runner)

        self._root_model = runner.model.module if hasattr(runner.model, "module") else runner.model

        for module_name, v in self.custom_fp16.items():
            if module_name not in self._root_model._modules:
                runner.logger.warning(f"[PAdaMFed-VR] custom_fp16 module '{module_name}' not found; skipping")
                continue
            self._root_model._modules[module_name].fp16_enabled = v

        os.makedirs(self.work_dir, exist_ok=True)

        if self.local_steps_per_round <= 0:
            raise ValueError("local_steps_per_round must be positive")
        if self.total_rounds <= 0:
            raise ValueError("total_rounds must be positive")
        if self.num_sampled_clients <= 0:
            raise ValueError("num_sampled_clients must be positive")

        if self.eta is None:
            self.eta = 1.0 / (self.local_steps_per_round * self.total_rounds)

        if self.beta is None:
            self.beta = ((self.num_sampled_clients * self.local_steps_per_round) ** (1.0 / 3.0)) / (self.total_rounds ** (2.0 / 3.0))

        self.param_name_to_param = {
            name: param for name, param in self._root_model.named_parameters() if param.requires_grad
        }

        self.mergeable_param_names = []
        self.excluded_param_names = []
        for name in self.param_name_to_param.keys():
            if self._is_excluded(name):
                self.excluded_param_names.append(name)
            else:
                self.mergeable_param_names.append(name)

        runner.logger.info(f"--- Initializing PAdaMFed-VR hook for {self.client_id} ---")
        runner.logger.info(f"[PAdaMFed-VR] eta={self.eta}, beta={self.beta}")
        runner.logger.info(f"[PAdaMFed-VR] mergeable params={len(self.mergeable_param_names)}, excluded params={len(self.excluded_param_names)}")
        runner.logger.info(f"[PAdaMFed-VR] use_paramwise_lr_multipliers={self.use_paramwise_lr_multipliers}")
        runner.logger.info(f"[PAdaMFed-VR] use_grad_clip={self.use_grad_clip}")
        runner.logger.info(f"[PAdaMFed-VR] train_excluded_locally={self.train_excluded_locally}")

        self._load_prev_local_control(runner)
        self._load_server_broadcast_direction(runner)
        self._build_prev_global_model(runner)
        self._init_current_control_accum()
        self.local_iter_count = 0

        self.active_vr_mode = (self.prev_global_model is not None) and (len(self.broadcast_direction) > 0)
        runner.logger.info(f"[PAdaMFed-VR] active_vr_mode={self.active_vr_mode}")

        if not self.active_vr_mode:
            runner.logger.warning(
                "[PAdaMFed-VR] Falling back to warmup mode: normalized local raw-gradient updates only. "
                "This is expected when previous global/server broadcast state is not yet available."
            )

    def before_train_iter(self, runner) -> None:
        if hasattr(runner, "data_batch"):
            self._last_data_batch = runner.data_batch

    def after_train_iter(self, runner) -> None:
        self._root_model.zero_grad()
        runner.optimizer.zero_grad()

        if "loss" not in runner.outputs:
            raise RuntimeError("runner.outputs does not contain 'loss'")

        self.loss_scaler.scale(runner.outputs["loss"]).backward()
        self.loss_scaler.unscale_(runner.optimizer)

        raw_current_grads = self._collect_current_raw_grads_and_accumulate_control()
        raw_current_grads = self._sanitize_grad_dict(raw_current_grads, runner, "raw_current_grads")

        prev_global_grads = None
        if self.active_vr_mode:
            prev_global_grads = self._compute_prev_global_grads_same_batch(runner)
            prev_global_grads = self._sanitize_grad_dict(prev_global_grads, runner, "prev_global_grads")

        update_grads = {}
        total_sq_norm = 0.0

        for name in self.mergeable_param_names:
            grad_cur = raw_current_grads[name]

            if self.active_vr_mode:
                grad_prev_global = prev_global_grads[name]
                c_i_prev = self.prev_local_control[name].to(grad_cur.device)
                broadcast = self.broadcast_direction[name].to(grad_cur.device)
                g_local = grad_cur + broadcast - float(self.beta) * c_i_prev - (1.0 - float(self.beta)) * grad_prev_global
            else:
                g_local = grad_cur

            update_grads[name] = g_local
            total_sq_norm += float(torch.sum(g_local * g_local).item())

        if self.train_excluded_locally:
            for name in self.excluded_param_names:
                g_local = raw_current_grads[name]
                update_grads[name] = g_local
                total_sq_norm += float(torch.sum(g_local * g_local).item())

        unclipped_norm = None
        if self.use_grad_clip:
            update_grads, clip_stats = self._apply_manual_grad_clip(update_grads, runner)
            total_sq_norm = 0.0
            unclipped_norm = clip_stats['pre_clip_norm']            
            for g in update_grads.values():
                total_sq_norm += float(torch.sum(g * g).item())
            if clip_stats["was_clipped"]:
                runner.logger.info(
                    f"[PAdaMFed-VR] manual grad clipping applied: "
                    f"pre_clip_norm={clip_stats['pre_clip_norm']:.6f}, "
                    f"max_norm={clip_stats['max_norm']:.6f}, "
                    f"clip_coef={clip_stats['clip_coef']:.6f}"
                )

        global_norm = math.sqrt(max(total_sq_norm, self.eps))

        if not math.isfinite(global_norm):
            raise RuntimeError(f"[PAdaMFed-VR] Non-finite global_norm detected: {global_norm}")

        with torch.no_grad():
            for name, param in self.param_name_to_param.items():
                if name not in update_grads:
                    continue

                lr_mult = 1.0
                if self.use_paramwise_lr_multipliers:
                    lr_mult = self._get_lr_multiplier(name)

                step_alpha = -float(self.eta) * float(lr_mult) / float(global_norm)
                param.data.add_(update_grads[name], alpha=step_alpha)

        self.loss_scaler.update(self._scale_update_param)
        runner.meta.setdefault("fp16", {})["loss_scaler"] = self.loss_scaler.state_dict()

        log_vars = {
                "padamfed_vr_global_norm": float(global_norm),
                "padamfed_vr_eta": float(self.eta),
                "padamfed_vr_beta": float(self.beta),
                "padamfed_vr_active_vr_mode": int(self.active_vr_mode),
                }
        if unclipped_norm is not None:
            log_vars["padamfed_vr_unclipped_norm"] = float(unclipped_norm)

        runner.log_buffer.update(
            log_vars,
            runner.outputs.get("num_samples", 1),
        )

        self.local_iter_count += 1

        if self.local_iter_count == 1 or self.local_iter_count % self.log_interval == 0:
            runner.logger.info(
                f"[PAdaMFed-VR] client={self.client_id} "
                f"iter={self.local_iter_count}/{self.local_steps_per_round} "
                f"global_norm={global_norm:.6f} "
                f"active_vr_mode={self.active_vr_mode}"
            )

    def after_run(self, runner):
        control_path = self._control_path()
        torch.save(self.current_control_accum, control_path)

        runner.logger.info(f"[PAdaMFed-VR] Saved control variate for {self.client_id} to {control_path}")
        runner.logger.info(f"[PAdaMFed-VR] local_iter_count={self.local_iter_count}, configured_local_steps={self.local_steps_per_round}")

        if self.local_iter_count != self.local_steps_per_round:
            runner.logger.warning(
                f"[PAdaMFed-VR] local_iter_count ({self.local_iter_count}) != local_steps_per_round ({self.local_steps_per_round}). "
                "The saved control variate was averaged using local_steps_per_round."
            )

        runner.meta.setdefault("PAdaMFed_VR", {})
        runner.meta["PAdaMFed_VR"]["client_id"] = self.client_id
        runner.meta["PAdaMFed_VR"]["control_path"] = control_path
        runner.meta["PAdaMFed_VR"]["local_steps_per_round"] = self.local_steps_per_round
        runner.meta["PAdaMFed_VR"]["active_vr_mode"] = self.active_vr_mode

    def _collect_current_raw_grads_and_accumulate_control(self):
        raw = {}

        for name, param in self.param_name_to_param.items():
            if param.grad is None:
                raw[name] = torch.zeros_like(param.data)
            else:
                raw[name] = param.grad.detach().clone()

        for name in self.mergeable_param_names:
            self.current_control_accum[name].add_(
                raw[name].detach().cpu(),
                alpha=1.0 / float(self.local_steps_per_round)
            )

        return raw

    def _unpack_data_batch(self, data_batch, device):
        """Manually unpack DataContainers since the unwrapped model doesn't do it automatically."""
        unpacked = {}
        for key, value in data_batch.items():
            if isinstance(value, DataContainer):
                # In DDP, data is wrapped in a list of length 1 for the current GPU
                val = value.data[0] if isinstance(value.data, list) else value.data
                
                # 1. Handle single tensors
                if isinstance(val, torch.Tensor):
                    val = val.to(device)
                # 2. Handle lists of tensors (mmdet3d standard for 'points', 'gt_bboxes', etc.)
                elif isinstance(val, list):
                    val = [v.to(device) if isinstance(v, torch.Tensor) else v for v in val]
                
                unpacked[key] = val
            else:
                unpacked[key] = value
        return unpacked


    def _compute_prev_global_grads_same_batch(self, runner):
        if self.prev_global_model is None:
            raise RuntimeError("prev_global_model is not initialized")

        data_batch = self._get_current_data_batch(runner)
        if data_batch is None:
            raise RuntimeError(
                "Could not access current data batch. "
                "Exact PAdaMFed-VR needs the same batch for both current and previous-global gradients."
            )
        device = next(self.prev_global_model.parameters()).device
        unpacked_data = self._unpack_data_batch(data_batch, device)
        self.prev_global_model.zero_grad()

        # outputs = self.prev_global_model.train_step(data_batch, optimizer=None)
        outputs = self.prev_global_model.train_step(unpacked_data, optimizer=None)
        
        if not isinstance(outputs, dict) or "loss" not in outputs:
            raise RuntimeError("prev_global_model.train_step did not return a dict containing 'loss'")

        prev_loss = outputs["loss"]
        prev_loss.backward()

        prev_named_params = dict(self.prev_global_model.named_parameters())
        grads = {}

        for name in self.mergeable_param_names:
            ref_param = self.param_name_to_param[name]
            prev_param = prev_named_params[name]

            if prev_param.grad is None:
                grads[name] = torch.zeros_like(ref_param.data)
            else:
                grads[name] = prev_param.grad.detach().clone().to(ref_param.device)

        return grads

    def _get_current_data_batch(self, runner):
        if hasattr(runner, "data_batch") and runner.data_batch is not None:
            return runner.data_batch
        if self._last_data_batch is not None:
            return self._last_data_batch
        return None

    def _build_prev_global_model(self, runner):
        if not os.path.exists(self.prev_global_path):
            if self.warmup_if_missing_server_state:
                runner.logger.warning(f"[PAdaMFed-VR] Previous global checkpoint not found: {self.prev_global_path}")
                self.prev_global_model = None
                return
            raise FileNotFoundError(f"Previous global checkpoint not found: {self.prev_global_path}")

        self.prev_global_model = copy.deepcopy(self._root_model)
        
        # Load the checkpoint manually
        ckpt = torch.load(self.prev_global_path, map_location="cpu")
        
        # 1. Auto-detect the root state dictionary
        state_dict = ckpt
        for possible_key in ["state_dict", "model", "model_state"]:
            if possible_key in state_dict and isinstance(state_dict[possible_key], dict):
                state_dict = state_dict[possible_key]
                break

        model_state_dict = self.prev_global_model.state_dict()
        transposed_count = 0
        loaded_count = 0

        # Wrap in no_grad to ensure clean, untracked memory updates
        with torch.no_grad():
            for k, param in model_state_dict.items():
                if k in state_dict:
                    # 2. Safely unwrap the tensor regardless of MMCV wrappers
                    ckpt_val = self._unwrap_tensor(state_dict[k])
                    
                    if ckpt_val is None or not torch.is_tensor(ckpt_val):
                        runner.logger.warning(f"[PAdaMFed-VR] Skip {k}: Could not extract a valid tensor from checkpoint.")
                        continue
                        
                    # 3. Handle spconv permutation manually
                    if ckpt_val.shape != param.shape and len(ckpt_val.shape) == 5 and len(param.shape) == 5:
                        expected_shape = (ckpt_val.shape[1], ckpt_val.shape[2], ckpt_val.shape[3], ckpt_val.shape[4], ckpt_val.shape[0])
                        if expected_shape == tuple(param.shape):
                            ckpt_val = ckpt_val.permute(1, 2, 3, 4, 0).contiguous()
                            transposed_count += 1
                    
                    # 4. Safely copy if shapes match
                    if ckpt_val.shape == param.shape:
                        param.copy_(ckpt_val)
                        loaded_count += 1
                    else:
                        runner.logger.warning(f"[PAdaMFed-VR] Skip {k}: ckpt shape {ckpt_val.shape} != model shape {param.shape}")

        runner.logger.info(f"[PAdaMFed-VR] Directly loaded {loaded_count} params. Auto-transposed {transposed_count} spconv weights.")

        if self.freeze_prev_global_norm_stats:
            self._freeze_norm_stats(self.prev_global_model)

        self.prev_global_model.train()
        self.prev_global_model.to(next(self._root_model.parameters()).device)

    def _load_prev_local_control(self, runner):
        path = self._control_path()

        if os.path.exists(path):
            loaded = torch.load(path, map_location="cpu")
            self.prev_local_control = {}

            for name in self.mergeable_param_names:
                if name not in loaded:
                    raise RuntimeError(f"Previous control variate missing key: {name}")
                # Target the exact device of the model parameter
                target_device = self.param_name_to_param[name].device
                self.prev_local_control[name] = loaded[name].clone().to(target_device)

            runner.logger.info(f"[PAdaMFed-VR] Loaded previous local control from {path}")
        else:
            self.prev_local_control = {}
            for name in self.mergeable_param_names:
                param = self.param_name_to_param[name]
                # Inherits parameter's device
                self.prev_local_control[name] = torch.zeros_like(param.data)
            runner.logger.info(f"[PAdaMFed-VR] No previous local control found for {self.client_id}; initialized zeros")

    def _load_server_broadcast_direction(self, runner):
        self.broadcast_direction = {}

        if not os.path.exists(self.server_state_path):
            if self.warmup_if_missing_server_state:
                runner.logger.warning(f"[PAdaMFed-VR] Server state not found: {self.server_state_path}")
                return
            raise FileNotFoundError(f"Server state not found: {self.server_state_path}")

        server_state = torch.load(self.server_state_path, map_location="cpu")

        loaded = server_state.get("last_broadcast_direction", None)
        if loaded is None:
            if self.warmup_if_missing_server_state:
                runner.logger.warning(
                    "[PAdaMFed-VR] server_state['last_broadcast_direction'] is missing or None. Warmup mode will be used."
                )
                return
            raise RuntimeError("server_state['last_broadcast_direction'] is missing.")

        for name in self.mergeable_param_names:
            if name not in loaded:
                raise RuntimeError(f"Broadcast direction missing key: {name}")
            # Target the exact device of the model parameter
            target_device = self.param_name_to_param[name].device
            self.broadcast_direction[name] = loaded[name].clone().to(target_device)

        runner.logger.info("[PAdaMFed-VR] Loaded broadcast direction from server state")

    def _init_current_control_accum(self):
        self.current_control_accum = {}
        for name in self.mergeable_param_names:
            param = self.param_name_to_param[name]
            # torch.zeros_like automatically inherits the device and dtype of `param.data`
            self.current_control_accum[name] = torch.zeros_like(param.data)

    def _apply_manual_grad_clip(self, grad_dict, runner):
        if not self.use_grad_clip:
            return grad_dict, {
                "was_clipped": False,
                "pre_clip_norm": 0.0,
                "max_norm": 0.0,
                "clip_coef": 1.0,
            }

        if self.grad_clip is None:
            raise ValueError("use_grad_clip=True but grad_clip is None")

        max_norm = float(self.grad_clip.get("max_norm", 0.0))
        norm_type = float(self.grad_clip.get("norm_type", 2.0))

        if max_norm <= 0:
            raise ValueError("grad_clip['max_norm'] must be positive when use_grad_clip=True")

        total_norm = self._compute_grad_dict_norm(grad_dict, norm_type)
        clip_coef = max_norm / (total_norm + self.eps)
        was_clipped = clip_coef < 1.0

        if was_clipped:
            for k in grad_dict:
                grad_dict[k] = grad_dict[k] * clip_coef

        return grad_dict, {
            "was_clipped": was_clipped,
            "pre_clip_norm": float(total_norm),
            "max_norm": float(max_norm),
            "clip_coef": float(min(1.0, clip_coef)),
        }

    def _compute_grad_dict_norm(self, grad_dict, norm_type):
        if len(grad_dict) == 0:
            return 0.0

        if norm_type == float("inf"):
            max_val = 0.0
            for g in grad_dict.values():
                if g.numel() == 0:
                    continue
                max_val = max(max_val, float(g.detach().abs().max().item()))
            return max_val

        total = 0.0
        for g in grad_dict.values():
            total += float(torch.sum(torch.abs(g.detach()) ** norm_type).item())
        return total ** (1.0 / norm_type)
    
    def _unwrap_tensor(self, val):
        """Recursively unwraps lists, tuples, or dicts to find the underlying tensor."""
        if torch.is_tensor(val):
            return val
        elif isinstance(val, (list, tuple)):
            # MMCV sometimes wraps tensors in tuples; usually the last item is the tensor
            return self._unwrap_tensor(val[-1])
        elif isinstance(val, dict):
            # If wrapped in a dict, check common keys
            for k in ["data", "tensor", "weight", "bias"]:
                if k in val:
                    return self._unwrap_tensor(val[k])
            # If it's a single-key dictionary, assume that's the tensor
            if len(val) == 1:
                return self._unwrap_tensor(next(iter(val.values())))
            return None
        return None

    def _sanitize_grad_dict(self, grad_dict, runner, tag):
        out = {}
        nonfinite_count = 0
        for name, g in grad_dict.items():
            if not torch.isfinite(g).all():
                nonfinite_count += 1
                out[name] = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                out[name] = g
        if nonfinite_count > 0:
            runner.logger.warning(f"[PAdaMFed-VR] Replaced non-finite gradients in {nonfinite_count} tensors for {tag}")
        return out

    def _get_lr_multiplier(self, name):
        for key, mult in self.paramwise_lr_multipliers.items():
            if key in name:
                return float(mult)
        return 1.0

    def _freeze_norm_stats(self, model):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                m.eval()

    def _control_path(self):
        return os.path.join(self.work_dir, f"{self.client_id}_control_variate.pth")

    def _is_excluded(self, name):
        return any(prefix in name for prefix in self.exclude_prefixes)
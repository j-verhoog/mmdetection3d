import os
import torch
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class FedSelectLocalTrainingHook(Hook):
    """
    Applies element-wise gradient scaling for FedSelect during local client training.

    Personalized parameters (mask=1) receive normal gradients.
    Global parameters (mask=0) receive heavily scaled-down gradients (or 0)
    to preserve global knowledge.

    NOT YET DONE IMPLEMENTATION!!! For Ali & Ricardo this frozen/unfrozen logic did not help training.
    """

    def __init__(self, mask_path, global_lr_scale=0.01, personal_lr_scale=1.0):
        """
        Args:
            mask_path (str): Path to the client's saved binary mask dictionary.
            global_lr_scale (float): Multiplier for global parameter gradients (default 0.01 per paper).
            personal_lr_scale (float): Multiplier for personalized parameter gradients.
        """
        self.mask_path = mask_path
        self.global_lr_scale = global_lr_scale
        self.personal_lr_scale = personal_lr_scale
        self.client_mask = None
        self._hook_handles = []
        self.trigger_path = '/workspace/work_dirs/fedselect_states/global_model.pth'

    def before_train(self, runner):
        """
        Loads the mask and registers PyTorch backward hooks on the parameters
        before training begins.
        """
        # Early exit: Do absolutely nothing during the FedAvg phase (Rounds 1-10)
        if not os.path.exists(self.trigger_path):
            runner.logger.info(
                f"FedAvg phase detected (trigger {self.trigger_path} not found). "
                f"FedSelect hooks disabled."
            )
            return

        runner.logger.info(f"--- Initializing FedSelect Hook ---")
        runner.logger.info(f"Loading mask from {self.mask_path}")

        if not os.path.exists(self.mask_path):
            runner.logger.warning(
                f"No mask found at {self.mask_path}. "
                f"FedSelect hooks disabled to avoid scaling all gradients as global."
            )
            self.client_mask = {}
            return
        else:
            self.client_mask = torch.load(self.mask_path, map_location='cpu')

        # Extra safety: if the mask exists but is empty, disable the hook.
        # This preserves normal FedAvg/local training behavior during rounds where no FedCKA mask was selected yet.
        if self.client_mask is None or len(self.client_mask) == 0:
            runner.logger.info(
                "FedSelect mask is empty. FedSelect hooks disabled for this round "
                "to keep current training behavior unchanged."
            )
            self.client_mask = {}
            return

        # Normalize mask keys by removing possible DDP prefix.
        # This prevents module.xxx vs xxx mismatches between checkpoint state_dict and runner.model.
        self.client_mask = {
            k.replace('module.', ''): v
            for k, v in self.client_mask.items()
        }

        model_keys = {
            name.replace('module.', '')
            for name, param in runner.model.named_parameters()
            if param.requires_grad
        }

        mask_keys = set(self.client_mask.keys())
        covered = model_keys & mask_keys
        missing = sorted(model_keys - mask_keys)
        extra = sorted(mask_keys - model_keys)

        coverage = len(covered) / max(1, len(model_keys))

        runner.logger.info(
            f"FedCKA mask coverage: {len(covered)}/{len(model_keys)} "
            f"trainable params covered ({coverage:.2%})"
        )

        if missing:
            runner.logger.warning(f"Example trainable params missing from mask: {missing[:10]}")

        if extra:
            runner.logger.warning(f"Example mask keys not found in model params: {extra[:10]}")

        # Hard stop if the mask clearly does not belong to this model/config.
        # This is especially important when switching from CMT to PointPillars.
        if coverage < 0.95:
            raise RuntimeError(
                f"FedSelect mask coverage too low ({coverage:.2%}). "
                f"Check checkpoint key names, module prefixes, or PointPillars config mismatch."
            )

        hooked_params = 0
        total_params = 0

        # Iterate through the model's parameters and attach the gradient modifier
        for name, param in runner.model.named_parameters():
            if not param.requires_grad:
                continue

            total_params += 1

            # CRITICAL FIX: Strip 'module.' prefix if wrapped in DistributedDataParallel
            clean_name = name.replace('module.', '')

            # Create a localized closure for the hook
            def make_grad_hook(param_name_clean):
                def grad_hook(grad):
                    # If parameter is not in mask despite passing coverage check,
                    # keep the original behavior: treat it as global.
                    if param_name_clean not in self.client_mask:
                        return grad * self.global_lr_scale

                    # Load the mask and move it to the parameter's device (GPU)
                    mask = self.client_mask[param_name_clean].to(
                        device=grad.device,
                        dtype=torch.bool
                    )

                    # Extra safety: prevent silent broadcasting or wrong masks.
                    if mask.shape != grad.shape:
                        raise RuntimeError(
                            f"FedSelect mask shape mismatch for {param_name_clean}: "
                            f"mask={tuple(mask.shape)}, grad={tuple(grad.shape)}"
                        )

                    # mask == 1 (Personalized) -> scale by personal_lr_scale
                    # mask == 0 (Global) -> scale by global_lr_scale
                    scale_matrix = torch.where(
                        mask,
                        torch.full_like(grad, self.personal_lr_scale),
                        torch.full_like(grad, self.global_lr_scale)
                    )

                    # Return the modified gradient
                    return grad * scale_matrix

                return grad_hook

            # Register the hook and save the handle so we can remove it later
            handle = param.register_hook(make_grad_hook(clean_name))
            self._hook_handles.append(handle)
            hooked_params += 1

        runner.logger.info(
            f"Registered FedSelect element-wise gradient hooks on "
            f"{hooked_params}/{total_params} trainable layers."
        )
        runner.logger.info(
            f"Global LR Scale: {self.global_lr_scale} | "
            f"Personal LR Scale: {self.personal_lr_scale}"
        )

    def after_train(self, runner):
        """
        Clean up the hooks after training to prevent memory leaks if the model
        is kept in memory for evaluation.
        """
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        runner.logger.info("Removed FedSelect gradient hooks.")
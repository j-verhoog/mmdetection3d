import os
import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class AlternatingPFLHook(Hook):
    """
    Implements Alternating Optimization for element-wise masked PFL.
    Alternates between training only personalized parameters and only global parameters.

        NOT YET DONE IMPLEMENTATION!!! For Ali & Ricardo this frozen/unfrozen logic did not help training.
    """
    def __init__(self, mask_path, personal_steps=50, global_steps=50):
        self.mask_path = mask_path
        self.personal_steps = personal_steps
        self.global_steps = global_steps
        
        self.client_mask = {}
        self._hook_handles = []
        self.is_personal_phase = False

    def before_train(self, runner):
        """Loads the mask and attaches the dynamic gradient hooks."""
        if os.path.exists(self.mask_path):
            self.client_mask = torch.load(self.mask_path, map_location='cpu')
            runner.logger.info(f"Loaded PFL mask from {self.mask_path}")
        else:
            runner.logger.warning("No mask found. Treating all parameters as Global.")

        # Attach backward hooks to all trainable parameters
        for name, param in runner.model.named_parameters():
            if not param.requires_grad:
                continue
            
            def make_grad_hook(param_name):
                def grad_hook(grad):
                    # If parameter has no mask, it defaults to entirely global
                    if param_name not in self.client_mask:
                        return grad * 0.0 if self.is_personal_phase else grad

                    mask = self.client_mask[param_name].to(grad.device)
                    
                    if self.is_personal_phase:
                        # Personal Phase: Keep mask==1 (Personal), zero out mask==0 (Global)
                        return grad * mask.float()
                    else:
                        # Global Phase: Keep mask==0 (Global), zero out mask==1 (Personal)
                        return grad * (~mask).float()
                        
                return grad_hook

            handle = param.register_hook(make_grad_hook(name))
            self._hook_handles.append(handle)

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        """Switches the active phase based on the current global iteration step."""
        phase_length = self.personal_steps + self.global_steps
        current_step = runner.iter % phase_length
        
        # Toggle phase flag used by the gradient hooks
        self.is_personal_phase = (current_step < self.personal_steps)

    def after_train(self, runner):
        """Cleans up hooks to prevent memory leaks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
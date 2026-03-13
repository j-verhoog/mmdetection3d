import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS

# Import the core calculation functions from standard mmdet so it functions 
# identically to the original when AutoFed is disabled.
from mmdet.models.losses.focal_loss import (
    py_focal_loss_with_prob,
    py_sigmoid_focal_loss,
    sigmoid_focal_loss
)

@MODELS.register_module()
class FocalLossCustom(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False,
                 use_autofed=False,  
                 p_th=0.9):          
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            activated (bool, optional): Whether the input is activated.
                If True, it means the input has been activated and can be
                treated as probabilities. Else, it should be treated as logits.
                Defaults to False.
            use_autofed (bool, optional): Whether to use AutoFed logic to set 
                loss to 0 when model confidence is high on background labels. Defaults to False.
            p_th (float, optional): Confidence threshold for AutoFed. Defaults to 0.9.
        """
        super(FocalLossCustom, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated
        
        # New AutoFed properties
        self.use_autofed = use_autofed
        self.p_th = p_th
        self.print_frequency = 100
        self._step_counter = 0

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
            
        # =====================================================================
        # NEW AUTOFED LOGIC
        # =====================================================================
        if self.use_autofed:
            with torch.no_grad(): # Mask creation shouldn't track gradients
                # 1. Get the actual probabilities
                prob = pred if self.activated else torch.sigmoid(pred)
                
                # 2. Align target shape to pred (handle class indices vs one-hot)
                if target.shape != pred.shape:
                    num_classes = pred.size(-1)
                    target_safe = target.clone()
                    target_safe[target_safe < 0] = num_classes # map ignore indices to background
                    target_one_hot = F.one_hot(target_safe, num_classes=num_classes + 1)[..., :num_classes]
                else:
                    target_one_hot = target
                
                # 3. Create the mask: Prob > threshold AND ground truth says nothing is there (0)
                is_negative = (target_one_hot == 0)
                trust_model_mask = (prob > self.p_th) & is_negative
                
                # 4. Throttled Logging (Every 100 steps)
                if self._step_counter % self.print_frequency == 0:
                    triggered_count = trust_model_mask.sum().item()
                    bg_count = is_negative.sum().item()
                    if bg_count > 0:
                        pct = (triggered_count / bg_count) * 100
                        print(f"\n[AutoFed] Step {self._step_counter} | Ignored {triggered_count}/{bg_count} background anchors ({pct:.4f}%)")
                self._step_counter += 1
                
            # 5. Apply the mask by zeroing out the weights for these predictions
            if weight is None:
                weight = torch.ones_like(pred)
            else:
                # Ensure weight matches pred dimensions so we can apply the boolean mask
                if weight.dim() == pred.dim() - 1:
                    weight = weight.unsqueeze(-1)
                weight = weight.expand_as(pred).clone()
            
            # Loss will be exactly 0 where weight is 0
            weight[trust_model_mask] = 0.0
        # =====================================================================

        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = py_focal_loss_with_prob
            else:
                if torch.cuda.is_available() and pred.is_cuda:
                    calculate_loss_func = sigmoid_focal_loss
                else:
                    num_classes = pred.size(1)
                    target = F.one_hot(target, num_classes=num_classes + 1)
                    target = target[:, :num_classes]
                    calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
            
        return loss_cls
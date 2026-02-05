"""
Implementation of specific metrics for model comparison:
- CKA (Centered Kernel Alignment) for layer activations
- W2 (Wasserstein-2) distance for BatchNorm running statistics
- Effective Scale/Bias for BatchNorm affine parameters
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple, Any
from torch.nn.modules.batchnorm import _BatchNorm
from scipy.stats import wasserstein_distance

from metrics_base import ActivationMetric, BatchNormMetric, MetricCalculator


class CKAMetric(ActivationMetric):
    """
    Centered Kernel Alignment (CKA) for layer activation similarity.
    
    Measures representational similarity between two layers by comparing
    normalized kernel matrices. Range: [0, 1] where 1 = perfect alignment.
    
    Reference: Kornblith et al., "Similarity of Neural Network Representations 
    Revisited" (ICML 2019)
    """
    
    def compute_layer_score(self, activation_a: torch.Tensor, activation_b: torch.Tensor) -> float:
        """
        Compute CKA between two layer activations.
        
        Args:
            activation_a: Activation tensor from model A
            activation_b: Activation tensor from model B
            
        Returns:
            float: CKA score in range [0, 1]
        """
        if activation_a is None or activation_b is None:
            return float('nan')
        
        x = self._normalize_activation(activation_a)
        y = self._normalize_activation(activation_b)
        
        if x is None or y is None:
            return float('nan')
        
        x, y = self._match_sample_count(x, y)
        if x is None or y is None:
            return float('nan')
        
        # Ensure same device for computation
        if x.device != y.device:
            y = y.to(x.device)
        
        # Center the features
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        
        # Compute Gram matrices
        gram_x = torch.matmul(x.t(), x)
        gram_y = torch.matmul(y.t(), y)
        gram_xy = torch.matmul(y.t(), x)
        
        # HSIC computation
        hsic_xy = torch.norm(gram_xy, p='fro') ** 2
        hsic_xx = torch.norm(gram_x, p='fro') ** 2
        hsic_yy = torch.norm(gram_y, p='fro') ** 2
        
        # CKA formula
        denom = torch.sqrt(hsic_xx * hsic_yy)
        if denom > 0:
            return (hsic_xy / denom).item()
        else:
            return float('nan')
    
    def get_name(self) -> str:
        return "CKA"


class W2BNStatsMetric(BatchNormMetric):
    """
    Wasserstein-2 (W2) distance for BatchNorm running statistics.
    
    Measures how much the internal statistics (running mean/variance) differ
    between models. Lower values = more similar distributions.
    
    Normalized to [0, 1] where 0 = identical, 1 = completely different.
    """
    
    def compute_layer_score(self, bn_module_a: _BatchNorm, bn_module_b: _BatchNorm) -> float:
        """
        Compute W2 distance between BN statistics of two modules.
        
        Args:
            bn_module_a: BatchNorm module from model A
            bn_module_b: BatchNorm module from model B
            
        Returns:
            float: Normalized W2 distance (lower = more similar)
        """
        if not isinstance(bn_module_a, _BatchNorm) or not isinstance(bn_module_b, _BatchNorm):
            return float('nan')
        
        stats_a = self.extract_bn_stats(bn_module_a)
        stats_b = self.extract_bn_stats(bn_module_b)
        
        if stats_a is None or stats_b is None:
            return float('nan')
        
        mean_a = stats_a['mean'].cpu().numpy()
        var_a = stats_a['var'].cpu().numpy()
        mean_b = stats_b['mean'].cpu().numpy()
        var_b = stats_b['var'].cpu().numpy()
        
        if len(mean_a) == 0 or len(mean_b) == 0:
            return float('nan')
        
        # W2 distance between 1D Gaussians: sqrt((mu1-mu2)^2 + (sigma1-sigma2)^2)
        # This is the standard form for comparing distributions
        mean_diff_sq = np.sum((mean_a - mean_b) ** 2)
        var_diff_sq = np.sum((np.sqrt(var_a + 1e-5) - np.sqrt(var_b + 1e-5)) ** 2)
        
        w2_distance = np.sqrt(mean_diff_sq + var_diff_sq)
        
        # Normalize: assume max reasonable distance is ~10
        # (can be scaled based on typical values in your models)
        normalized = np.clip(w2_distance / 10.0, 0.0, 1.0)
        
        return float(normalized)
    
    def get_name(self) -> str:
        return "W2 BN Stats"


class EffectiveScaleBiasMetric(BatchNormMetric):
    """
    Effective scale/bias impact for BatchNorm affine parameters.
    
    Measures how much the learned scale (γ) and bias (β) parameters
    change the function. Computes the relative magnitude of these
    learnable parameters.
    
    Returns normalized score based on parameter magnitudes.
    """
    
    def compute_layer_score(self, bn_module_a: _BatchNorm, bn_module_b: _BatchNorm) -> float:
        """
        Compute effective scale/bias divergence between two BN modules.
        
        Measures the difference in how much the affine parameters (γ, β)
        scale the normalized inputs. Returns similarity score [0, 1].
        
        Args:
            bn_module_a: BatchNorm module from model A
            bn_module_b: BatchNorm module from model B
            
        Returns:
            float: Similarity of effective parameters (1 = identical, 0 = different)
        """
        if not isinstance(bn_module_a, _BatchNorm) or not isinstance(bn_module_b, _BatchNorm):
            return float('nan')
        
        affine_a = self.extract_bn_affine(bn_module_a)
        affine_b = self.extract_bn_affine(bn_module_b)
        
        if affine_a is None or affine_b is None:
            return float('nan')
        
        weight_a = affine_a['weight']
        bias_a = affine_a['bias']
        weight_b = affine_b['weight']
        bias_b = affine_b['bias']
        
        if weight_a is None or weight_b is None or bias_a is None or bias_b is None:
            return float('nan')
        
        # Compute "effective parameter" vectors: [gamma, beta]
        params_a = torch.cat([weight_a.unsqueeze(1), bias_a.unsqueeze(1)], dim=1)
        params_b = torch.cat([weight_b.unsqueeze(1), bias_b.unsqueeze(1)], dim=1)
        
        if params_a.shape != params_b.shape:
            return float('nan')
        
        # Cosine similarity between parameter vectors
        # Higher = more similar function behavior
        params_a_flat = params_a.reshape(-1)
        params_b_flat = params_b.reshape(-1)
        
        # Normalize
        norm_a = torch.norm(params_a_flat)
        norm_b = torch.norm(params_b_flat)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        cosine_sim = torch.dot(params_a_flat, params_b_flat) / (norm_a * norm_b)
        
        # Convert from [-1, 1] to [0, 1]
        score = (cosine_sim.item() + 1.0) / 2.0
        return float(np.clip(score, 0.0, 1.0))
    
    def get_name(self) -> str:
        return "Effective Scale/Bias"


# Factory for easy metric retrieval
METRICS_REGISTRY = {
    'cka': CKAMetric,
    'w2': W2BNStatsMetric,
    'effective_scale_bias': EffectiveScaleBiasMetric,
}


def get_metric(metric_name: str) -> Optional[MetricCalculator]:
    """Get metric instance by name."""
    metric_class = METRICS_REGISTRY.get(metric_name.lower())
    if metric_class:
        return metric_class()
    return None


def get_available_metrics() -> list:
    """Get list of available metric names."""
    return list(METRICS_REGISTRY.keys())
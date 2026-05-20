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
    Centered Kernel Alignment (CKA) - Memory Efficient Version.
    Uses feature-space computation (d x d) instead of sample-space (n x n)
    to avoid OOM on large feature maps.
    """
    
    def compute_layer_score(self, activation_a: torch.Tensor, activation_b: torch.Tensor) -> float:
        if activation_a is None or activation_b is None:
            return float('nan')
        
        # 1. Normalize and Flatten -> (Num_Pixels, Num_Features)
        # Shape becomes [40000, 256] for a 200x200 map
        x = self._normalize_activation(activation_a)
        y = self._normalize_activation(activation_b)
        
        if x is None or y is None:
            return float('nan')
        
        x, y = self._match_sample_count(x, y)
        if x is None or y is None:
            return float('nan')
            
        if x.device != y.device:
            y = y.to(x.device)
            
        # 2. Use Double Precision for stability
        x = x.double() 
        y = y.double()
        
        # 3. Center the features (Mean centering columns)
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

        # 4. MEMORY EFFICIENT COMPUTATION
        # Instead of Gram Matrix (N x N), we compute Covariance (D x D)
        # Identity: ||X X^T||_F^2 == ||X^T X||_F^2
        
        # Transpose first: (256, 40000) @ (40000, 256) -> (256, 256)
        # This matrix is tiny (256x256) regardless of image size!
        xtx = torch.matmul(x.t(), x)
        yty = torch.matmul(y.t(), y)
        xty = torch.matmul(x.t(), y)
        
        # 5. HSIC Calculation using Frobenius Norms
        # HSIC(X,Y) = ||Y^T X||_F^2
        hsic_xy = torch.norm(xty, p='fro') ** 2
        hsic_xx = torch.norm(xtx, p='fro') ** 2
        hsic_yy = torch.norm(yty, p='fro') ** 2
        
        # 6. Final Score
        denom = torch.sqrt(hsic_xx * hsic_yy)
        
        if denom > 0:
            cka_score = hsic_xy / denom
            cka_score = torch.clamp(cka_score, 0.0, 1.0)
            
            # Debug Print (Only if significant mismatch)
            val = cka_score.item()
            
            # debug if same model is used
            # if 1.0 - val > 1e-3: 
            #      print(f"CKA MISMATCH: {1.0 - val:.4f} distance at {activation_a.shape}")

            print(f"CKA Score: {1.0 - val:.4f} (Distance) | {val:.4f} (Similarity)")
            # Return Distance (0 = Identical)
            return float(1.0 - val)
        else:
            return float('nan')
    
    def get_name(self) -> str:
        return "CKA"
    

class W2BNStatsMetric(BatchNormMetric):
    """
    Wasserstein-2 inspired metric for BatchNorm running statistics.

    Computes per-channel normalized mean and sigma differences:
        d_mu    = |mu_a - mu_b| / sqrt((var_a + var_b)/2)
        d_sigma = |sigma_a - sigma_b| / ((sigma_a + sigma_b)/2)

    Final score = RMS over channels of sqrt(d_mu^2 + d_sigma^2)
    """
    
    def compute_layer_score(self, bn_module_a: _BatchNorm, bn_module_b: _BatchNorm) -> float:
        # 1. Validation
        if not isinstance(bn_module_a, _BatchNorm) or not isinstance(bn_module_b, _BatchNorm):
            return float('nan')
        
        stats_a = self.extract_bn_stats(bn_module_a)
        stats_b = self.extract_bn_stats(bn_module_b)
        
        if stats_a is None or stats_b is None:
            return float('nan')
        
        # 2. Robust Extraction
        try:
            ma, va = stats_a['mean'].cpu().numpy(), stats_a['var'].cpu().numpy()
            mb, vb = stats_b['mean'].cpu().numpy(), stats_b['var'].cpu().numpy()
        except Exception:
            return float('nan')

        # Check for model corruption
        if not (np.isfinite(ma).all() and np.isfinite(va).all() and
                np.isfinite(mb).all() and np.isfinite(vb).all()):
            return float('nan')

        eps = 1e-7

        # 3. Robust sigma computation
        sigma_a = np.sqrt(np.maximum(va, 0) + eps)
        sigma_b = np.sqrt(np.maximum(vb, 0) + eps)

        # 4. Channel-wise normalized differences
        pooled_std = np.sqrt(0.5 * (va + vb) + eps)
        d_mu = np.abs(ma - mb) / pooled_std

        mean_sigma = 0.5 * (sigma_a + sigma_b) + eps
        d_sigma = np.abs(sigma_a - sigma_b) / mean_sigma

        # 5. Combine per channel
        per_channel_score = np.sqrt(d_mu**2 + d_sigma**2)

        # 6. Aggregate across channels
        layer_score = float(np.sqrt(np.mean(per_channel_score**2)))

        # 7. Informative diagnostics
        mean_mu = float(np.mean(d_mu))
        mean_sigma_diff = float(np.mean(d_sigma))
        median_score = float(np.median(per_channel_score))
        p90_score = float(np.percentile(per_channel_score, 90))
        max_score = float(np.max(per_channel_score))

        print(
            f"W2-rel stats | "
            f"mean(d_mu)={mean_mu:.4f}, "
            f"mean(d_sigma)={mean_sigma_diff:.4f}, "
            f"median={median_score:.4f}, "
            f"p90={p90_score:.4f}, "
            f"max={max_score:.4f}, "
            f"layer_score(RMS)={layer_score:.4f}"
        )

        # 8. Clip for visualization consistency
        return float(np.clip(layer_score, 0.0, 1.0))

    def get_name(self) -> str:
        return "W2 BN Stats"


class EffectiveScaleBiasMetric(BatchNormMetric):
    """
    Relative drift of BN affine parameters (gamma=scale, beta=shift),
    reported as symmetric relative L2 differences for each.
    """
    
    def compute_layer_score(self, bn_module_a: _BatchNorm, bn_module_b: _BatchNorm) -> float:
        if not isinstance(bn_module_a, _BatchNorm) or not isinstance(bn_module_b, _BatchNorm):
            return float('nan')
        
        affine_a = self.extract_bn_affine(bn_module_a)
        affine_b = self.extract_bn_affine(bn_module_b)
        
        if affine_a is None or affine_b is None:
            return float('nan')
        
        w_a, b_a = affine_a['weight'], affine_a['bias']
        w_b, b_b = affine_b['weight'], affine_b['bias']
        
        if w_a is None or w_b is None or b_a is None or b_b is None:
            return float('nan')

        if w_a.shape != w_b.shape or b_a.shape != b_b.shape:
            return float('nan')

        eps = 1e-6

        d_gamma_num = torch.norm(w_a.flatten() - w_b.flatten(), p=2)
        d_gamma_den = 0.5 * (torch.norm(w_a.flatten(), p=2) + torch.norm(w_b.flatten(), p=2)) + eps
        d_gamma = d_gamma_num / d_gamma_den

        d_beta_num = torch.norm(b_a.flatten() - b_b.flatten(), p=2)
        d_beta_den = 0.5 * (torch.norm(b_a.flatten(), p=2) + torch.norm(b_b.flatten(), p=2)) + eps
        d_beta = d_beta_num / d_beta_den

        layer_score = 0.5 * (d_gamma + d_beta)

        print(
            f"Affine-rel | "
            f"d_gamma={float(d_gamma.item()):.4f}, "
            f"d_beta={float(d_beta.item()):.4f}, "
            f"layer_score(avg)={float(layer_score.item()):.4f}"
        )

        return float(layer_score.item())
    
    def get_name(self) -> str:
        return "Effective Scale/Bias"


class BhattacharyyaMetric(BatchNormMetric):
    """
    Bhattacharyya Coefficient for BatchNorm distributions.
    
    Measures the amount of overlap between two distributions.
    Returns the 'Bhattacharyya Coefficient' (BC) which is naturally normalized.
    
    Range:
    1.0 = Distributions are identical (Maximum Overlap)
    0.0 = Distributions have no overlap
    I swapped this to be a distance metric where 0 = identical and 1 = different, to be consistent with other metrics.
    """
    
    def compute_layer_score(self, bn_module_a: _BatchNorm, bn_module_b: _BatchNorm) -> float:
        if not isinstance(bn_module_a, _BatchNorm) or not isinstance(bn_module_b, _BatchNorm):
            return float('nan')
        
        stats_a = self.extract_bn_stats(bn_module_a)
        stats_b = self.extract_bn_stats(bn_module_b)
        
        if stats_a is None or stats_b is None:
            return float('nan')
        
        # Get Means (mu) and Variances (sigma^2)
        mu1 = stats_a['mean'].cpu().numpy()
        v1 = stats_a['var'].cpu().numpy() + 1e-6  # Add epsilon for stability
        
        mu2 = stats_b['mean'].cpu().numpy()
        v2 = stats_b['var'].cpu().numpy() + 1e-6
        
        if len(mu1) == 0 or len(mu2) == 0:
            return float('nan')

        # --- Calculate Bhattacharyya Distance (Element-wise) ---
        # Term 1: Separability due to means
        # (mu1 - mu2)^2 / (4 * (v_avg)) 
        # Note: 4 * avg = 2 * (v1 + v2)
        term1 = (mu1 - mu2)**2 / (4 * (v1 + v2) / 2)
        
        # Term 2: Separability due to covariance (variances)
        # 0.5 * ln( (v1+v2)/2 / sqrt(v1*v2) )
        term2 = 0.5 * np.log((v1 + v2) / (2 * np.sqrt(v1 * v2)))
        
        b_distance = term1 + term2
        
        # --- Convert to Coefficient (Similarity) ---
        # BC = exp( -Distance )
        # This is mathematically bounded [0, 1]
        b_coefficient = np.exp(-b_distance)
        
        print(
            f"Bhattacharyya | "
            f"mean_term={float(np.mean(term1)):.4f}, "
            f"cov_term={float(np.mean(term2)):.4f}, "
            f"avg_distance={float(np.mean(b_distance)):.4f}, "
            f"b_coefficient={float(np.mean(b_coefficient)):.4f}"
        )
        # Return the average overlap across all channels
        return 1 - float(np.mean(b_coefficient))

    def get_name(self) -> str:
        return "Bhattacharyya Coeff"
    

# Factory for easy metric retrieval
METRICS_REGISTRY = {
    'cka': CKAMetric,
    'w2': W2BNStatsMetric,
    'effective_scale_bias': EffectiveScaleBiasMetric,
    'bhattacharyya': BhattacharyyaMetric,
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
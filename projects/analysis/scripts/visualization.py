"""
Visualization utilities for model comparison results.
Generates heatmaps for each metric with consistent styling.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from comparison_utils import sort_layers_by_network_order


class ComparisonVisualizer:
    """
    Handles visualization of multi-metric comparison results.
    """
    
    # Color schemes for different metrics
    COLORMAPS = {
        'CKA': 'viridis',
        'W2 BN Stats': 'RdYlGn_r',  # Red (high) to Green (low)
        'Effective Scale/Bias': 'viridis',
    }
    
    # V-min and max for consistent scaling
    VMIN_MAX = {
        'CKA': (0.0, 1.0),
        'W2 BN Stats': (0.0, 1.0),
        'Effective Scale/Bias': (0.0, 1.0),
    }
    
    @staticmethod
    def _get_comparison_titles(comparisons: List[Tuple[str, str, str]]) -> List[str]:
        """Extract comparison titles from (title, model1, model2) tuples."""
        return [c[0] for c in comparisons]
    
    @staticmethod
    def _build_matrix(
        results: Dict[str, Dict[str, float]],
        metric_name: str,
        comparisons: List[Tuple[str, str, str]],
        layer_order: List[str]
    ) -> np.ndarray:
        """
        Build matrix for a specific metric.
        
        Args:
            results: Dict[comparison_title] -> Dict[layer_name] -> score
            metric_name: Name of metric to extract
            comparisons: List of (title, model1, model2) tuples
            layer_order: Ordered list of layer names
            
        Returns:
            Matrix of shape (num_comparisons, num_layers)
        """
        matrix = np.zeros((len(comparisons), len(layer_order)))
        
        for i, (comp_title, _, _) in enumerate(comparisons):
            if comp_title in results and metric_name in results[comp_title]:
                metric_scores = results[comp_title][metric_name]
                
                for j, layer in enumerate(layer_order):
                    matrix[i, j] = metric_scores.get(layer, 0.0)
        
        return matrix
    
    @staticmethod
    def plot_single_metric(
        matrix: np.ndarray,
        comparisons: List[Tuple[str, str, str]],
        layer_order: List[str],
        metric_name: str,
        figsize: Tuple[int, int] = (20, 6),
        output_path: str = None
    ) -> str:
        """
        Plot heatmap for a single metric.
        
        Args:
            matrix: Comparison matrix (num_comparisons, num_layers)
            comparisons: List of (title, model1, model2) tuples
            layer_order: Ordered list of layer names
            metric_name: Name of metric (for title and coloring)
            figsize: Figure size
            output_path: If provided, save to this path
            
        Returns:
            Output filename
        """
        comp_titles = [c[0] for c in comparisons]
        
        # Get colormap and vmin/vmax
        cmap = ComparisonVisualizer.COLORMAPS.get(metric_name, 'viridis')
        vmin, vmax = ComparisonVisualizer.VMIN_MAX.get(metric_name, (0.0, 1.0))
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            matrix,
            xticklabels=layer_order,
            yticklabels=comp_titles,
            annot=False,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            cbar_kws={'label': metric_name}
        )
        
        plt.title(f"{metric_name} Comparison Heatmap")
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        
        # Generate output filename if not provided
        if output_path is None:
            metric_safe_name = metric_name.lower().replace(" ", "_").replace("/", "_")
            output_path = f"{metric_safe_name}_heatmap.png"
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    @staticmethod
    def plot_all_metrics(
        results: Dict[str, Dict[str, Dict[str, float]]],
        comparisons: List[Tuple[str, str, str]],
        metric_names: List[str],
        figsize_per_metric: Tuple[int, int] = (20, 6),
        output_dir: str = ".",
    ) -> Dict[str, str]:
        """
        Generate heatmaps for all metrics.
        
        Args:
            results: Dict[comparison_title] -> Dict[metric_name] -> Dict[layer_name] -> score
            comparisons: List of (title, model1, model2) tuples
            metric_names: List of metric names to plot
            figsize_per_metric: Figure size per metric
            output_dir: Directory to save plots
            
        Returns:
            Dict mapping metric names to output filenames
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        output_files = {}
        
        # Get all layers from all comparisons and metrics
        all_layers = set()
        for comp_results in results.values():
            for metric_scores in comp_results.values():
                all_layers.update(metric_scores.keys())
        
        layer_order = sort_layers_by_network_order(list(all_layers))
        
        # Plot each metric
        for metric_name in metric_names:
            print(f"Plotting {metric_name}...")
            
            matrix = ComparisonVisualizer._build_matrix(
                results, metric_name, comparisons, layer_order
            )
            
            output_path = os.path.join(
                output_dir,
                f"{metric_name.lower().replace(' ', '_').replace('/', '_')}_heatmap.png"
            )
            
            output_file = ComparisonVisualizer.plot_single_metric(
                matrix, comparisons, layer_order, metric_name,
                figsize=figsize_per_metric,
                output_path=output_path
            )
            
            output_files[metric_name] = output_file
            print(f"  Saved: {output_file}")
        
        return output_files


def create_comparison_report(
    results: Dict[str, Dict[str, Dict[str, float]]],
    comparisons: List[Tuple[str, str, str]],
    metric_names: List[str],
    num_samples: int,
    output_dir: str = "."
) -> str:
    """
    Create a text report of comparison results.
    
    Args:
        results: Comparison results
        comparisons: List of comparisons
        metric_names: List of metrics
        num_samples: Number of samples used
        output_dir: Directory for output
        
    Returns:
        Path to report file
    """
    import os
    
    report_path = os.path.join(output_dir, "comparison_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Samples processed: {num_samples}\n")
        f.write(f"Comparisons: {len(comparisons)}\n")
        f.write(f"Metrics: {', '.join(metric_names)}\n\n")
        
        # Write results per comparison
        for comp_title, model1, model2 in comparisons:
            f.write(f"\n--- Comparison: {comp_title} ---\n")
            f.write(f"  Model A: {model1}\n")
            f.write(f"  Model B: {model2}\n\n")
            
            if comp_title in results:
                for metric_name in metric_names:
                    if metric_name in results[comp_title]:
                        scores = results[comp_title][metric_name]
                        
                        f.write(f"  {metric_name}:\n")
                        f.write(f"    Mean: {np.mean(list(scores.values())):.4f}\n")
                        f.write(f"    Std:  {np.std(list(scores.values())):.4f}\n")
                        f.write(f"    Min:  {np.min(list(scores.values())):.4f}\n")
                        f.write(f"    Max:  {np.max(list(scores.values())):.4f}\n\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    return report_path
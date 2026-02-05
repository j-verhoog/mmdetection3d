"""
Main multi-metric comparison runner

Compares two architecturally identical models across all three metrics:
- CKA: Representational similarity (layer activations)
- W2: BatchNorm statistics divergence (running mean/variance)
- Effective Scale/Bias: BN affine parameter similarity (γ/β)

This generates comprehensive heatmaps for all metrics side-by-side.

Usage:
    python main.py --config <config> --pcd_dir <dir> --pt <model1> <model2> \
                   --labels <label1> <label2> --max_samples 100 --output_dir ./results
"""

import sys
import os
import glob
import argparse
import numpy as np

INTERNAL_PATH = '/opt/src/mmdetection3d'
if INTERNAL_PATH not in sys.path:
    sys.path.insert(0, INTERNAL_PATH)

from metrics import CKAMetric, W2BNStatsMetric, EffectiveScaleBiasMetric
from comparison_engine import ModelComparator
from comparison_utils import default_hook_filter
from visualization import ComparisonVisualizer, create_comparison_report


def main():
    parser = argparse.ArgumentParser(
        description="Multi-metric comparison of two architecturally identical models"
    )
    parser.add_argument('--config', type=str, required=True, help="Model config file")
    parser.add_argument('--pcd_dir', type=str, required=True, help="Directory with .bin files")
    parser.add_argument('--pt', nargs='+', required=True, help="Two model checkpoints")
    parser.add_argument('--labels', nargs='+', required=True, help="Two model labels")
    parser.add_argument('--max_samples', type=int, default=100, help="Max samples to process")
    parser.add_argument('--output_dir', type=str, default=".", help="Output directory")
    
    args = parser.parse_args()
    
    if len(args.pt) != 2 or len(args.labels) != 2:
        print("ERROR: Comparison requires exactly 2 models and 2 labels")
        return
    
    # Gather PCD files
    pcd_files = sorted(glob.glob(os.path.join(args.pcd_dir, "*.bin")))
    if len(pcd_files) == 0:
        print(f"ERROR: No .bin files found in {args.pcd_dir}")
        return
    
    if args.max_samples > 0:
        pcd_files = pcd_files[:args.max_samples]
    
    print("=" * 80)
    print("MULTI-METRIC MODEL COMPARISON")
    print("=" * 80)
    print(f"Samples: {len(pcd_files)}")
    print(f"Model A: {args.labels[0]}")
    print(f"Model B: {args.labels[1]}")
    print(f"Metrics: CKA, W2 BN Stats, Effective Scale/Bias")
    print("=" * 80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run all three metrics
    comparator = ModelComparator(args.config, pcd_files,
                                  os.path.join(args.output_dir, "temp"))
    
    metrics = [
        CKAMetric(),
        W2BNStatsMetric(),
        EffectiveScaleBiasMetric()
    ]
    
    print("\n[Comparisons]")
    results = comparator.run_full_comparison(
        args.pt[0], args.pt[1],
        metrics,
        hook_filter=default_hook_filter
    )
    
    # Prepare visualization data
    comp_title = f"{args.labels[0]}_vs_{args.labels[1]}"
    comparisons = [(comp_title, args.labels[0], args.labels[1])]
    
    viz_results = {comp_title: results}
    metric_names = ['CKA', 'W2 BN Stats', 'Effective Scale/Bias']
    
    # Generate visualizations
    print("\n[Visualization]")
    visualizer = ComparisonVisualizer()
    output_files = visualizer.plot_all_metrics(
        viz_results, comparisons, metric_names,
        output_dir=args.output_dir
    )
    
    # Generate report
    report_path = create_comparison_report(
        viz_results, comparisons, metric_names,
        len(pcd_files), args.output_dir
    )
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    for metric_name in metric_names:
        if metric_name in results:
            scores = results[metric_name]
            print(f"\n{metric_name}:")
            print(f"  Layers compared: {len(scores)}")
            print(f"  Mean score:      {np.mean(list(scores.values())):.4f}")
            print(f"  Std dev:         {np.std(list(scores.values())):.4f}")
            print(f"  Min score:       {np.min(list(scores.values())):.4f}")
            print(f"  Max score:       {np.max(list(scores.values())):.4f}")
    
    print("\n" + "=" * 80)
    print(f"Report saved: {report_path}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
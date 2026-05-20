"""
Individual metric runner: CKA (Centered Kernel Alignment)

Compares layer activations between two models using CKA.
Focuses on representational similarity during forward pass.

Usage:
    python run_cka.py --config <config> --pcd_dir <dir> --pt <model1> <model2> \
                      --labels <label1> <label2> --max_samples 100 --output_dir ./results
"""

import sys
import os
import glob
import argparse

INTERNAL_PATH = '/opt/src/mmdetection3d'
if INTERNAL_PATH not in sys.path:
    sys.path.insert(0, INTERNAL_PATH)

from metrics import CKAMetric
from comparison_engine import ModelComparator
from comparison_utils import default_hook_filter, sort_layers_by_network_order
from visualization import ComparisonVisualizer, create_comparison_report


def main():
    parser = argparse.ArgumentParser(
        description="Compare models using CKA (layer activation similarity)"
    )
    parser.add_argument('--config', type=str, required=True, help="Model config file")
    parser.add_argument('--pcd_dir', type=str, required=True, help="Directory with .bin files")
    parser.add_argument('--pt', nargs='+', required=True, help="Two model checkpoints")
    parser.add_argument('--labels', nargs='+', required=True, help="Two model labels")
    parser.add_argument('--max_samples', type=int, default=100, help="Max samples to process")
    parser.add_argument('--output_dir', type=str, default=".", help="Output directory")
    
    args = parser.parse_args()
    
    if len(args.pt) != 2 or len(args.labels) != 2:
        print("ERROR: CKA comparison requires exactly 2 models and 2 labels")
        return
    
    # Gather PCD files
    pcd_files = sorted(glob.glob(os.path.join(args.pcd_dir, "*.bin")))
    if len(pcd_files) == 0:
        print(f"ERROR: No .bin files found in {args.pcd_dir}")
        return
    
    if args.max_samples > 0:
        pcd_files = pcd_files[:args.max_samples]
    
    print(f"[CKA Comparison]")
    print(f"Processing {len(pcd_files)} samples")
    print(f"Model A: {args.labels[0]} ({args.pt[0]})")
    print(f"Model B: {args.labels[1]} ({args.pt[1]})")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run comparison
    comparator = ModelComparator(args.config, pcd_files, 
                                  os.path.join(args.output_dir, "temp"))
    
    metric = CKAMetric()
    results = comparator.run_full_comparison(
        args.pt[0], args.pt[1],
        [metric],
        hook_filter=default_hook_filter
    )
    
    # Visualize
    print("\n[Visualization]")
    comp_title = f"{args.labels[0]}_vs_{args.labels[1]}"
    comparisons = [(comp_title, args.labels[0], args.labels[1])]
    
    viz_results = {comp_title: results}
    
    visualizer = ComparisonVisualizer()
    output_files = visualizer.plot_all_metrics(
        viz_results, comparisons, ['CKA'],
        output_dir=args.output_dir
    )
    
    report_path = create_comparison_report(
        viz_results, comparisons, ['CKA'],
        len(pcd_files), args.output_dir
    )
    
    print(f"\nReport saved: {report_path}")
    print(f"[DONE] Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
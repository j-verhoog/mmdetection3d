import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from nuscenes.nuscenes import NuScenes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run two-model comparison on all samples in a nuscenes scene"
    )
    
    # Core model and config args
    parser.add_argument(
        "config",
        nargs="?",
        default="projects/cmt/fed/fedSelect/improved_lightweight_cmt_iterated_FedSelect.py",
        help="train config file path",
    )
    parser.add_argument(
        "checkpoint_a",
        nargs="?",
        default="/home/jolle/Desktop/thesisqualitative/fedavg_merged_E.pth",
        help="first checkpoint file (.pth)",
    )
    parser.add_argument(
        "checkpoint_b",
        nargs="?",
        default="/home/jolle/Desktop/thesisqualitative/fedcka_merged_E.pth",
        help="second checkpoint file (.pth)",
    )
    
    # Scene/image selection
    parser.add_argument(
        "--image-name",
        required=True,
        help="Image filename or full path. Will extract filename and search for it. Finds the scene containing this image.",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root output directory. Will create scene_token subdir with camera subfolders.",
    )
    
    # NuScenes data
    parser.add_argument(
        "--nuscenes-dir",
        default="/home/jolle/mmdet/datasets/nuscenes",
        help="Path to nuscenes dataset root",
    )
    
    parser.add_argument(
        "--nuscenes-version",
        default="v1.0-trainval",
        help="NuScenes version (v1.0-mini, v1.0-trainval, etc.)",
    )
    
    # Remote Data Settings
    parser.add_argument(
        "--remote-nuscenes",
        default="daic:/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets/nuscenes",
        help="Remote SSH path to pull missing files from via rsync",
    )
    
    # Visualization options passed to run_two_models.py
    parser.add_argument("--camera", default="CAM_FRONT", help="Camera name to visualize")
    parser.add_argument("--score-thr", type=float, default=0.35, help="Bounding box score threshold")
    parser.add_argument("--line-thickness", type=int, default=2)
    
    parser.add_argument(
        "--draw-space",
        choices=["processed", "padded", "raw"],
        default="padded",
        help="Space to draw bounding boxes in",
    )
    
    parser.add_argument(
        "--projection-mode",
        choices=["auto", "lidar2img", "cam2img", "lidar2img_with_img_aug", "img_aug_with_lidar2img"],
        default="auto",
        help="Projection mode for 3D to 2D projection",
    )
    
    parser.add_argument("--z-shift", type=float, default=0.0, help="Debug z-shift for boxes")
    parser.add_argument("--font-scale", type=float, default=0.5, help="Font scale for labels")
    parser.add_argument("--lidar-range", type=float, default=50.0, help="LiDAR range for BEV")
    
    parser.add_argument("--draw-class-names", action="store_true", help="Draw class names on boxes")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    
    # BEV labels
    parser.add_argument("--bev-gt-label", type=str, default="Ground Truth", help="Label for GT in BEV")
    parser.add_argument("--bev-model-a-label", type=str, default="FedAvg", help="Label for Model A in BEV")
    parser.add_argument("--bev-model-b-label", type=str, default="FedCKA", help="Label for Model B in BEV")
    
    return parser.parse_args()


def sync_scene_dependencies(nusc: NuScenes, samples: list, local_root: str, remote_root: str) -> bool:
    """Pre-fetches all 6 cameras and 10 LiDAR sweeps for all samples in the scene."""
    print(f"\n[INFO] Compiling dependency list for {len(samples)} samples (Cameras + 10 LiDAR sweeps)...")
    required_files = set()
    
    for sample in samples:
        # 1. Grab all Camera images
        for key, token in sample["data"].items():
            if "CAM" in key:
                cam_data = nusc.get("sample_data", token)
                required_files.add(cam_data["filename"])
                
        # 2. Grab LiDAR keyframe + 10 previous sweeps
        if "LIDAR_TOP" in sample["data"]:
            curr_token = sample["data"]["LIDAR_TOP"]
            sweep_count = 0
            # mmdet3d typically uses 10 sweeps + the keyframe (11 total)
            while curr_token and sweep_count <= 10:
                sd = nusc.get("sample_data", curr_token)
                required_files.add(sd["filename"])
                curr_token = sd["prev"]
                sweep_count += 1
                
    # Check local disk
    missing_files = []
    for rel_path in required_files:
        local_full = os.path.join(local_root, rel_path)
        if not os.path.exists(local_full):
            missing_files.append(rel_path)
            
    if not missing_files:
        print("[INFO] All required files are already present locally.")
        return True
        
    print(f"[WARN] Found {len(missing_files)} missing files. Initiating batch sync from DAIC...")
    list_file = "temp_rsync_missing.txt"
    with open(list_file, "w") as f:
        for mf in missing_files:
            f.write(f"{mf}\n")
            
    # Ensure proper trailing slashes for rsync directory mapping
    remote_path = remote_root if remote_root.endswith("/") else remote_root + "/"
    local_path = local_root if local_root.endswith("/") else local_root + "/"
    
    cmd = [
        "rsync", "-avz", "--progress", 
        f"--files-from={list_file}", 
        remote_path, 
        local_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        os.remove(list_file) # Clean up temp file on success
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Rsync batch failed: {e}")
        return False
        
    # Final hard verification
    still_missing = [mf for mf in missing_files if not os.path.exists(os.path.join(local_root, mf))]
    if still_missing:
        print(f"[FATAL] {len(still_missing)} files are still missing after rsync.")
        return False
        
    print("[INFO] All missing files successfully synced.")
    return True


def find_scene_from_image(nusc: NuScenes, image_name: str) -> Optional[str]:
    """Find scene_token from an image filename."""
    search_name = os.path.basename(image_name)
    print(f"[INFO] Searching for scene containing image: {search_name}")
    
    for sample in nusc.sample:
        for camera_key in sample["data"]:
            if "CAM" in camera_key:
                cam_token = sample["data"][camera_key]
                cam_data = nusc.get("sample_data", cam_token)
                
                if search_name in cam_data["filename"]:
                    scene_token = sample["scene_token"]
                    print(f"[INFO] Found image in scene: {scene_token}")
                    return scene_token
    
    print(f"[ERROR] Image {search_name} not found in any scene")
    return None


def get_all_samples_in_scene(nusc: NuScenes, scene_token: str) -> list:
    """Get all samples in a scene in chronological order."""
    scene = nusc.get("scene", scene_token)
    samples = []
    
    sample_token = scene["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)
        samples.append(sample)
        sample_token = sample["next"]
    
    print(f"[INFO] Scene has {len(samples)} samples")
    return samples


def get_images_for_sample(nusc: NuScenes, sample: dict, target_camera: str) -> list:
    """Get all image filenames for a sample. Returns list of (camera_name, image_filename)"""
    images = []
    for camera_key in sample["data"]:
        if "CAM" not in camera_key:
            continue
        cam_token = sample["data"][camera_key]
        cam_data = nusc.get("sample_data", cam_token)
        image_filename = cam_data["filename"]
        images.append((camera_key, image_filename))
    return images


def create_output_structure(output_dir: str, scene_token: str) -> dict:
    """Create output directory structure and return paths."""
    scene_dir = os.path.join(output_dir, scene_token)
    lidar_cam_dir = os.path.join(scene_dir, "lidar_camera")
    os.makedirs(lidar_cam_dir, exist_ok=True)
    return {
        "scene_dir": scene_dir,
        "lidar_cam_dir": lidar_cam_dir,
    }


def run_comparison_for_sample(
    sample_idx: int, sample_token: str, image_filename: str,
    camera_name: str, config: str, checkpoint_a: str, checkpoint_b: str,
    output_dir: str, args
):
    """Run run_two_models.py for a single image."""
    image_basename = os.path.splitext(os.path.basename(image_filename))[0]
    output_base = os.path.join(output_dir, f"{image_basename}_{camera_name}")
    
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "run_two_models.py"),
        config, checkpoint_a, checkpoint_b,
        "--target-img", os.path.basename(image_filename),
        "--out", output_base + ".jpg",
        "--camera", camera_name,
        "--score-thr", str(args.score_thr),
        "--line-thickness", str(args.line_thickness),
        "--draw-space", args.draw_space,
        "--projection-mode", args.projection_mode,
        "--z-shift", str(args.z_shift),
        "--font-scale", str(args.font_scale),
        "--lidar-range", str(args.lidar_range),
        "--bev-gt-label", args.bev_gt_label,
        "--bev-model-a-label", args.bev_model_a_label,
        "--bev-model-b-label", args.bev_model_b_label,
    ]
    
    if args.draw_class_names: cmd.append("--draw-class-names")
    if args.debug: cmd.append("--debug")
    
    print(f"\n[INFO] Processing: {camera_name} - {image_basename}")
    
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"[INFO] Successfully processed {camera_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to process {camera_name}: {e}")
        return False


def main():
    args = parse_args()
    
    if not os.path.isfile(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)
    if not os.path.isfile(args.checkpoint_a):
        print(f"[ERROR] Checkpoint A not found: {args.checkpoint_a}")
        sys.exit(1)
    if not os.path.isfile(args.checkpoint_b):
        print(f"[ERROR] Checkpoint B not found: {args.checkpoint_b}")
        sys.exit(1)
    
    print(f"[INFO] Loading NuScenes from {args.nuscenes_dir} ({args.nuscenes_version})")
    nusc = NuScenes(
        version=args.nuscenes_version,
        dataroot=args.nuscenes_dir,
        verbose=False,
    )
    
    scene_token = find_scene_from_image(nusc, args.image_name)
    if scene_token is None:
        sys.exit(1)
    
    samples = get_all_samples_in_scene(nusc, scene_token)
    
    # --- PRE-FETCH CRITICAL DEPENDENCIES ---
    if not sync_scene_dependencies(nusc, samples, args.nuscenes_dir, args.remote_nuscenes):
        print("[FATAL] Dependency sync failed. Halting execution to prevent cascade errors.")
        sys.exit(1)
    
    output_paths = create_output_structure(args.output_dir, scene_token)
    output_dir = output_paths["lidar_cam_dir"]
    
    print(f"\n[INFO] Processing {len(samples)} samples in scene {scene_token}...\n")
    
    processed_count = 0
    success_count = 0
    
    for sample_idx, sample in enumerate(samples):
        images = get_images_for_sample(nusc, sample, args.camera)
        for camera_name, image_filename in images:
            if camera_name == args.camera:
                processed_count += 1
                success = run_comparison_for_sample(
                    sample_idx=sample_idx,
                    sample_token=sample["token"],
                    image_filename=image_filename,
                    camera_name=camera_name,
                    config=args.config,
                    checkpoint_a=args.checkpoint_a,
                    checkpoint_b=args.checkpoint_b,
                    output_dir=output_dir,
                    args=args,
                )
                if success:
                    success_count += 1
                    
                else:
                    # Halt if model inference crashes
                    print(f"[FATAL] run_two_models.py failed on {image_filename}. Halting.")
                    sys.exit(1)
    
    print(f"\n" + "="*60)
    print(f"[INFO] PROCESSING COMPLETE")
    print(f"[INFO] Scene: {scene_token}")
    print(f"[INFO] Total samples processed: {processed_count}")
    print(f"[INFO] Successfully processed: {success_count}")
    print(f"[INFO] Failed: {processed_count - success_count}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"="*60 + "\n")


if __name__ == "__main__":
    main()
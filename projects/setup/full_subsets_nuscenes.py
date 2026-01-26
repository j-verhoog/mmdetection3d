#!/usr/bin/env python3
"""
NuScenes Robust Subset Creator (Metadata Filtering + Symlinking)
------------------------------------------------------------------
VERIFIED FOR MMDETECTION3D V1.x
------------------------------------------------------------------
1. Reads the Domain Summary Excel.
2. For each subset:
   a. Identifies all related Scenes.
   b. Traverses Scene -> Sample -> SampleData (including PREV sweeps).
   c. Filters the JSON metadata tables to include ONLY these items.
   d. Writes clean JSONs to subset_dir/v1.0-trainval.
   e. Symlinks only the relevant sensor files.
   f. Copies full maps (required for SDK).
"""

import os
import json
import shutil
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

# ================= CONFIGURATION =================
# EXACT PATHS PROVIDED BY USER
NUSC_SOURCE_ROOT = Path("/home/jolle/mmdet/nuscenes_shadow_root")
EXCEL_PATH = Path("/home/jolle/mmdet/mmdetection3d/projects/setup/scene_domains_summary.xlsx")
OUT_ROOT = Path("/home/jolle/mmdet/nuscenes_subsets_full")

# Options
COPY_METHOD = "symlink" # not used, but is symlinks now by default
OVERWRITE = False  # Overwrite existing files in output

# Mapping Configuration
SUBSET_MAPPING = {
    ("Boston", "day_rain"): "boston_day_rain",
    ("Boston", "day_clear"): "boston_day_clear",
    ("Singapore", "day_clear"): "singapore_day_clear",
    ("Singapore", "night_clear"): "singapore_night_clear",
}
# =================================================

# Setup Logging
OUT_ROOT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUT_ROOT / "subset_creation.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================= HELPER FUNCTIONS =================

def load_scene_map(file_path: Path) -> Dict[str, str]:
    """Reads the Excel and maps scene_token -> subset_name."""
    logger.info(f"Loading scene map from {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Explicitly use openpyxl for xlsx
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        logger.error("Failed to read Excel. Is openpyxl installed? (pip install openpyxl)")
        raise e

    # Normalize columns (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]
    
    # Check for required columns
    if 'scene_token' not in df.columns or 'city' not in df.columns or 'combo' not in df.columns:
        # Fallback: Check if headers are on the second row or similar issues? 
        # But assuming standard format:
        raise ValueError(f"Excel missing columns. Found: {df.columns.tolist()}")

    scene_map = {}
    stats = {k: 0 for k in SUBSET_MAPPING.values()}

    for _, row in df.iterrows():
        key = (row['city'], row['combo'])
        subset = SUBSET_MAPPING.get(key)
        if subset:
            scene_map[row['scene_token']] = subset
            stats[subset] += 1
            
    logger.info(f"Planned distribution: {stats}")
    return scene_map

def load_json_table(root: Path, table_name: str) -> List[Dict]:
    p = root / "v1.0-trainval" / f"{table_name}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing metadata table: {p}")
    with open(p, 'r') as f:
        return json.load(f)

def save_json_table(data: List[Dict], root: Path, table_name: str):
    out_dir = root / "v1.0-trainval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{table_name}.json", 'w') as f:
        json.dump(data, f, indent=0)

def safe_symlink(src: Path, dst: Path):
    """Creates a RELATIVE symlink so datasets are portable."""
    if not src.exists(): return # Skip missing files

    if dst.exists() or dst.is_symlink():
        if not OVERWRITE: return
        try:
            if dst.is_dir(): shutil.rmtree(dst)
            else: dst.unlink()
        except OSError: pass
    
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    # CALCULATE RELATIVE PATH
    try:
        # e.g. ../../../nuscenes_shadow_root/samples/LIDAR/x.bin
        rel_src = os.path.relpath(src, dst.parent)
        os.symlink(rel_src, dst)
    except Exception as e:
        logger.error(f"Symlink failed {src} -> {dst}: {e}")

# ================= MAIN LOGIC =================

def main():
    logger.info("Starting Verified NuScenes Subset Creator...")
    
    # 1. Load Map
    scene_map = load_scene_map(EXCEL_PATH)
    
    # 2. Initialize NuScenes (Required for graph traversal)
    logger.info("Initializing NuScenes DB (this takes a minute)...")
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(NUSC_SOURCE_ROOT), verbose=True)

    # 3. Load Raw Metadata Tables (We will filter these later)
    logger.info("Loading raw JSON tables into memory...")
    raw_tables = {}
    # Tables we need to filter:
    for t in ["scene", "sample", "sample_data", "sample_annotation", "instance", 
              "log", "ego_pose", "calibrated_sensor"]:
        raw_tables[t] = load_json_table(NUSC_SOURCE_ROOT, t)
    # Tables we keep full (Static data - small and harmless to keep):
    for t in ["map", "sensor", "category", "attribute", "visibility"]:
        raw_tables[t] = load_json_table(NUSC_SOURCE_ROOT, t)

    # 4. Group Scenes by Subset
    subsets: Dict[str, List[str]] = {name: [] for name in SUBSET_MAPPING.values()}
    for token, subset in scene_map.items():
        subsets[subset].append(token)

    # 5. Process Each Subset
    for subset_name, scene_tokens in subsets.items():
        if not scene_tokens:
            logger.warning(f"Skipping empty subset: {subset_name}")
            continue
            
        logger.info(f"=== Building Subset: {subset_name} ({len(scene_tokens)} scenes) ===")
        subset_root = OUT_ROOT / subset_name
        subset_root.mkdir(parents=True, exist_ok=True)

        # --- A. Identification Phase (Graph Traversal) ---
        # We find exactly which tokens (samples, sweeps, etc) belong to these scenes.
        
        target_scene_tokens = set(scene_tokens)
        keep_samples = set()
        keep_sample_data = set() # Images/Lidar (Keyframes AND Sweeps)
        keep_instances = set()
        
        # Traverse via NuScenes API (Indices are already built)
        # This is faster and safer than manual recursion
        for scene_token in tqdm(scene_tokens, desc=f"Analyzing {subset_name} graph"):
            try:
                scene = nusc.get('scene', scene_token)
            except KeyError:
                logger.warning(f"Scene {scene_token} in Excel but not in DB. Skipping.")
                continue

            # Walk through Samples
            curr_sample_token = scene['first_sample_token']
            while curr_sample_token:
                keep_samples.add(curr_sample_token)
                sample = nusc.get('sample', curr_sample_token)
                
                # Walk through Sample Data (Sensors)
                for sensor, sd_token in sample['data'].items():
                    # 1. Keep the Keyframe
                    keep_sample_data.add(sd_token)
                    
                    # 2. Walk BACKWARDS for Sweeps (Crucial for mmdet3d Lidar)
                    # We walk back until we hit the previous keyframe
                    sd = nusc.get('sample_data', sd_token)
                    curr_sd_prev = sd['prev']
                    
                    while curr_sd_prev:
                        prev_sd = nusc.get('sample_data', curr_sd_prev)
                        if prev_sd['is_key_frame']:
                            break # Stop, we reached the previous keyframe
                        
                        keep_sample_data.add(curr_sd_prev)
                        curr_sd_prev = prev_sd['prev']
                
                # Collect Instance IDs from Annotations
                for ann_token in sample['anns']:
                    ann = nusc.get('sample_annotation', ann_token)
                    keep_instances.add(ann['instance_token'])

                curr_sample_token = sample['next']

        # --- B. Filtering Phase ---
        logger.info("Filtering metadata tables...")
        
        new_tables = {}
        
        # Filter primary tables based on the sets we built above
        new_tables['scene'] = [s for s in raw_tables['scene'] if s['token'] in target_scene_tokens]
        new_tables['sample'] = [s for s in raw_tables['sample'] if s['token'] in keep_samples]
        new_tables['sample_data'] = [sd for sd in raw_tables['sample_data'] if sd['token'] in keep_sample_data]
        new_tables['sample_annotation'] = [a for a in raw_tables['sample_annotation'] if a['sample_token'] in keep_samples]
        new_tables['instance'] = [i for i in raw_tables['instance'] if i['token'] in keep_instances]
        
        # Filter dependent tables
        keep_logs = set(s['log_token'] for s in new_tables['scene'])
        new_tables['log'] = [l for l in raw_tables['log'] if l['token'] in keep_logs]
        
        keep_ego = set(sd['ego_pose_token'] for sd in new_tables['sample_data'])
        new_tables['ego_pose'] = [ep for ep in raw_tables['ego_pose'] if ep['token'] in keep_ego]
        
        keep_cs = set(sd['calibrated_sensor_token'] for sd in new_tables['sample_data'])
        new_tables['calibrated_sensor'] = [cs for cs in raw_tables['calibrated_sensor'] if cs['token'] in keep_cs]
        
        # Keep static tables full
        for t in ["map", "sensor", "category", "attribute", "visibility"]:
            new_tables[t] = raw_tables[t]

        # --- C. Writing Phase ---
        for table_name, data in new_tables.items():
            save_json_table(data, subset_root, table_name)
            
        # --- D. Symlinking Phase ---
        # Only symlink files referenced in our FILTERED sample_data
        # This ensures 100% consistency between JSON and Filesystem
        logger.info("Symlinking sensor files (Images/Lidar/Radar)...")
        for sd in tqdm(new_tables['sample_data'], desc="Linking Files"):
            filename = sd['filename'] # e.g. samples/CAM_FRONT/xxx.jpg
            src_path = NUSC_SOURCE_ROOT / filename
            dst_path = subset_root / filename
            safe_symlink(src_path, dst_path)
            
        # --- E. Map Copying Phase ---
        logger.info("Copying maps...")
        src_maps = NUSC_SOURCE_ROOT / "maps"
        dst_maps = subset_root / "maps"
        if src_maps.exists() and (not dst_maps.exists() or OVERWRITE):
             shutil.copytree(src_maps, dst_maps, dirs_exist_ok=True)

    logger.info("-" * 30)
    logger.info("SUCCESS. Subsets created.")
    logger.info(f"Log: {OUT_ROOT / 'subset_creation.log'}")

if __name__ == "__main__":
    main()
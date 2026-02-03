#!/usr/bin/env python3
"""
NuScenes Robust Subset Creator (Metadata Filtering + Symlinking)
------------------------------------------------------------------
VERIFIED FOR MMDETECTION3D V1.x WITH TEMPORAL CONSISTENCY
------------------------------------------------------------------
Research-Grade Features:
1. Multi-configuration support (run multiple experiments in one go)
2. Temporal grouping (sequential vs interleaved client distribution)
3. Fair validation set handling (never dropped or subsampled)
4. Sample drop strategies (TAIL temporal vs RANDOM)
5. Fairness balancing (MIN-based, not MAX)
6. Data integrity (heals broken linked lists)
7. Robust symlinking (handles different mount points)

For each configuration:
   a. Separates train/val scenes from Excel.
   b. Applies fairness filtering to train samples only.
   c. Distributes train samples to clients (temporal or interleaved).
   d. Keeps 100% of validation samples for all clients.
   e. Traverses Scene -> Sample -> SampleData (including PREV sweeps).
   f. Filters JSON metadata tables to include ONLY assigned items.
   g. Heals broken sample_data pointers (prev/next).
   h. Writes clean JSONs to client_dir/v1.0-trainval.
   i. Symlinks only the relevant sensor files.
   j. Copies full maps (required for SDK).
"""

import os
import json
import shutil
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
import random

# ================= GLOBAL CONFIGURATION =================
# Paths (same for all configurations)
NUSC_SOURCE_ROOT = Path("/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets/nuscenes")
EXCEL_PATH = Path("/home/nfs/jtverhoog/mmdet/mmdetection3d/projects/subsets_creation/scene_domains_summary.xlsx")
BASE_OUT_ROOT = Path("/tudelft.net/staff-umbrella/MscThesisjverhoog/nusc_datasets")

# Global options
COPY_METHOD = "symlink"  # not used, but symlinks now by default
OVERWRITE = True  # Overwrite existing files in output
DEFAULT_RANDOM_SEED = 2026

# Domain mapping
DOMAIN_MAPPING = {
    ("Boston", "day_rain"): "boston_day_rain",
    ("Boston", "day_clear"): "boston_day_clear",
    ("Singapore", "day_clear"): "singapore_day_clear",
    ("Singapore", "night_clear"): "singapore_night_clear",
    ("Singapore", "night_rain"): "singapore_night_rain",
}

# ================= EXPERIMENT CONFIGURATIONS =================
# Each configuration creates a different dataset variant
# These can be run sequentially to generate multiple experimental conditions
CONFIGURATIONS = [
    {
        "name": "Default_NoFair_SingleClient",
        "temporal_grouping": False,  # False = Interleaved (stride), True = Sequential (chunks)
        "drop_strategy": "RANDOM",   # 'TAIL' = take first N, 'RANDOM' = random sample
        "fairness_mode": False,      # False, 'TOTAL', 'COMPARATIVE'
        "clients_per_domain": 1,     # Number of client subsets per domain
        "random_seed": DEFAULT_RANDOM_SEED,
    },
    {
        "name": "Exp1_CompFair_SingleClient",
        "temporal_grouping": False,
        "drop_strategy": "RANDOM",
        "fairness_mode": "COMPARATIVE",
        "clients_per_domain": 1,
        "random_seed": DEFAULT_RANDOM_SEED,
    },
    {
        "name": "Exp2_TotalFair_SingleClient",
        "temporal_grouping": False,
        "drop_strategy": "RANDOM",
        "fairness_mode": "TOTAL",
        "clients_per_domain": 1,
        "random_seed": DEFAULT_RANDOM_SEED,
    },
    {
        "name": "Exp3_CompFair_DualClient",
        "temporal_grouping": False,
        "drop_strategy": "RANDOM",
        "fairness_mode": "COMPARATIVE",
        "clients_per_domain": 2,
        "random_seed": DEFAULT_RANDOM_SEED,
    },
]


# =================================================
# LOGGING SETUP (Dual Logger System)
# =================================================
BASE_OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Main logger (summary of important events)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(BASE_OUT_ROOT / "subset_creation.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Detailed logger (extensive debugging information)
detailed_logger = logging.getLogger('detailed')
detailed_logger.setLevel(logging.CRITICAL)                         # 'DEBUG' or 'CRITICAL' to surpress most logs
detailed_handler = logging.FileHandler(BASE_OUT_ROOT / "subset_creation_detailed.log", mode='w')
detailed_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d: %(message)s"
))
detailed_logger.addHandler(detailed_handler)

# ================= HELPER FUNCTIONS =================

def load_scene_map(file_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Reads the Excel and returns two mappings:
    - scene_token -> subset_name (e.g., 'boston_day_rain')
    - scene_token -> split (e.g., 'train' or 'val')
    
    Excel must have columns: 'scene_token', 'city', 'combo', 'split'
    """
    logger.info(f"Loading scene map from {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        logger.error("Failed to read Excel. Is openpyxl installed? (pip install openpyxl)")
        raise e

    # Normalize columns (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]
    
    # Check for required columns
    required = {'scene_token', 'city', 'combo', 'split'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Excel missing required columns {required}. Found: {df.columns.tolist()}")

    scene_to_subset = {}
    scene_to_split = {}
    stats = {k: {"train": 0, "val": 0} for k in DOMAIN_MAPPING.values()}

    for _, row in df.iterrows():
        key = (row['city'], row['combo'])
        subset = DOMAIN_MAPPING.get(key)
        if not subset:
            continue
            
        scene_token = row['scene_token']
        split_val = str(row['split']).strip().lower()  # 'train' or 'val'
        
        if split_val not in ('train', 'val'):
            logger.warning(f"Scene {scene_token}: invalid split '{split_val}', defaulting to 'train'")
            split_val = 'train'
        
        scene_to_subset[scene_token] = subset
        scene_to_split[scene_token] = split_val
        stats[subset][split_val] += 1
            
    logger.info(f"Scene distribution: {json.dumps(stats, indent=2)}")
    return scene_to_subset, scene_to_split

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
    """
    Creates a symlink, preferring relative paths for portability.
    Falls back to absolute symlinks if paths are on different mount points.
    Logs warnings for missing source files.
    
    CRITICAL FIX: Properly handles symlinks to directories by checking
    is_symlink() BEFORE is_dir(), since a symlink to a directory returns
    True for is_dir() but cannot be removed with rmtree().
    """
    if not src.exists():
        logger.warning(f"SOURCE FILE MISSING: {src} (cannot create symlink to {dst})")
        detailed_logger.debug(f"Missing source file: {src}")
        return  # Skip missing files

    if dst.exists() or dst.is_symlink():
        if not OVERWRITE:
            return
        try:
            # CRITICAL: Check is_symlink() BEFORE is_dir()
            # If dst is a symlink to a directory, is_dir() returns True
            # but rmtree() will fail on it
            if dst.is_symlink():
                # It's a symlink, remove it regardless of target type
                dst.unlink()
                detailed_logger.debug(f"Removed symlink: {dst}")
            elif dst.is_dir():
                # It's an actual directory, use rmtree
                shutil.rmtree(dst)
                detailed_logger.debug(f"Removed directory: {dst}")
            else:
                # It's a file, unlink it
                dst.unlink()
                detailed_logger.debug(f"Removed file: {dst}")
        except OSError as e:
            logger.warning(f"Failed to remove existing {dst}: {e}")
            detailed_logger.debug(f"OSError removing {dst}: {e}")
    
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try relative path first (most portable)
        rel_src = os.path.relpath(src, dst.parent)
        os.symlink(src.resolve(), dst)
        detailed_logger.debug(f"Symlink created (absolute): {src.resolve()} -> {dst}")
    except ValueError:
        # ValueError: paths on different drives/mount points
        # Fallback to absolute symlink
        try:
            os.symlink(src.resolve(), dst)
            detailed_logger.debug(f"Symlink created (absolute): {src.resolve()} -> {dst}")
            logger.debug(f"Used absolute symlink for {src} (different mount point)")
        except Exception as e:
            logger.error(f"SYMLINK FAILED: {src} -> {dst}: {e}")
            detailed_logger.error(f"Symlink exception: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"SYMLINK FAILED: {src} -> {dst}: {e}")
        detailed_logger.error(f"Symlink exception: {e}", exc_info=True)

# ================= FAIRNESS & CLIENT DISTRIBUTION FUNCTIONS =================

def get_fairness_filtered_samples(
    subset_name: str,
    scene_tokens: List[str],
    nusc: NuScenes,
    fairness_mode: str = False,
    drop_strategy: str = "TAIL",
    random_seed: int = DEFAULT_RANDOM_SEED
) -> Tuple[List[str], int]:
    """
    Identifies which training samples to include based on fairness mode.
    Uses MIN-based capping to ensure fair comparison between domains.
    
    CRITICAL CHANGE: Collects ALL samples first, then applies drop strategy.
    This ensures DROP_STRATEGY works on the entire dataset, not just the head.
    
    Args:
        subset_name: Domain name (e.g., 'boston_day_rain')
        scene_tokens: List of scene tokens for this domain
        nusc: NuScenes instance
        fairness_mode: False (no filtering), 'TOTAL', or 'COMPARATIVE'
        drop_strategy: 'TAIL' (keep first N) or 'RANDOM' (random sample)
        random_seed: Seed for reproducible random sampling
    
    Returns:
        Tuple of (sample_tokens_list, limit_used)
        - sample_tokens_list: List of samples to keep (ordered)
        - limit_used: The fairness limit that was applied
    
    Workflow:
        1. Collect ALL valid samples for this domain
        2. Calculate fairness limit (if fairness_mode enabled)
        3. Apply drop_strategy to select which samples to keep
        4. Return filtered list
    """
    detailed_logger.info(f"get_fairness_filtered_samples: {subset_name}, fairness={fairness_mode}, drop_strategy={drop_strategy}")
    
    # ===== STEP 1: COLLECT ALL VALID SAMPLES =====
    detailed_logger.debug(f"Collecting all samples for {len(scene_tokens)} scenes...")
    all_valid_samples = []
    scenes_processed = 0
    
    for scene_token in scene_tokens:
        try:
            scene = nusc.get('scene', scene_token)
            scenes_processed += 1
        except KeyError:
            logger.warning(f"Scene {scene_token} not found in DB")
            detailed_logger.debug(f"Scene not found: {scene_token}")
            continue
        
        curr_sample_token = scene['first_sample_token']
        sample_count_in_scene = 0
        while curr_sample_token:
            all_valid_samples.append(curr_sample_token)
            sample_count_in_scene += 1
            try:
                sample = nusc.get('sample', curr_sample_token)
                curr_sample_token = sample['next']
            except KeyError as e:
                logger.error(f"Sample chain broken at {curr_sample_token}: {e}")
                detailed_logger.error(f"Sample chain broken: {curr_sample_token}", exc_info=True)
                break
        
        detailed_logger.debug(f"Scene {scene_token[:8]}: {sample_count_in_scene} samples")
    
    logger.info(f"Collected {len(all_valid_samples)} samples from {scenes_processed}/{len(scene_tokens)} scenes for {subset_name}")
    detailed_logger.debug(f"Total samples collected: {len(all_valid_samples)}")
    
    if not all_valid_samples:
        logger.warning(f"No valid samples found for {subset_name}")
        return [], 0
    
    # ===== STEP 2: CALCULATE FAIRNESS LIMIT =====
    if not fairness_mode:
        limit = len(all_valid_samples)
        logger.debug(f"{subset_name}: No fairness, using all {limit} samples")
    else:
        SINGAPORE_DOMAINS = {'singapore_day_clear', 'singapore_night_clear'}
        BOSTON_DOMAINS = {'boston_day_clear', 'boston_day_rain'}
        ALL_COMPARISON_DOMAINS = SINGAPORE_DOMAINS | BOSTON_DOMAINS
        
        if subset_name not in ALL_COMPARISON_DOMAINS:
            logger.warning(f"Domain {subset_name} not in comparison list. No fairness applied.")
            detailed_logger.debug(f"Domain {subset_name} not comparable")
            limit = len(all_valid_samples)
        else:
            # CRITICAL FIX: Count only TRAIN samples (respect train/val split)
            # This ensures fairness limits are based on train-only data
            domain_sample_counts = {}
            scene_map = load_scene_map(EXCEL_PATH)[0]  # scene_token -> domain
            scene_split = load_scene_map(EXCEL_PATH)[1]  # scene_token -> 'train'/'val'
            domain_scenes = {d: [] for d in ALL_COMPARISON_DOMAINS}
            
            # Collect only TRAIN scenes for each domain (exclude val)
            for token, domain in scene_map.items():
                if domain in ALL_COMPARISON_DOMAINS:
                    # Only include if this scene is marked as 'train' in Excel
                    if scene_split.get(token, '').lower() == 'train':
                        if token not in domain_scenes[domain]:
                            domain_scenes[domain].append(token)
            
            detailed_logger.debug(f"Counting TRAIN-ONLY samples per domain (excluding val scenes)")
            
            # Count TRAIN samples only
            for domain, tokens in domain_scenes.items():
                count = 0
                for scene_token in tokens:
                    try:
                        scene = nusc.get('scene', scene_token)
                    except KeyError:
                        continue
                    curr_sample_token = scene['first_sample_token']
                    while curr_sample_token:
                        count += 1
                        try:
                            sample = nusc.get('sample', curr_sample_token)
                            curr_sample_token = sample['next']
                        except KeyError:
                            break
                domain_sample_counts[domain] = count
            
            detailed_logger.debug(f"TRAIN-ONLY sample counts per domain: {domain_sample_counts}")
            
            detailed_logger.debug(f"Sample counts per domain: {domain_sample_counts}")
            
            # Determine limit using MIN
            if fairness_mode == 'TOTAL':
                limit = min(domain_sample_counts.values()) if domain_sample_counts else len(all_valid_samples)
                logger.info(f"FAIRNESS=TOTAL: {subset_name} limited to {limit} samples (min across all)")
            
            elif fairness_mode == 'COMPARATIVE':
                singapore_limit = min(
                    domain_sample_counts.get('singapore_day_clear', float('inf')),
                    domain_sample_counts.get('singapore_night_clear', float('inf'))
                )
                boston_limit = min(
                    domain_sample_counts.get('boston_day_clear', float('inf')),
                    domain_sample_counts.get('boston_day_rain', float('inf'))
                )
                
                if singapore_limit == float('inf'):
                    singapore_limit = len(all_valid_samples)
                if boston_limit == float('inf'):
                    boston_limit = len(all_valid_samples)
                
                limit = singapore_limit if subset_name in SINGAPORE_DOMAINS else boston_limit
                logger.info(f"FAIRNESS=COMPARATIVE: {subset_name} limited to {limit} samples")
            
            else:
                raise ValueError(f"Invalid fairness_mode: {fairness_mode}")
    
    # ===== STEP 3: APPLY DROP STRATEGY =====
    if len(all_valid_samples) <= limit:
        keep_samples = all_valid_samples
        logger.debug(f"{subset_name}: No reduction needed ({len(all_valid_samples)} <= {limit})")
    else:
        logger.info(f"Fairness: Reducing {len(all_valid_samples)} -> {limit} samples using {drop_strategy}")
        detailed_logger.info(f"Applying drop_strategy={drop_strategy} to {subset_name}")
        
        if drop_strategy == "TAIL":
            # Keep first N samples (temporal order preserved)
            keep_samples = all_valid_samples[:limit]
            logger.debug(f"DROP=TAIL: Keeping first {limit} samples (dropping last {len(all_valid_samples) - limit})")
            detailed_logger.debug(f"TAIL: kept samples [0:{limit}]")
        
        elif drop_strategy == "RANDOM":
            # Randomly select N samples from entire set
            random.seed(random_seed)
            selected_indices = sorted(random.sample(range(len(all_valid_samples)), limit))
            keep_samples = [all_valid_samples[i] for i in selected_indices]
            logger.debug(f"DROP=RANDOM: Randomly selected {limit} samples (seed={random_seed})")
            detailed_logger.debug(f"RANDOM: selected {len(selected_indices)} indices using seed {random_seed}")
        
        else:
            logger.warning(f"Unknown drop_strategy '{drop_strategy}', defaulting to TAIL")
            keep_samples = all_valid_samples[:limit]
    
    detailed_logger.info(f"Final for {subset_name}: {len(keep_samples)} samples")
    return keep_samples, limit




def distribute_samples_to_clients(
    sample_tokens: List[str],
    num_clients: int,
    temporal_grouping: bool = False,
    drop_strategy: str = "TAIL",
    random_seed: int = DEFAULT_RANDOM_SEED
) -> Dict[int, List[str]]:
    """
    Distributes training samples across multiple clients.
    
    Args:
        sample_tokens: List of sample tokens (should be ordered by time if temporal_grouping=True)
        num_clients: Number of clients
        temporal_grouping: True = sequential chunks, False = interleaved stride
        drop_strategy: 'TAIL' = take first N, 'RANDOM' = random sample
        random_seed: Seed for reproducible shuffling
    
    Returns:
        Dict[client_id (1-indexed) -> List[sample_tokens]]
    
    Logic:
        TEMPORAL_GROUPING=True (Sequential chunks):
            Client 1: [0:N]
            Client 2: [N:2N]
            Client 3: [2N:3N]
            Preserves temporal/spatial continuity per client
        
        TEMPORAL_GROUPING=False (Interleaved stride):
            Client 1: [0::num_clients]
            Client 2: [1::num_clients]
            Maximizes diversity per client, breaks continuity
        
        DROP_STRATEGY='TAIL': Take samples sequentially from start
        DROP_STRATEGY='RANDOM': Randomly select samples first, then organize
    """
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")
    
    if not sample_tokens:
        raise ValueError("Cannot distribute empty sample list")
    
    logger.debug(f"Distributing {len(sample_tokens)} samples to {num_clients} clients "
                 f"(temporal={temporal_grouping}, drop_strategy={drop_strategy})")
    
    # Handle drop strategy
    if drop_strategy == "RANDOM":
        # Randomly select samples (for when fairness limit was enforced)
        random.seed(random_seed)
        # Note: this is only needed if samples were already filtered by fairness
        # and we want to randomly choose from them
        shuffled = sample_tokens.copy()
        random.shuffle(shuffled)
        working_samples = shuffled
    elif drop_strategy == "TAIL":
        # Use samples sequentially (temporal order preserved)
        working_samples = sample_tokens
    else:
        raise ValueError(f"Invalid drop_strategy: {drop_strategy}. Must be 'TAIL' or 'RANDOM'")
    
    client_samples = {}
    
    if temporal_grouping:
        # Sequential chunking: divide into contiguous blocks
        chunk_size = len(working_samples) // num_clients
        remainder = len(working_samples) % num_clients
        
        idx = 0
        for client_id in range(1, num_clients + 1):
            # Give remainder samples to the first clients
            size = chunk_size + (1 if client_id <= remainder else 0)
            client_samples[client_id] = working_samples[idx:idx + size]
            logger.debug(f"Client {client_id}: {size} samples (temporal chunk)")
            idx += size
    
    else:
        # Interleaved stride: distribute round-robin
        for client_id in range(1, num_clients + 1):
            offset = client_id - 1  # 0-indexed offset
            samples = working_samples[offset::num_clients]
            client_samples[client_id] = samples
            logger.debug(f"Client {client_id}: {len(samples)} samples (interleaved)")
    
    # Verify distribution
    total_assigned = sum(len(v) for v in client_samples.values())
    if total_assigned != len(working_samples):
        raise RuntimeError(
            f"Distribution error: assigned {total_assigned} but had {len(working_samples)}"
        )
    
    logger.info(f"Successfully distributed samples: {[(i, len(v)) for i, v in sorted(client_samples.items())]}")
    return client_samples


# ================= DATA PROCESSING =================

def heal_pointers(records: List[Dict], table_name: str) -> None:
    """
    Fixes broken prev/next pointers in any table (sample_data, sample, etc.).
    
    When subsampling, some pointers may reference removed records.
    Set them to empty string to prevent crashes in mmdetection3d and external tools.
    
    CRITICAL FIX: Now heals both sample_data and sample pointers.
    External tools (NuScenes SDK utilities, some pipelines) may traverse samples,
    so sample.prev/next must also be healed for full data integrity.
    
    Args:
        records: List of record dicts with 'token', 'prev', 'next' keys
        table_name: Name of table (for logging, e.g., 'sample_data', 'sample')
    """
    valid_tokens = {r['token'] for r in records}
    healed_count = 0
    prev_healed = 0
    next_healed = 0
    
    for record in records:
        token = record['token']
        
        # Check and heal PREV pointer
        if record.get('prev'):
            if record['prev'] not in valid_tokens:
                detailed_logger.warning(f"Healing broken PREV in {table_name}: {token[:8]} -> {record['prev'][:8]} (missing)")
                record['prev'] = ""
                healed_count += 1
                prev_healed += 1
        
        # Check and heal NEXT pointer
        if record.get('next'):
            if record['next'] not in valid_tokens:
                detailed_logger.warning(f"Healing broken NEXT in {table_name}: {token[:8]} -> {record['next'][:8]} (missing)")
                record['next'] = ""
                healed_count += 1
                next_healed += 1
    
    if healed_count > 0:
        logger.warning(f"Healed {healed_count} broken pointers in {table_name} ({prev_healed} PREV, {next_healed} NEXT)")
        detailed_logger.info(f"Healing summary ({table_name}): {healed_count} total ({prev_healed} PREV, {next_healed} NEXT)")
    else:
        detailed_logger.debug(f"All {table_name} pointers valid (no healing needed)")



def _process_single_client_subset(
    subset_name: str,
    client_id: int,
    scene_tokens: List[str],
    client_train_samples: Set[str],
    client_val_samples: Set[str],
    subset_root: Path,
    nusc: NuScenes,
    raw_tables: Dict[str, List[Dict]]
):
    """
    Processes a single client's subset by filtering and linking data.
    
    Each client gets:
    - Unique partial TRAINING set (samples divided among clients)
    - FULL VALIDATION set (same for all clients in the domain)
    
    Args:
        subset_name: Domain name (e.g., 'boston_day_rain')
        client_id: Client identifier (1-indexed)
        scene_tokens: All scene tokens for this domain (used for traversal)
        client_train_samples: Training samples assigned to this client
        client_val_samples: ALL validation samples (full set for this domain)
        subset_root: Output directory for this client
        nusc: NuScenes instance
        raw_tables: Raw metadata tables
    
    Raises:
        KeyError: If referenced tokens are missing from the database
        RuntimeError: If filtering/writing fails
    """
    logger.debug(f"_process_single_client_subset: {subset_name}/client {client_id}")
    logger.debug(f"  Train samples: {len(client_train_samples)}, Val samples: {len(client_val_samples)}")
    
    subset_root.mkdir(parents=True, exist_ok=True)

    # --- A. Identification Phase (Graph Traversal) ---
    # Traverse graph for ALL samples (train + val) assigned to this client
    all_client_samples = client_train_samples | client_val_samples
    
    keep_samples = all_client_samples.copy()
    keep_sample_data = set()  # Images/Lidar (Keyframes AND Sweeps)
    keep_instances = set()
    
    logger.debug(f"Traversing graph for {len(keep_samples)} total samples (train+val)...")
    
    lidar_sweep_warnings = []
    missing_samples = []
    
    for sample_token in tqdm(
        keep_samples,
        desc=f"Analyzing {subset_name}/client{client_id} graph",
        leave=False
    ):
        try:
            sample = nusc.get('sample', sample_token)
        except KeyError as e:
            logger.error(f"Sample {sample_token} in client distribution but not in DB")
            missing_samples.append(sample_token)
            raise KeyError(f"Missing sample: {sample_token}") from e

        # Walk through Sample Data (Sensors)
        for sensor, sd_token in sample['data'].items():
            # 1. Keep the Keyframe
            keep_sample_data.add(sd_token)
            
            # 2. Walk BACKWARDS for Sweeps (Crucial for mmdet3d Lidar)
            try:
                sd = nusc.get('sample_data', sd_token)
            except KeyError as e:
                logger.error(f"SampleData {sd_token} missing from DB")
                raise KeyError(f"Missing sample_data: {sd_token}") from e
            
            curr_sd_prev = sd['prev']
            sweep_count = 0
            
            while curr_sd_prev:
                try:
                    prev_sd = nusc.get('sample_data', curr_sd_prev)
                except KeyError as e:
                    logger.error(f"Broken sweep chain at {curr_sd_prev} for sensor {sensor}")
                    detailed_logger.error(f"Sweep chain broken: sample={sample_token[:8]}, sensor={sensor}, prev={curr_sd_prev[:8]}")
                    break
                
                if prev_sd['is_key_frame']:
                    break  # Stop, we reached the previous keyframe
                
                keep_sample_data.add(curr_sd_prev)
                sweep_count += 1
                curr_sd_prev = prev_sd['prev']
            
            # Check for insufficient LiDAR sweeps
            if 'LIDAR' in sensor and sweep_count < 9:
                msg = f"Sample {sample_token[:8]}: {sensor} has only {sweep_count} sweeps (< 9)"
                lidar_sweep_warnings.append(msg)
                detailed_logger.warning(f"Low LiDAR sweeps: {msg}")
        
        # Collect Instance IDs from Annotations
        for ann_token in sample['anns']:
            try:
                ann = nusc.get('sample_annotation', ann_token)
            except KeyError as e:
                logger.error(f"SampleAnnotation {ann_token} missing from DB")
                raise KeyError(f"Missing annotation: {ann_token}") from e
            keep_instances.add(ann['instance_token'])

    logger.debug(f"Graph traversal complete: {len(keep_sample_data)} sample_data, {len(keep_instances)} instances")
    
    # Log traverse summary
    if lidar_sweep_warnings:
        logger.warning(f"Found {len(lidar_sweep_warnings)} samples with < 10 LiDAR sweeps")
        detailed_logger.warning(f"Low LiDAR sweep summary:\n" + "\n".join(lidar_sweep_warnings[:20]))
        if len(lidar_sweep_warnings) > 20:
            detailed_logger.warning(f"... and {len(lidar_sweep_warnings) - 20} more")
    
    if missing_samples:
        logger.error(f"Found {len(missing_samples)} missing samples in DB")
        detailed_logger.error(f"Missing samples: {missing_samples}")

    # --- B. Filtering Phase ---
    logger.debug(f"Filtering metadata tables...")
    
    target_scene_tokens = set(scene_tokens)
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

    logger.debug(f"Filtered tables: scene={len(new_tables['scene'])}, sample={len(new_tables['sample'])}, "
                 f"sample_data={len(new_tables['sample_data'])}, annotation={len(new_tables['sample_annotation'])}")

    # --- C. HEAL DATA INTEGRITY ---
    # Fix broken prev/next pointers in both sample and sample_data tables
    # This ensures external tools (NuScenes SDK, other pipelines) don't break
    # on traversals that encounter removed samples
    heal_pointers(new_tables['sample'], 'sample')
    heal_pointers(new_tables['sample_data'], 'sample_data')

    # --- D. Writing Phase ---
    try:
        for table_name, data in new_tables.items():
            save_json_table(data, subset_root, table_name)
        logger.debug(f"Successfully wrote JSON tables to {subset_root}")
    except Exception as e:
        logger.error(f"Failed to write JSON tables: {e}")
        raise RuntimeError(f"JSON writing failed: {e}") from e

    # --- E. Symlinking Phase ---
    logger.debug(f"Symlinking sensor files...")
    symlink_count = 0
    for sd in tqdm(new_tables['sample_data'], desc="Linking Files", leave=False):
        filename = sd['filename']  # e.g. samples/CAM_FRONT/xxx.jpg
        src_path = NUSC_SOURCE_ROOT / filename
        dst_path = subset_root / filename
        safe_symlink(src_path, dst_path)
        symlink_count += 1
    
    logger.debug(f"Created {symlink_count} symlinks")
    
    # --- F. Map Copying Phase ---
    logger.debug(f"Copying maps...")
    src_maps = NUSC_SOURCE_ROOT / "maps"
    dst_maps = subset_root / "maps"
    if src_maps.exists() and (not dst_maps.exists() or OVERWRITE):
        try:
            shutil.copytree(src_maps, dst_maps, dirs_exist_ok=True)
            logger.debug(f"Maps copied successfully")
        except Exception as e:
            logger.error(f"Failed to copy maps: {e}")
            raise RuntimeError(f"Map copying failed: {e}") from e
    elif not src_maps.exists():
        logger.warning(f"Source maps not found at {src_maps}")
    else:
        logger.debug(f"Maps already exist, skipping (OVERWRITE={OVERWRITE})")






def count_files_in_subsets() -> None:
    """
    Counts and logs the number of files in each subset folder.
    Provides summary statistics for verification and data auditing.
    """
    logger.info("="*80)
    logger.info("FILE COUNT SUMMARY")
    logger.info("="*80)
    
    if not BASE_OUT_ROOT.exists():
        logger.warning(f"Output root {BASE_OUT_ROOT} does not exist")
        return
    
    # Iterate through all configurations
    for config in CONFIGURATIONS:
        config_root = BASE_OUT_ROOT / config['name']
        if not config_root.exists():
            logger.warning(f"Configuration folder not found: {config_root}")
            continue
        
        logger.info(f"\nConfiguration: {config['name']}")
        logger.info(f"  temporal_grouping={config['temporal_grouping']}, "
                   f"fairness_mode={config['fairness_mode']}, "
                   f"clients_per_domain={config['clients_per_domain']}")
        
        # Iterate through domains
        for domain_dir in sorted(config_root.iterdir()):
            if not domain_dir.is_dir():
                continue
            
            domain_name = domain_dir.name
            
            # Check if multi-client (has numeric subdirectories) or single-client
            client_dirs = [d for d in domain_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            
            if client_dirs:
                # Multi-client setup
                logger.info(f"  Domain: {domain_name} (multi-client)")
                for client_dir in sorted(client_dirs, key=lambda x: int(x.name)):
                    client_id = client_dir.name
                    file_count = sum(1 for _ in client_dir.rglob('*') if _.is_file())
                    logger.info(f"    Client {client_id}: {file_count} files")
            else:
                # Single-client setup (files directly in domain folder)
                file_count = sum(1 for _ in domain_dir.rglob('*') if _.is_file())
                logger.info(f"  Domain: {domain_name}: {file_count} files")
    
    logger.info("="*80)


def main():
    """
    Main execution loop that processes multiple configurations.
    Each configuration generates a complete dataset variant.
    """
    logger.info("="*80)
    logger.info("Starting NuScenes Robust Subset Creator (Multi-Configuration Mode)")
    logger.info(f"Number of configurations to process: {len(CONFIGURATIONS)}")
    logger.info("="*80)
    
    # Load static data (NuScenes DB, metadata tables)
    logger.info("Loading NuScenes database and metadata tables...")
    scene_to_subset, scene_to_split = load_scene_map(EXCEL_PATH)
    
    logger.info("Initializing NuScenes DB...")
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(NUSC_SOURCE_ROOT), verbose=True)

    logger.info("Loading raw JSON tables...")
    raw_tables = {}
    for t in ["scene", "sample", "sample_data", "sample_annotation", "instance", 
              "log", "ego_pose", "calibrated_sensor"]:
        raw_tables[t] = load_json_table(NUSC_SOURCE_ROOT, t)
    for t in ["map", "sensor", "category", "attribute", "visibility"]:
        raw_tables[t] = load_json_table(NUSC_SOURCE_ROOT, t)

    # ===== LOOP THROUGH CONFIGURATIONS =====
    for config_idx, config in enumerate(CONFIGURATIONS, 1):
        logger.info("="*80)
        logger.info(f"CONFIGURATION {config_idx}/{len(CONFIGURATIONS)}: {config['name']}")
        logger.info(f"  temporal_grouping: {config['temporal_grouping']}")
        logger.info(f"  drop_strategy: {config['drop_strategy']}")
        logger.info(f"  fairness_mode: {config['fairness_mode']}")
        logger.info(f"  clients_per_domain: {config['clients_per_domain']}")
        logger.info("="*80)
        
        out_root = BASE_OUT_ROOT / config['name']
        out_root.mkdir(parents=True, exist_ok=True)
        
        try:
            _process_configuration(
                config=config,
                out_root=out_root,
                scene_to_subset=scene_to_subset,
                scene_to_split=scene_to_split,
                nusc=nusc,
                raw_tables=raw_tables
            )
        except Exception as e:
            logger.error(f"Configuration {config['name']} failed: {e}", exc_info=True)
            raise

    logger.info("="*80)
    logger.info("SUCCESS. All configurations processed.")
    logger.info(f"Log: {BASE_OUT_ROOT / 'subset_creation.log'}")
    logger.info("="*80)
    
    # Count and log files in each subset
    count_files_in_subsets()


def _process_configuration(
    config: Dict,
    out_root: Path,
    scene_to_subset: Dict[str, str],
    scene_to_split: Dict[str, str],
    nusc: NuScenes,
    raw_tables: Dict[str, List[Dict]]
):
    """
    Process one configuration.
    
    Workflow:
    1. Separate scenes into train/val by split
    2. Extract samples from train scenes, apply fairness filtering
    3. Extract ALL samples from val scenes (no filtering)
    4. Distribute train samples to clients
    5. For each client: merge train + val samples, process
    """
    logger.info(f"Processing configuration: {config['name']}")
    
    # === STEP 1: Separate train/val scenes ===
    domains_train = {}  # subset_name -> list of train scene tokens
    domains_val = {}    # subset_name -> list of val scene tokens
    
    for scene_token, subset_name in scene_to_subset.items():
        split = scene_to_split.get(scene_token, 'train')  # Default to train if missing
        
        if subset_name not in domains_train:
            domains_train[subset_name] = []
            domains_val[subset_name] = []
        
        if split == 'train':
            domains_train[subset_name].append(scene_token)
        else:  # 'val'
            domains_val[subset_name].append(scene_token)
    
    # Log distribution
    for subset_name in sorted(set(scene_to_subset.values())):
        n_train = len(domains_train.get(subset_name, []))
        n_val = len(domains_val.get(subset_name, []))
        logger.info(f"  {subset_name}: {n_train} train scenes, {n_val} val scenes")
    
    # === STEP 2: Process each domain ===
    for subset_name in sorted(set(scene_to_subset.values())):
        train_scene_tokens = domains_train.get(subset_name, [])
        val_scene_tokens = domains_val.get(subset_name, [])
        
        if not train_scene_tokens and not val_scene_tokens:
            logger.warning(f"Skipping empty domain: {subset_name}")
            continue
        
        logger.info(f"Processing domain: {subset_name} ({len(train_scene_tokens)} train, {len(val_scene_tokens)} val)")
        
        # === STEP 2A: Extract and filter TRAIN samples ===
        if train_scene_tokens:
            try:
                train_samples, fairness_limit = get_fairness_filtered_samples(
                    subset_name,
                    train_scene_tokens,
                    nusc,
                    fairness_mode=config['fairness_mode'],
                    drop_strategy=config['drop_strategy'],
                    random_seed=config['random_seed']
                )
            except Exception as e:
                logger.error(f"Fairness filtering failed for {subset_name}: {e}")
                detailed_logger.error(f"Fairness filtering exception: {e}", exc_info=True)
                raise
            
            logger.info(f"  Train samples after filtering: {len(train_samples)}")
        else:
            train_samples = []
            logger.info(f"  No train scenes for {subset_name}")
        
        # === STEP 2B: Extract FULL VAL samples (no filtering) ===
        if val_scene_tokens:
            val_samples = []
            for scene_token in val_scene_tokens:
                try:
                    scene = nusc.get('scene', scene_token)
                except KeyError:
                    logger.warning(f"Scene {scene_token} not found in DB")
                    continue
                
                curr_sample_token = scene['first_sample_token']
                while curr_sample_token:
                    val_samples.append(curr_sample_token)
                    sample = nusc.get('sample', curr_sample_token)
                    curr_sample_token = sample['next']
            
            logger.info(f"  Validation samples (no filtering): {len(val_samples)}")
        else:
            val_samples = []
            logger.info(f"  No validation scenes for {subset_name}")
        
        # === STEP 2C: Distribute TRAIN samples to clients ===
        if train_samples:
            try:
                client_train_dist = distribute_samples_to_clients(
                    train_samples,
                    num_clients=config['clients_per_domain'],
                    temporal_grouping=config['temporal_grouping'],
                    drop_strategy=config['drop_strategy'],
                    random_seed=config['random_seed']
                )
            except Exception as e:
                logger.error(f"Client distribution failed for {subset_name}: {e}")
                raise
        else:
            # No train samples, but still create clients with only validation
            client_train_dist = {
                i: [] for i in range(1, config['clients_per_domain'] + 1)
            }
        
        # === STEP 2D: Process each client ===
        for client_id in range(1, config['clients_per_domain'] + 1):
            client_train_samples = set(client_train_dist.get(client_id, []))
            client_val_samples = set(val_samples)
            
            logger.info(f"  {subset_name} / Client {client_id}: {len(client_train_samples)} train, {len(client_val_samples)} val")
            
            # Determine output directory
            if config['clients_per_domain'] == 1:
                # Single client: no subdirectory
                subset_root = out_root / subset_name
            else:
                # Multiple clients: create subdirectories
                subset_root = out_root / subset_name / str(client_id)
            
            try:
                _process_single_client_subset(
                    subset_name=subset_name,
                    client_id=client_id,
                    scene_tokens=train_scene_tokens + val_scene_tokens,  # All scenes for traversal
                    client_train_samples=client_train_samples,
                    client_val_samples=client_val_samples,
                    subset_root=subset_root,
                    nusc=nusc,
                    raw_tables=raw_tables
                )
            except Exception as e:
                logger.error(f"Processing failed for {subset_name}/client {client_id}: {e}", exc_info=True)
                raise
    
    logger.info(f"Configuration {config['name']} completed successfully")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os
import json
import csv
import argparse
from collections import Counter, defaultdict

def load_json(p):
    if not os.path.exists(p):
        print(f"Error: File not found {p}")
        return []
    with open(p, "r") as f:
        return json.load(f)

def resolve_tables_dir(root):
    for sub in ["v1.0-trainval", "v1.0-mini", "v1.0-test"]:
        v = os.path.join(root, sub)
        if os.path.isdir(v):
            return v
    return root

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Metadata root")
    ap.add_argument("out_dir", help="Output directory")
    ap.add_argument("--channel", default="LIDAR_TOP")
    ap.add_argument("--cap", type=int, default=10)
    args = ap.parse_args()

    tables = resolve_tables_dir(args.root)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading tables from: {tables}")
    samples = load_json(os.path.join(tables, "sample.json"))
    sd_all = load_json(os.path.join(tables, "sample_data.json"))

    if not sd_all:
        print("CRITICAL: sample_data.json is empty or failed to load.")
        return

    # Diagnostic: Peek at the first entry to see actual keys
    print(f"DEBUG: First record keys: {list(sd_all[0].keys())}")

    sd_lookup = {}
    per_sample = defaultdict(list)
    target = args.channel.upper()
    found_channels = set()

    for r in sd_all:
        tok = r["token"]
        
        # 1. Try to find the channel name
        # Some schemas use 'channel', others 'sensor_modality', others use the filename prefix
        chan_val = str(r.get("channel", r.get("sensor_modality", "UNKNOWN"))).upper()
        
        # Fallback: if it's still UNKNOWN, check the filename for 'LIDAR'
        if chan_val == "UNKNOWN" and "LIDAR" in r.get("file_name", r.get("filename", "")).upper():
            chan_val = "LIDAR_TOP" # Assume top if lidar is found in path

        found_channels.add(chan_val)
        
        sd_lookup[tok] = {
            "prev": r.get("prev", ""),
            "ts": int(r.get("timestamp", 0)),
            "isk": bool(r.get("is_key_frame", False)), 
            "st": r.get("sample_token", ""),
            "chan": chan_val
        }
        
        # Match against our target
        if chan_val == target:
            st = r.get("sample_token")
            if st:
                per_sample[st].append(tok)

    print(f"DEBUG: Detected channels: {sorted(list(found_channels))}")

    hist = Counter()
    rows = []
    missing_target = 0

    for s in samples:
        st = s["token"]
        ts_s = int(s.get("timestamp", 0))
        
        toks = per_sample.get(st)
        if not toks:
            missing_target += 1
            rows.append((st, 0, 0))
            continue

        # Find best Lidar anchor for this sample
        best = None
        best_key = None
        for tok in toks:
            rec = sd_lookup[tok]
            # Keyframes get priority, then temporal proximity
            key = (1 if rec["isk"] else 0, -abs(rec["ts"] - ts_s))
            if best_key is None or key > best_key:
                best_key = key
                best = tok

        # Trace the 'prev' chain
        cnt = 0
        cur_tok = sd_lookup[best]["prev"]
        while cur_tok and cur_tok in sd_lookup:
            cnt += 1
            cur_tok = sd_lookup[cur_tok]["prev"]
            if cnt > 100: break # Safety break for circular refs

        bucket = min(cnt, args.cap)
        hist[bucket] += 1
        rows.append((st, cnt, bucket))

    out_csv = os.path.join(args.out_dir, f"prev_frames_{args.channel}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_token", "num_prev_frames_available", f"bucket_0_to_{args.cap}"])
        w.writerows(rows)

    print(f"\n--- Results ---")
    print(f"Samples processed: {len(samples)}")
    print(f"Targeting: {target}")
    print(f"Samples missing target: {missing_target}")
    print(f"Samples with at least 1 {target}: {len(samples) - missing_target}")
    
    print("\nHistogram (Prev Frames):")
    for i in range(args.cap + 1):
        label = f"{i}" if i < args.cap else f"{i}+"
        print(f"  {label.ljust(4)} : {hist.get(i, 0)}")
    
    print(f"\nWrote CSV to: {out_csv}")

if __name__ == "__main__":
    main()
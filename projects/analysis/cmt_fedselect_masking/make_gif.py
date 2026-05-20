import os
import re
from PIL import Image

# ==========================================
# --- TUNEABLE PARAMETERS (Modify here) ---
# ==========================================
# INPUT_FOLDER = "/home/jolle/mmdet/mmdetection3d/projects/analysis/cmt_fedselect_masking/outputs/all_final_no_elastic"
# OUTPUT_FOLDER = "/home/jolle/mmdet/mmdetection3d/projects/analysis/cmt_fedselect_masking/outputs/videos_final_no_elastic"

INPUT_FOLDER = "/home/jolle/mmdet/mmdetection3d/projects/analysis/cmt_fedselect_masking/outputs/all_final_full_elastic"
OUTPUT_FOLDER = "/home/jolle/mmdet/mmdetection3d/projects/analysis/cmt_fedselect_masking/outputs/videos_final_full_elastic"

# Time (in milliseconds) for each frame in the GIF
FRAME_DURATION_MS = 2000  # <--- Change this to set speed

# If True, Round_0 images are skipped.
SKIP_ROUND_0 = True

# The model letters to look for and process
MODELS_TO_PROCESS = ["A", "B", "C", "D", "E"]
# ==========================================

def get_round_number(filename):
    """Helper to extract the round number for sorting."""
    # Pattern: 'Round_[digit]_model_'
    match = re.search(r"Round_(\d+)_model_", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return -1

def create_model_gif(model_letter, input_path, output_path, duration, skip_round_0):
    """Creates a sorted GIF for a single model."""
    print(f"\nProcessing Model {model_letter}...")

    # Collect matching files: e.g., 'Round_X_model_A.png'
    matching_files = []
    for f in os.listdir(input_path):
        # Must contain 'model_[letter].png' and follow pattern
        match = re.search(rf"Round_(\d+)_model_{model_letter}\.png", f, re.IGNORECASE)
        if not match:
            continue

        round_num = int(match.group(1))
        if skip_round_0 and round_num == 0:
            continue

        matching_files.append(f)

    if not matching_files:
        print(f"  No images found for Model {model_letter}. Skipping.")
        return

    # Sort files correctly by the round number
    sorted_files = sorted(matching_files, key=get_round_number)
    print(f"  Found {len(sorted_files)} images, sorting chronologically...")

    images = []
    for filename in sorted_files:
        try:
            filepath = os.path.join(input_path, filename)
            img = Image.open(filepath)
            images.append(img)
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
            continue

    if not images:
        print(f"  Error: No valid images could be loaded for Model {model_letter}.")
        return

    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Define output filename
    output_filename = f"model_{model_letter}.gif"
    output_filepath = os.path.join(output_path, output_filename)

    # Save as an animated GIF
    # duration is per frame in ms, loop=0 means loop indefinitely
    print(f"  Saving {output_filename} with {duration}ms per frame...")
    images[0].save(
        output_filepath,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"GIF created: {output_filepath}")

if __name__ == '__main__':
    # Initialize variables with the parameters
    in_dir = INPUT_FOLDER
    out_dir = OUTPUT_FOLDER
    frame_time = FRAME_DURATION_MS
    models = MODELS_TO_PROCESS
    skip_round_0 = SKIP_ROUND_0

    # Ensure input directory exists
    if not os.path.exists(in_dir):
        print(f"Input folder '{in_dir}' not found. Please create it.")
    else:
        # Loop over models to create GIFs
        for model in models:
            create_model_gif(model, in_dir, out_dir, frame_time, skip_round_0)
        print("\nAll model GIF generation tasks complete.")
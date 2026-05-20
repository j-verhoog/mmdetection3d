import os
import re
import shutil
import subprocess
import tempfile

# ==========================================
# --- TUNEABLE PARAMETERS (Modify here) ---
# ==========================================

# INPUT_FOLDER = "/home/jolle/mmdet/mmdetection3d/projects/analysis/cmt_fedselect_masking/outputs/all_final_no_elastic"
# OUTPUT_FOLDER = "/home/jolle/mmdet/mmdetection3d/projects/analysis/cmt_fedselect_masking/outputs/videos_final_no_elastic"

INPUT_FOLDER = "/home/jolle/mmdet/mmdetection3d/projects/analysis/cmt_fedselect_masking/outputs/all_final_full_elastic"
OUTPUT_FOLDER = "/home/jolle/mmdet/mmdetection3d/projects/analysis/cmt_fedselect_masking/outputs/videos_final_full_elastic"

# Time (in milliseconds) for each frame in the output video
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

def create_model_video(model_letter, input_path, output_path, duration_ms, skip_round_0):
    """Creates a sorted, lossless video for a single model."""
    print(f"\nProcessing Model {model_letter}...")

    # Collect matching files: e.g., 'Round_X_model_A.png'
    matching_files = []
    for f in os.listdir(input_path):
        # Must contain 'model_[letter].png' and follow pattern.
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

    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Define output filename
    output_filename = f"model_{model_letter}.mov"
    output_filepath = os.path.join(output_path, output_filename)

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        print("  Error: ffmpeg not found in PATH. Install ffmpeg to export videos.")
        return

    # Use ffmpeg concat demuxer so each PNG is shown for duration_ms.
    duration_s = duration_ms / 1000.0
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as concat_file:
        concat_path = concat_file.name
        for filename in sorted_files:
            filepath = os.path.join(input_path, filename)
            concat_file.write(f"file '{filepath}'\n")
            concat_file.write(f"duration {duration_s}\n")
        # ffmpeg concat expects the last file to be repeated to honor final duration.
        last_filepath = os.path.join(input_path, sorted_files[-1])
        concat_file.write(f"file '{last_filepath}'\n")

    ffmpeg_cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_path,
        "-vsync",
        "vfr",
        "-c:v",
        "png",
        "-pix_fmt",
        "rgb24",
        output_filepath,
    ]

    try:
        print(f"  Saving lossless video {output_filename} with {duration_ms}ms per frame...")
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        print(f"Video created: {output_filepath}")
    except subprocess.CalledProcessError as e:
        print(f"  ffmpeg failed for Model {model_letter}: {e.stderr}")
    finally:
        if os.path.exists(concat_path):
            os.remove(concat_path)

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
        # Loop over models to create videos
        for model in models:
            create_model_video(model, in_dir, out_dir, frame_time, skip_round_0)
        print("\nAll model video generation tasks complete.")
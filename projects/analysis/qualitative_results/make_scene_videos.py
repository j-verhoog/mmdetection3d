import os
import glob
import cv2

def extract_timestamp(filename):
    """
    Extracts the numerical timestamp from the filename to guarantee 
    perfect chronological sorting regardless of string edge-cases.
    Example input: n015-2018-11-21-19-21-35+0800__CAM_FRONT__1542799602112460_CAM_FRONT_bev.jpg
    Returns: 1542799602112460 (as an integer)
    """
    try:
        basename = os.path.basename(filename)
        # Split by the double underscore to separate the scene info from the timestamp component
        parts = basename.split('__')
        # Grab the last part and split by single underscore to isolate the number
        timestamp_str = parts[-1].split('_')[0]
        return int(timestamp_str)
    except (IndexError, ValueError):
        # Fallback to standard alphanumeric parsing if the format deviates unexpectedly
        return filename

def create_videos_from_frames(image_folder, output_folder):
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # The 4 distinct categories requested
    video_types = ['bev', 'gt', 'model_a', 'model_b', 'bev_overlay']
    

    # 0.5 seconds per frame translates to 2.0 frames per second (FPS)
    fps = 2.0
    
    for v_type in video_types:
        # Match filenames ending with the specific suffix pattern
        search_pattern = os.path.join(image_folder, f"*_CAM_FRONT_{v_type}.jpg")
        image_files = glob.glob(search_pattern)
        
        if not image_files:
            print(f"[-] No images found matching pattern for: {v_type}")
            continue
            
        # Sort images cleanly by their explicit microsecond timestamp
        image_files.sort(key=extract_timestamp)
        
        print(f"[+] Processing '{v_type}' video sequence ({len(image_files)} frames found)...")
        
        # Read the first frame to capture spatial canvas dimensions
        first_img = cv2.imread(image_files[0])
        if first_img is None:
            print(f"[!] Error: Could not load baseline image layout for {image_files[0]}")
            continue
            
        height, width, _ = first_img.shape
        dimensions = (width, height)
        
        # Construct output file path
        output_video_path = os.path.join(output_folder, f"sequence_{v_type}.mp4")
        
        # Initialize VideoWriter targeting MP4 container format
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, dimensions)
        
        # Write frames sequentially into the stream
        for idx, file_path in enumerate(image_files):
            img = cv2.imread(file_path)
            if img is not None:
                video_writer.write(img)
            else:
                print(f"[!] Warning: Frame skipped due to read failure at index {idx}: {file_path}")
                
        # Close the file system stream wrapper safely
        video_writer.release()
        print(f"[✔] Successfully exported: {output_video_path}\n")

if __name__ == "__main__":
    # --- CONFIGURATION PATHS ---
    # Change this to the path where your folder of images is stored
    IMAGE_SOURCE_DIR = "/home/jolle/mmdet/mmdetection3d/projects/analysis/qualitative_results/test/e8834785d9ff4783a5950281a4579943/lidar_camera" 
    
    # Change this to where you want the 4 output videos saved
    VIDEO_OUTPUT_DIR = "/home/jolle/mmdet/mmdetection3d/projects/analysis/qualitative_results/test/e8834785d9ff4783a5950281a4579943/video"
    # ---------------------------
    
    print("Starting video synthesis pipeline...")
    create_videos_from_frames(IMAGE_SOURCE_DIR, VIDEO_OUTPUT_DIR)
    print("Pipeline compilation finished.")
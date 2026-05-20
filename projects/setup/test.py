import os

def count_files(folder_path):
    """
    Count the number of files in a folder and all its subfolders.
    
    Args:
        folder_path (str): Path to the folder
        
    Returns:
        int: Total number of files
    """
    file_count = 0
    for root, dirs, files in os.walk(folder_path):
        file_count += len(files)
    return file_count


if __name__ == "__main__":
    folder = "/home/jolle/mmdet/nuscenes_subsets_full/boston_day_clear"
    total_files = count_files(folder)
    print(f"Total files day clear: {total_files}")
    
    folder = "/home/jolle/mmdet/nuscenes_subsets_full/boston_day_rain"
    total_files = count_files(folder)
    print(f"Total files boston day rain: {total_files}")
    
    folder = "/home/jolle/mmdet/nuscenes_subsets_full/singapore_day_clear"
    total_files = count_files(folder)
    print(f"Total files singapore day clear: {total_files}")
    
    folder = "/home/jolle/mmdet/nuscenes_subsets_full/singapore_night_clear"
    total_files = count_files(folder)
    print(f"Total files singapore night clear: {total_files}")
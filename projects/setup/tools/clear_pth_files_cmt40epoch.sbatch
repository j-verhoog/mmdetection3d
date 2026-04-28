#!/bin/bash

# Define your base directory
BASE_DIR="/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/cmt_40_epoch"

# 1. EXCLUDE FOLDERS: Add any folders you want to COMPLETELY ignore here
# For example: IGNORE_FOLDERS=("PCGrad" "NoComm_40Epochs")
IGNORE_FOLDERS=()

cd "$BASE_DIR" || { echo "Failed to change directory to $BASE_DIR"; exit 1; }

# Loop through all directories in the base path
for dir in */; do
    dir_name=${dir%/} # Remove the trailing slash for cleaner matching

    # Check if the folder is in the completely ignore list
    skip=false
    for ignore in "${IGNORE_FOLDERS[@]}"; do
        if [[ "$dir_name" == "$ignore" ]]; then
            skip=true
            break
        fi
    done

    if [ "$skip" = true ]; then
        echo "⏭️ Skipping excluded folder: $dir_name"
        continue
    fi

    # 2. CENTRALIZED MODEL LOGIC (Keep epochs 10-25 & 40)
    if [[ "$dir_name" == "centralized_model" ]]; then
        echo "📂 Processing $dir_name (Rules: Keep epochs 10-25 & 40)..."
        
        # Find all .pth files in the centralized folder
        find "$dir_name" -type f -name "*.pth" | while read -r file; do
            filename=$(basename "$file")

            # Extract the epoch number exactly from the 'epoch_XX.pth' format
            if [[ "$filename" =~ ^epoch_([0-9]+)\.pth$ ]]; then
                epoch=${BASH_REMATCH[1]}

                # Check if epoch is between 10 and 25, OR exactly 40
                if { [ "$epoch" -ge 10 ] && [ "$epoch" -le 25 ]; } || [ "$epoch" -eq 40 ]; then
                    echo "  ✅ [KEEPING] $file"
                else
                    echo "  🗑️ [DELETING] $file"
                    # UNCOMMENT THE LINE BELOW TO ACTUALLY DELETE
                    # rm "$file"
                fi
            else
                # If it's a .pth file but doesn't match 'epoch_XX.pth' (like latest.pth), delete it
                echo "  🗑️ [DELETING] $file (Not a targeted epoch file)"
                # UNCOMMENT THE LINE BELOW TO ACTUALLY DELETE
                # rm "$file"
            fi
        done

    # 3. ALL OTHER FOLDERS LOGIC (Keep epoch_40.pth only)
    else
        echo "📂 Processing $dir_name (Rules: Keep epoch_40.pth only)..."
        
        # Finds all .pth files deep inside the round_X/ModelY/ folders, EXCEPT 'epoch_40.pth'
        # This will target latest.pth, merged_X.pth, epoch_1.pth, etc.
        find "$dir_name" -type f -name "*.pth" ! -name "epoch_40.pth" | while read -r file; do
            echo "  🗑️ [DELETING] $file"
            # UNCOMMENT THE LINE BELOW TO ACTUALLY DELETE
            # rm "$file"
        done
    fi
done

echo "🎉 Cleanup script finished!"
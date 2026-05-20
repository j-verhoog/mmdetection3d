#!/bin/bash
#SBATCH --partition=general # request partition
#SBATCH --qos=short # Quality of service, short default
#SBATCH --job-name=cleanup_epochs
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/tools/total_cleanup_epochs_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/tools/total_cleanup_epochs_%j.err
#SBATCH --time=4:00:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# Define the parent directory where all these projects live
PARENT_DIR="/tudelft.net/staff-umbrella/MscThesisjverhoog/projects"

# Define the specific target folders you want to clean up
TARGET_DIRS=("cmt_fedavg" "cmt_fedavg_all_val" "fcos3d" "pointpillars" "pointpillars4x8" "test")

cd "$PARENT_DIR" || { echo "Failed to change directory to $PARENT_DIR"; exit 1; }

# Loop through each of the target directories
for t_dir in "${TARGET_DIRS[@]}"; do
    if [ ! -d "$t_dir" ]; then
        echo "⚠️ Skipping $t_dir: Directory does not exist."
        continue
    fi

    echo "=========================================================="
    echo "📂 Processing Project Folder: $t_dir"
    echo "=========================================================="
    
    # This clever command finds every unique sub-directory that contains a .pth file
    # This way, it doesn't matter how deeply nested your files are.
    find "$t_dir" -type f -name "*.pth" -exec dirname {} \; | sort -u | while read -r model_dir; do
        echo "  🔍 Checking sub-directory: $model_dir"
        
        # Find the highest two epoch numbers in THIS specific sub-directory
        latest_epochs=($(find "$model_dir" -maxdepth 1 -type f -name "epoch_*.pth" | grep -Eo 'epoch_[0-9]+\.pth' | grep -Eo '[0-9]+' | sort -n | tail -n 2))
        
        if [ ${#latest_epochs[@]} -gt 0 ]; then
            echo "    🛡️ Protecting latest epochs: ${latest_epochs[*]}"
        else
            echo "    ⚠️ No epoch_*.pth files found to protect here."
        fi
        
        # Loop through all .pth files in this specific directory (maxdepth 1 prevents double-counting)
        find "$model_dir" -maxdepth 1 -type f -name "*.pth" | while read -r file; do
            filename=$(basename "$file")
            
            # Extract the epoch number exactly from the 'epoch_XX.pth' format
            if [[ "$filename" =~ ^epoch_([0-9]+)\.pth$ ]]; then
                epoch=${BASH_REMATCH[1]}
                
                # Check if it's one of the latest two epochs
                is_latest=false
                for le in "${latest_epochs[@]}"; do
                    if [ "$epoch" -eq "$le" ]; then
                        is_latest=true
                        break
                    fi
                done
                
                if [ "$is_latest" = true ]; then
                    echo "    ✅ [KEEPING] $file"
                else
                    echo "    🗑️ [DELETING] $file"
                    # UNCOMMENT THE LINE BELOW TO ACTUALLY DELETE
                    rm "$file"
                fi
            else
                # Delete files that don't match 'epoch_XX.pth' (like latest.pth, merged.pth, etc.)
                echo "    🗑️ [DELETING] $file (Not a targeted epoch file)"
                # UNCOMMENT THE LINE BELOW TO ACTUALLY DELETE
                # rm "$file"
            fi
        done
    done
done

echo "🎉 Cleanup script finished!"
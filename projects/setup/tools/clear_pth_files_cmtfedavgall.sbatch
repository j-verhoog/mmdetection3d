#!/bin/bash
#SBATCH --partition=general # request partition
#SBATCH --qos=short # Quality of service, short default
#SBATCH --job-name=remove_stuff
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/tools/remove_stuff_cmtfedavg_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/tools/remove_stuff_cmtfedavg_%j.err
#SBATCH --time=4:00:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# Define your base directory
BASE_DIR="/tudelft.net/staff-umbrella/MscThesisjverhoog/projects/cmt_fedavg_all"

# 1. EXCLUDE FOLDERS: Add any folders you want to COMPLETELY ignore here
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

    # 2. CENTRALIZED MODEL LOGIC (Any folder with 'centralized' in the name)
    # Rules: Keep 10-25, 40, 50, 60, 70, 80, 90, and latest 2 epochs
    if [[ "$dir_name" == *"centralized"* ]]; then
        echo "📂 Processing $dir_name (Rules: Keep epochs 10-25, 40, 50-90 (tens), & latest 2)..."
        
        # Dynamically find the highest two epoch numbers to protect them if the run is active
        latest_epochs=($(find "$dir_name" -maxdepth 1 -type f -name "epoch_*.pth" | grep -Eo 'epoch_[0-9]+\.pth' | grep -Eo '[0-9]+' | sort -n | tail -n 2))
        if [ ${#latest_epochs[@]} -gt 0 ]; then
            echo "  🛡️ Protecting latest centralized epochs: ${latest_epochs[*]}"
        fi
        
        # Find all .pth files in the centralized folder
        find "$dir_name" -type f -name "*.pth" | while read -r file; do
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

                # Check against the expanded epoch keeping rules
                if { [ "$epoch" -ge 10 ] && [ "$epoch" -le 25 ]; } || \
                   [ "$epoch" -eq 40 ] || [ "$epoch" -eq 50 ] || \
                   [ "$epoch" -eq 60 ] || [ "$epoch" -eq 70 ] || \
                   [ "$epoch" -eq 80 ] || [ "$epoch" -eq 90 ] || \
                   [ "$is_latest" = true ]; then
                    echo "  ✅ [KEEPING] $file"
                else
                    echo "  🗑️ [DELETING] $file"
                    # UNCOMMENT THE LINE BELOW TO ACTUALLY DELETE
                    rm "$file"
                fi
            else
                # Catch-all for things like latest.pth or incorrectly formatted files
                echo "  🗑️ [DELETING] $file (Not a targeted epoch file)"
                # UNCOMMENT THE LINE BELOW TO ACTUALLY DELETE
                rm "$file"
            fi
        done

    # 3. ALL OTHER FOLDERS LOGIC (Keep epoch_40.pth & completely protect latest 2 rounds)
    else
        echo "📂 Processing $dir_name (Rules: Keep epoch_40.pth & protect latest 2 rounds)..."
        
        # Identify the latest two round numbers by parsing directory names and sorting numerically
        round_nums=$(find "$dir_name" -maxdepth 1 -type d -name "round_*" | grep -Eo 'round_[0-9]+$' | grep -Eo '[0-9]+' | sort -n)
        latest_rounds=($(echo "$round_nums" | tail -n 2))

        # Build 'prune' arguments to instruct the find command to completely skip these directories
        PRUNE_ARGS=()
        if [ ${#latest_rounds[@]} -gt 0 ]; then
            echo "  🛡️ Protecting latest rounds: ${latest_rounds[*]}"
            PRUNE_ARGS+=("-type" "d" "(")
            for i in "${!latest_rounds[@]}"; do
                if [ $i -gt 0 ]; then
                    PRUNE_ARGS+=("-o")
                fi
                PRUNE_ARGS+=("-name" "round_${latest_rounds[$i]}")
            done
            PRUNE_ARGS+=(")" "-prune" "-o")
        fi
        
        # Finds all .pth files, strictly skipping the protected rounds, EXCEPT 'epoch_40.pth'
        find "$dir_name" "${PRUNE_ARGS[@]}" -type f -name "*.pth" ! -name "epoch_40.pth" -print | while read -r file; do
            echo "  🗑️ [DELETING] $file"
            # UNCOMMENT THE LINE BELOW TO ACTUALLY DELETE
            rm "$file"
        done
    fi
done

echo "🎉 Cleanup script finished!"
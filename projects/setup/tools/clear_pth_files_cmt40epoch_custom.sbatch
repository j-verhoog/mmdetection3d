#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --job-name=retrospective_cleanup
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/tools/cleanup_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/tools/cleanup_%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# 1. SPECIFY EXACT DIRECTORIES TO CLEAN
# Add as many absolute paths as you want inside these parentheses.
TARGET_DIRS=(
    "/home/nfs/jtverhoog/mmdet/MscThesisjverhoog/projects/cmt_40_epoch/FedDyn_40epochs/"
    "/home/nfs/jtverhoog/mmdet/MscThesisjverhoog/projects/cmt_40_epoch/FedRep_retry/"
    "/home/nfs/jtverhoog/mmdet/MscThesisjverhoog/projects/cmt_40_epoch/FedRep/"
    "/home/nfs/jtverhoog/mmdet/MscThesisjverhoog/projects/cmt_40_epoch/FedOMG_40epochs/"
)

echo "=== Starting Retrospective Cleanup ==="

# Loop through each provided directory
for target_dir in "${TARGET_DIRS[@]}"; do
    # Remove trailing slash for consistency
    target_dir=${target_dir%/}
    
    echo "------------------------------------------------"
    echo "📂 Processing Directory: $target_dir"
    
    if [ ! -d "$target_dir" ]; then
        echo "  ⚠️ ERROR: Directory not found. Skipping."
        continue
    fi

    # Identify all round numbers by parsing directory names and sorting numerically
    # This finds folders like round_1, round_2, ..., round_11
    round_nums=$(find "$target_dir" -maxdepth 1 -type d -name "round_*" | grep -Eo 'round_[0-9]+$' | grep -Eo '[0-9]+' | sort -n)

    if [ -z "$round_nums" ]; then
        echo "  ⚠️ No 'round_X' folders found here. Skipping."
        continue
    fi

    # Get the highest 2 rounds to protect them
    keep_rounds=($(echo "$round_nums" | tail -n 2))
    echo "  🛡️ Protecting latest 2 rounds: ${keep_rounds[*]}"

    # Iterate over EVERY round number found in this directory
    for r in $round_nums; do
        
        # Check if this specific round is in our protected list
        is_kept=false
        for kr in "${keep_rounds[@]}"; do
            if [ "$r" -eq "$kr" ]; then
                is_kept=true
                break
            fi
        done

        if [ "$is_kept" = true ]; then
            echo "  ⏭️ Skipping protected round: round_$r"
        else
            echo "  🧹 Cleaning up old round: round_$r"
            
            # Target ONLY .pth files strictly inside this specific old round folder
            find "$target_dir/round_$r" -type f -name "*.pth" | while read -r file; do
                echo "    🗑️ Deleted: $file"
                # WARNING: rm is active! 
                rm "$file"
            done
        fi
    done
done

echo "------------------------------------------------"
echo "🎉 Cleanup script finished!"
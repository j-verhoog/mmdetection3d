#!/bin/bash
#SBATCH --partition=general # request partition
#SBATCH --qos=short # Quality of service, short default
#SBATCH --job-name=remove_stuff
#SBATCH --output=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/tools/remove_stuff_%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/MscThesisjverhoog/logs/tools/remove_stuff_%j.err
#SBATCH --time=4:00:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# Define the target directory
# Default projects root (can be overridden by first script argument)
PROJECTS_ROOT="${1:-/home/jolle/mmdetection3d/projects}"

# Subfolders to process under the projects root
SUBDIRS=(fcos3d pointpillars pointpillars4x8 test cmt_fedavg_all_val cmt_fedavg cmt)

# Dry-run by default. Pass --execute or -x as second arg to actually delete.
DRY_RUN=true
if [ "$2" = "--execute" ] || [ "$2" = "-x" ]; then
    DRY_RUN=false
fi

echo "Starting checkpoint cleanup at: $(date)"
echo "Projects root: $PROJECTS_ROOT"
echo "Dry run: $DRY_RUN"

removed_count=0
skipped_no_epochs=0

for sub in "${SUBDIRS[@]}"; do
    base="$PROJECTS_ROOT/$sub"
    [ -d "$base" ] || continue
    echo "Processing subdir: $base"

    # Iterate over immediate child directories (model runs)
    for run_dir in "$base"/*; do
        [ -d "$run_dir" ] || continue
        # Find epoch_*.pth files and extract epoch numbers
        mapfile -t epoch_files < <(find "$run_dir" -maxdepth 1 -type f -name 'epoch_*.pth' -printf '%f\n')
        if [ ${#epoch_files[@]} -eq 0 ]; then
            ((skipped_no_epochs++))
            continue
        fi

        max_epoch=-1
        for ef in "${epoch_files[@]}"; do
            # ef like epoch_123.pth
            num=${ef#epoch_}
            num=${num%.pth}
            if [[ "$num" =~ ^[0-9]+$ ]]; then
                if (( num > max_epoch )); then
                    max_epoch=$num
                fi
            fi
        done

        if [ $max_epoch -lt 0 ]; then
            ((skipped_no_epochs++))
            continue
        fi

        keep_file="epoch_${max_epoch}.pth"
        echo "In $run_dir keeping: $keep_file (max epoch=$max_epoch)"

        # Remove any .pth that is not the keep_file
        while IFS= read -r -d '' pth; do
            fname=$(basename "$pth")
            if [ "$fname" = "$keep_file" ]; then
                continue
            fi
            if $DRY_RUN; then
                echo "DRY-RUN: would remove $pth"
            else
                if rm -v "$pth"; then
                    ((removed_count++))
                fi
            fi
        done < <(find "$run_dir" -maxdepth 1 -type f -name '*.pth' -print0)
    done
done

echo "Total files removed: $removed_count"
echo "Run dirs skipped (no epoch_*.pth files): $skipped_no_epochs"

echo "Cleanup finished at: $(date)"
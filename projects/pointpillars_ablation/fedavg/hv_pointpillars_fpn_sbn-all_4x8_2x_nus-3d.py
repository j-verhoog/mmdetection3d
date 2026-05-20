_base_ = [
    '../_base_/models/hv_pointpillars_fpn_nus.py',
    '../_base_/datasets/nus-3d.py', '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

# 1. Tell MMDet3D to import your CMT project
custom_imports = dict(
    imports=['projects.pointpillars_ablation.hooks.disable_gt_sampling', 'projects.pointpillars_ablation.hooks.force_stop'], 
    allow_failed_imports=False)

custom_hooks = [
    dict(
        type='DisableGTImportanceHook',
        disable_after_epoch=999          # old placeholder; does not affect anything
    ),
    # Add this new hook:
    dict(
        type='ForceStopHook',
        stop_epoch=999  # Placeholder; will be overwritten by sbatch
    )
]
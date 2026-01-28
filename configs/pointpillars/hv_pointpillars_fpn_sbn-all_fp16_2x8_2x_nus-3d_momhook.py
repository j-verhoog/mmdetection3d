_base_ = './hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py'
data = dict(samples_per_gpu=2, workers_per_gpu=2)
# fp16 settings, the loss scale is specifically tuned to avoid Nan
fp16 = dict(loss_scale=32.)
custom_imports = dict(
    imports=['projects.pointpillars.dua_momentum_hook'],
    allow_failed_imports=False
)

custom_hooks = [
    dict(
        type='DUABNMomentumHook',
        start_momentum=0.01,
        end_momentum=0.0001,
        by_epoch=False
    )
]

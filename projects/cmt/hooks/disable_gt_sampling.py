from mmcv.runner.hooks import HOOKS, Hook

@HOOKS.register_module()
class DisableGTImportanceHook(Hook):
    def __init__(self, disable_after_epoch=15):
        self.disable_after_epoch = disable_after_epoch

    def before_train_epoch(self, runner):
        if runner.epoch >= self.disable_after_epoch:
            # Handle the CBGS nesting: runner.dataset is likely a CBGSDataset
            dataset = runner.data_loader.dataset
            
            # CBGS wraps the real dataset in .dataset
            if hasattr(dataset, 'dataset'):
                real_dataset = dataset.dataset
            else:
                real_dataset = dataset

            # Access the pipeline transforms
            if hasattr(real_dataset, 'pipeline'):
                for transform in real_dataset.pipeline.transforms:
                    # Target 'UnifiedObjectSample' (Line 48 in your config)
                    if 'ObjectSample' in type(transform).__name__:
                        # Setting db_sampler to None is the standard way to skip it
                        transform.db_sampler = None
                        # If the transform has a specific 'enabled' flag, toggle it
                        if hasattr(transform, 'sample_2d'):
                            transform.sample_2d = False
                        
                        runner.logger.info(f"Epoch {runner.epoch}: GT Sampling (Line 48) has been DISABLED.")
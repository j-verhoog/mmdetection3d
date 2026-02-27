import math
import numpy as np
from mmcv.runner import HOOKS, Hook

class SamplingDisablerProxy:
    """
    A wrapper that sits around the real DB Sampler.
    It passes all attribute access to the real sampler (so nothing breaks),
    but it intercepts 'sample_all' to return None (disabling sampling).
    """
    def __init__(self, original_sampler):
        self._original_sampler = original_sampler

    def sample_all(self, *args, **kwargs):
        # We return None. This tells UnifiedObjectSample that no samples 
        # were found, so it skips the mixing logic cleanly.
        return None

    def __getattr__(self, name):
        # If code asks for 'db_infos', 'data_root', etc., 
        # we fetch it from the real sampler.
        return getattr(self._original_sampler, name)


@HOOKS.register_module()
class DisableGTCBGSAndAlignScheduleHook(Hook):
    """
    Safely disables CBGS and GT Sampling at a specified epoch, 
    and dynamically recalculates the training schedule to align 
    learning rate decay (like CosineAnnealing) with the new epoch length.
    """
    priority = 'HIGH'
    def __init__(self, disable_after_epoch=15):
        self.disable_after_epoch = disable_after_epoch
        self._has_disabled = False

    def before_train_epoch(self, runner):
        if runner.epoch >= self.disable_after_epoch and not self._has_disabled:
            runner.logger.info(f"\n--- [DisableGTCBGSAndAlignScheduleHook] Triggered at Epoch {runner.epoch} ---")
            
            dataset = runner.data_loader.dataset
            
            # =========================================================
            # STEP 1: Disable CBGS & Update PyTorch Sampler Math
            # =========================================================
            if type(dataset).__name__ == 'CustomCBGSDataset' or hasattr(dataset, 'sample_indices'):
                old_dataset_len = len(dataset)
                original_len = len(dataset.dataset)
                
                # 1A. Overwrite biased indices with standard 1-to-1 mapping
                dataset.sample_indices = list(range(original_len))
                new_dataset_len = len(dataset)
                
                # 1B. Reset the 'flag' array for MMCV GroupSampler compatibility
                if hasattr(dataset, 'flag') and hasattr(dataset.dataset, 'flag'):
                    dataset.flag = np.array(
                        [dataset.dataset.flag[ind] for ind in dataset.sample_indices],
                        dtype=np.uint8)
                        
                runner.logger.info(f"  -> CBGS Disabled: Dataset length shrunk from {old_dataset_len} to {new_dataset_len}.")
                
                # 1C. SAFELY UPDATE PYTORCH SAMPLER (Crucial for preventing crashes)
                sampler = runner.data_loader.sampler
                if hasattr(sampler, 'num_samples'):
                    num_replicas = getattr(sampler, 'num_replicas', 1)
                    drop_last = getattr(sampler, 'drop_last', False)
                    
                    # Recalculate how many samples this specific GPU is responsible for
                    if drop_last and new_dataset_len % num_replicas != 0:
                        sampler.num_samples = math.ceil((new_dataset_len - num_replicas) / num_replicas)
                    else:
                        sampler.num_samples = math.ceil(new_dataset_len / num_replicas)
                    
                    sampler.total_size = sampler.num_samples * num_replicas
                    runner.logger.info(f"  -> PyTorch Sampler updated. New num_samples per GPU: {sampler.num_samples}")

            # =========================================================
            # STEP 2: Disable GT Database Sampling
            # =========================================================
            # Handle CBGS wrapping to find the raw pipeline
            real_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
            
            found_gt_sampler = False
            if hasattr(real_dataset, 'pipeline'):
                for transform in real_dataset.pipeline.transforms:
                    if 'UnifiedObjectSample' in type(transform).__name__:
                        
                        if not isinstance(transform.db_sampler, SamplingDisablerProxy):
                            transform.db_sampler = SamplingDisablerProxy(transform.db_sampler)
                            found_gt_sampler = True
                        
                        if hasattr(transform, 'sample_2d'):
                            transform.sample_2d = False

            if found_gt_sampler:
                runner.logger.info("  -> GT Sampling successfully disabled via DisablerProxy.")
            else:
                runner.logger.warning("  -> Warning: Could not find UnifiedObjectSample in pipeline.")

            # =========================================================
            # STEP 3: Recalculate and Align Optimizer Schedule
            # =========================================================
            old_max_iters = runner.max_iters
            current_iter = runner.iter
            
            # The dataloader's __len__ accurately reflects the new sampler math we just updated
            new_batches_per_epoch = len(runner.data_loader)
            remaining_epochs = runner.max_epochs - runner.epoch
            
            # Math: Steps already taken + (Remaining epochs * New smaller batch count)
            new_max_iters = current_iter + (remaining_epochs * new_batches_per_epoch)
            
            # Overwrite MMCV runner's global state so LrUpdaterHook calculates the cosine curve properly
            runner.max_iters = new_max_iters
            
            runner.logger.info(f"  -> Schedule Aligned: max_iters updated from {old_max_iters} to {new_max_iters}.")
            runner.logger.info(f"  -> Breakdown: {current_iter} steps taken + ({remaining_epochs} epochs remaining * {new_batches_per_epoch} batches/epoch).")
            runner.logger.info("--- [DisableGTCBGSAndAlignScheduleHook] Execution Complete ---\n")

            # Lock the hook so it only executes once
            self._has_disabled = True
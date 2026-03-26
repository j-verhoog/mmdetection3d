import os
import torch
from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class FisherComputationHook(Hook):
    """
    Hook to compute and save the empirical Fisher Information Matrix (diagonal)
    at the end of training. Required for FedMC aggregation.
    """
    def __init__(self, save_path, num_batches=100):
        self.save_path = save_path
        self.num_batches = num_batches
def after_train(self, runner):
        runner.logger.info(f"Computing Fisher Information Matrix (up to {self.num_batches} batches)...")
        model = runner.model
        optim_wrapper = runner.optim_wrapper
        
        model.train()
        
        fisher_diag = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_diag[name] = torch.zeros_like(param.data)

        dataloader = runner.train_dataloader
        batch_count = 0
        
        for data_batch in dataloader:
            if batch_count >= self.num_batches:
                break
                
            optim_wrapper.zero_grad()
            
            with optim_wrapper.optim_context(model):
                data = model.data_preprocessor(data_batch, training=True)
                if isinstance(data, dict):
                    losses = model(**data, mode='loss')
                elif isinstance(data, (list, tuple)):
                    losses = model(*data, mode='loss')
                
                parsed_losses, _ = model.parse_losses(losses)
                loss = parsed_losses['loss']
            
            optim_wrapper.backward(loss)
            
            # ACCUMULATE ONLY (Don't divide yet)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher_diag[name] += param.grad.data ** 2
            
            batch_count += 1
            if batch_count % 10 == 0 or batch_count == self.num_batches:
                runner.logger.info(f"Fisher computation: {batch_count} batches processed.")

        # DIVIDE BY ACTUAL BATCHES PROCESSED
        if batch_count > 0:
            for name in fisher_diag:
                fisher_diag[name] /= batch_count
            runner.logger.info(f"Averaged Fisher Information over {batch_count} actual batches.")
        else:
            runner.logger.warning("No batches processed for Fisher computation!")

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        cpu_fisher = {k: v.cpu() for k, v in fisher_diag.items()}
        torch.save({'state_dict': cpu_fisher}, self.save_path)
        runner.logger.info(f"Saved Fisher Information Matrix to {self.save_path}")
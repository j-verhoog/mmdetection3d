import math
from mmcv.runner import HOOKS, Hook
from torch.nn.modules.batchnorm import _BatchNorm

@HOOKS.register_module()
class DUABNMomentumHook(Hook):
    def __init__(self, start_momentum=0.01, end_momentum=0.0001, by_epoch=False):
        self.start = float(start_momentum)
        self.end = float(end_momentum)
        self.by_epoch = bool(by_epoch)

    def _cur_momentum(self, runner):
        t = runner.epoch if self.by_epoch else runner.iter
        T = runner.max_epochs if self.by_epoch else runner.max_iters
        if T <= 1:
            return self.end
        p = min(max(t / (T - 1), 0.0), 1.0)  # 0 -> 1 over training
        return self.start * math.exp(math.log(self.end / self.start) * p)

    def before_train_iter(self, runner):
        m = self._cur_momentum(runner)
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        for mod in model.modules():
            if isinstance(mod, _BatchNorm) or "BatchNorm" in mod.__class__.__name__:
                mod.momentum = m


# momemtum 0.0001:
# ~7k batches → weight halves
# ~10k batches → main “memory mass”
# ~30k batches → effectively gone
from torch.optim import Optimizer


class WeightEMA (object):
    """
    Exponential moving average weight optimizer for mean teacher model
    """
    def __init__(self, params, src_params, alpha=0.999):
        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

        for p, src_p in zip(self.params, self.src_params):
            p[:] = src_p[:].detach()

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.params, self.src_params):
            p.mul_(self.alpha)
            p.add_(src_p.detach() * one_minus_alpha)

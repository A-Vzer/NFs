import torch.nn as nn
from models.WaveletFlow.step import Step


class MultiStep(nn.Module):
    def __init__(self, params, shape, level, conditional):
        super().__init__()
        self.steps = nn.ModuleList()
        self.n_steps = params.stepsPerResolution[level]
        for n in range(self.n_steps):
            self.steps.append(Step(params, shape, conditional, level))

    def forward(self, x, logdet, conditioning=None, reverse=False):
        z = x
        if reverse:
            for step in reversed(self.steps):
                z, logdet = step.forward(z, logdet, conditioning)
        else:
            for step in self.steps:
                z, logdet = step.forward(z, logdet, conditioning)
        return z, logdet

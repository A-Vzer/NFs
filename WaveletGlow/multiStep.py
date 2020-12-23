import torch
import torch.nn as nn
from Glow.model import FlowNet


class MultiStep(nn.Module):
    def __init__(self, params, data_shape, level, conditional):
        super().__init__()
        self.steps = []
        self.n_steps = params.stepsPerResolution[level]
        for n in range(self.n_steps):
            self.steps.append(
                FlowNet(params, data_shape, level=level, conditional=conditional))

    def forward(self, x, logdet, conditioning=None, reverse=False):
        z = x
        if reverse:
            for step in reversed(self.steps):
                z, logdet = step.forward(z, logdet, conditioning)
        else:
            for step in self.steps:
                z, logdet = step.forward(z, logdet, conditioning)
        return z, logdet

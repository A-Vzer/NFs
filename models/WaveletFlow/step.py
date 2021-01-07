import torch
import torch.nn as nn
from models.Glow.model import FlowStep
from modules.Coupling.resCoupling import Residual


class Step(nn.Module):
    def __init__(self, params, shape, conditional, level):
        C, H, W = shape
        super().__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        for _ in range(params.K):
            self.layers.append(Flow(C, params, conditional, level))
            self.output_shapes.append([-1, C, H, W])

    def forward(self, x, logdet, conditioning=None, reverse=False):
        z = x
        if reverse:
            for layer in reversed(self.layers):
                z, logdet = layer.forward(z, logdet, conditioning)
        else:
            for layer in self.layers:
                z, logdet = layer.forward(z, logdet, conditioning)
        return z, logdet


class Flow(FlowStep):
    def __init__(self, C, params, conditional, level):
        params.coupling = None
        super().__init__(params, C)
        self.coupling = Residual(params, conditional, level)


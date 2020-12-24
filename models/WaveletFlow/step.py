import torch
import torch.nn as nn
from models.Glow.model import FlowStep
from modules.Coupling.resCoupling import Residual


class Step(nn.Module):
    def __init__(self, params, conditional, level):
        C, H, W = params.imShape
        super().__init__()
        for _ in range(self.K):
            self.layers.append(Flow(C, params, conditional, level))
            self.output_shapes.append([-1, C, H, W])


class Flow(FlowStep):
    def __init__(self, C, params, conditional, level):
        super().__init__(params, C)
        self.coupling = Residual(params, conditional, level)

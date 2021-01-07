import torch
import torch.nn as nn
from modules.Wavelets.Haar.split import Split
from modules.Wavelets.Haar.squeeze import Squeeze
from utilities.utils import to_attributes
import numpy as np


class SqueezeSplit(nn.Module):
    def __init__(self, compensate=False, device=None):
        super().__init__()
        self.compensate = compensate
        self.split = Split()
        self.haar_squeeze = Squeeze(device)

    def forward(self, full_res):
        haar_squeeze, _ = self.haar_squeeze.forward(full_res)
        base, details = self.split.forward(haar_squeeze)
        ldj = 0

        if self.compensate:
            C = base.size(1)
            H = base.size(2)
            W = base.size(3)
            n_dim = torch.tensor(H * W * C, dtype=torch.float, requires_grad=False)

            base = base * 0.5  # haar base is 2*average
            ldj = np.log(0.5) * n_dim

        haar_representation = to_attributes({})
        haar_representation.details = details
        haar_representation.base = base
        haar_representation.ldj = ldj

        return haar_representation


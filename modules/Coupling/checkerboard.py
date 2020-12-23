import torch
import torch.nn as nn
from models.ConvNet import convNet
from utilities.utils import split_feature, Rescale


class Checkerboard:
    def __init__(self, in_channels, out_channels, hidden_channels, flow):
        super().__init__()
        self.device = flow.device
        self.block = convNet.ConvNet(in_channels, out_channels, hidden_channels, self.device)
        self.rescale = nn.utils.weight_norm(Rescale(in_channels))

    def get_param(self, x):
        H = x.size(2)
        W = x.size(3)
        checkerboard = [[((i % 2) + j) % 2 for j in range(W)] for i in range(H)]
        mask = torch.tensor(checkerboard)
        mask = mask.view(1, 1, H, W).to(self.device)
        z1 = x * mask
        z2 = x * (1 - mask)
        h = self.block.cnn(z1)
        s, t = split_feature(h, "cross")
        s = self.rescale(torch.tanh(s))
        s = s * (1 - mask)
        t = t * (1 - mask)
        return s, t, z1, z2, mask

    def forward(self, x, logdet, reverse=False):
        s, t, z1, z2, mask = self.get_param(x)
        exp_s = s.exp()
        if reverse:
            z2 = z2 * exp_s - t
            logdet = -torch.sum(s, dim=[1, 2, 3]) + logdet
        else:
            z2 = (z2 + t) * exp_s
            logdet = torch.sum(s, dim=[1, 2, 3]) + logdet
        z = z1 * mask + z2 * (1 - mask)
        return z, logdet

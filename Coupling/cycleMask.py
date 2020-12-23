import torch
import torch.nn as nn
from utils import Rescale
from Glow import tools
from ConvNet.convNet import ConvNet
import sys
import numpy

numpy.set_printoptions(threshold=sys.maxsize)


class CycleMask(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, device, cycles=1):
        super().__init__()
        self.cycles = cycles
        # self.device = device
        self.block = ConvNet(in_channels, out_channels, hidden_channels, device)
        self.rescale = nn.utils.weight_norm(Rescale(in_channels))

    def get_mask(self, quadrant, c, h, w):
        b = torch.zeros((1, c, h, w), dtype=torch.float)
        split_h = h // 2
        split_w = w // 2
        if quadrant == 0:
            b[:, :, :split_h, :split_w] = 1.
        elif quadrant == 1:
            b[:, :, :split_h, split_w:] = 1.
        elif quadrant == 2:
            b[:, :, split_h:, split_w:] = 1.
        elif quadrant == 3:
            b[:, :, split_h:, :split_w] = 1.
        else:
            raise ValueError("Incorrect mask quadrant")
        return b

    def get_param(self, x, input_q, output_q):
        C, H, W = x.size(1), x.size(2), x.size(3)
        self.mask_in = self.get_mask(input_q, C, H, W).to(x.device)
        self.mask_out = self.get_mask(output_q, C, H, W).to(x.device)
        z1 = x * self.mask_in
        z2 = x * (1 - self.mask_in)
        h = self.block.cnn(z1)
        s, t = tools.split_feature(h, "cross")
        s = self.rescale(torch.tanh(s))
        return z1, z2, s, t

    def forwardStep(self, x, logdet, q_in, q_out, reverse=False):
        z1, z2, s, t = self.get_param(x, q_in, q_out)
        s = s * self.mask_out
        t = t * self.mask_out
        exp_s = s.exp()
        if reverse:
            z2 = z2 * exp_s - t
            logdet = -torch.sum(s, dim=[1, 2, 3]) + logdet
        else:
            z2 = (z2 + t) * exp_s
            logdet = torch.sum(s, dim=[1, 2, 3]) + logdet
        z = z1 * self.mask_in + z2 * (1 - self.mask_in)
        return z, logdet

    def forward(self, z, logdet, reverse=False):
        if reverse:
            for i in range(self.cycles):
                z, logdet = self.forwardStep(z, logdet, 0, 1, True)
                z, logdet = self.forwardStep(z, logdet, 1, 2, True)
                z, logdet = self.forwardStep(z, logdet, 2, 3, True)
                z, logdet = self.forwardStep(z, logdet, 3, 0, True)
        else:
            for i in range(self.cycles):
                z, logdet = self.forwardStep(z, logdet, 0, 1)
                z, logdet = self.forwardStep(z, logdet, 1, 2)
                z, logdet = self.forwardStep(z, logdet, 2, 3)
                z, logdet = self.forwardStep(z, logdet, 3, 0)
        return z, logdet

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from utilities.utils import split_feature, Rescale
from models.ConvNet.convNet import ConvNet
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import cv2
import numpy as np


class CycleMask(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, device, cycles=2):
        super().__init__()
        self.cycles = cycles
        self.block = ConvNet(in_channels, out_channels, hidden_channels, device)
        self.rescale = nn.utils.weight_norm(Rescale(in_channels))

    def get_mask(self, x):
        mean = torch.mean(x, dim=1)
        std = torch.std(x, dim=1)
        thres1 = torch.unsqueeze(mean + std, dim=1)
        thres2 = torch.unsqueeze(mean - std, dim=1)
        b1 = torch.unsqueeze(thres1, dim=1)
        b2 = torch.unsqueeze(thres2, dim=1)
        mask1 = torch.tensor(np.where(x < b1, 0, x), device=x.device)
        mask2 = torch.tensor(np.where(b1 < x < b2, 0, x), device=x.device)
        mask3 = torch.tensor(np.where(x > b2, 0, x), device=x.device)
        return mask1, mask2, mask3

    def get_param(self, x, n):
        self.mask1, self.mask2, self.mask3 = self.get_mask(x)
        if n == 0:
            z1 = x * self.mask1
            z2 = x * (1 - self.mask1)
            self.mask_in = self.mask1
            self.mask_out = self.mask2 + self.mask3
        if n == 1:
            z1 = x * self.mask2
            z2 = x * (1 - self.mask2)
            self.mask_in = self.mask2
            self.mask_out = self.mask1 + self.mask3
        if n == 2:
            z1 = x * self.mask3
            z2 = x * (1 - self.mask3)
            self.mask_in = self.mask3
            self.mask_out = self.mask1 + self.mask2

        h = self.block.cnn(z1)
        s, t = split_feature(h, "cross")
        s = self.rescale(torch.tanh(s))
        return z1, z2, s, t

    def forwardStep(self, x, logdet, n, reverse=False):
        z1, z2, s, t = self.get_param(x, n)
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
                z, logdet = self.forwardStep(z, logdet, 0, True)
                z, logdet = self.forwardStep(z, logdet, 1, True)
                z, logdet = self.forwardStep(z, logdet, 2, True)
        else:
            for i in range(self.cycles):
                z, logdet = self.forwardStep(z, logdet, 0)
                z, logdet = self.forwardStep(z, logdet, 1)
                z, logdet = self.forwardStep(z, logdet, 2)
        return z, logdet
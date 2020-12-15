import torch
import torch.nn as nn
from Glow import tools
from Glow import layers as l
from utils import Rescale
from ConvNet.convNet import ConvNet
import torch.nn.functional as F


class Additive(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.block = ConvNet(in_channels, out_channels, hidden_channels)

    def forward(self, x, logdet, reverse=True):
        z1, z2 = tools.split_feature(x, 'split')
        if reverse:
            z2 = z2 - self.block(z1)
            z = torch.cat((z1, z2), dim=1)
        else:
            z2 = z2 + self.block(z1)
            z = torch.cat((z1, z2), dim=1)
        return z, logdet


class Affine(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.block = ConvNet(in_channels, out_channels, hidden_channels)

    def get_param(self, x):
        z1, z2 = tools.split_feature(x, "split")
        h = self.block.cnn(z1)
        s, t = tools.split_feature(h, "cross")
        s = torch.sigmoid(s + 2.0)
        return s, t, z1, z2

    def forward(self, x, logdet, reverse=True):
        s, t, z1, z2 = self.get_param(x)
        if reverse:
            s = torch.sigmoid(s + 2.0)
            z2 = z2 / s
            z2 = z2 - t
            logdet = -torch.sum(torch.log(s), dim=[1, 2, 3]) + logdet
        else:
            s, t, z1, z2 = self.get_param(x)
            z2 = z2 + t
            z2 = z2 * s
            logdet = torch.sum(torch.log(s), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)
        return z, logdet


class SoftCheckerboard(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, device):
        super().__init__()
        self.block = ConvNet(in_channels, out_channels, hidden_channels, device)
        self.device = device
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
        s, t = tools.split_feature(h, "cross")
        s = self.rescale(torch.tanh(s))
        s = s * (1 - mask)
        t = t * (1 - mask)
        return s, t, z1, z2, mask

    def forward(self, x, logdet, reverse=True):
        s, t, z1, z2, mask = self.get_param(x)
        exp_s = s.exp()
        if reverse:
            s, t, z1, z2, mask = self.get_param(x)
            z2 = z2 * exp_s - t
            logdet = -torch.sum(s, dim=[1, 2, 3]) + logdet
        else:
            s, t, z1, z2, mask = self.get_param(x)
            z2 = (z2 + t) * exp_s

            logdet = torch.sum(s, dim=[1, 2, 3]) + logdet
        z = z1 * mask + z2 * (1 - mask)
        return z, logdet
    
    

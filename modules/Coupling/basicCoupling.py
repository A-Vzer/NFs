import torch
import torch.nn as nn
from utilities.utils import split_feature
from models.ConvNet import openai_conv
from models.ConvNet import convNet


class Additive(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, device):
        super().__init__()
        self.block = convNet.ConvNet(in_channels, out_channels, hidden_channels, device)

    def forward(self, x, logdet, reverse=False):
        z1, z2 = split_feature(x, 'split')
        if reverse:
            z2 = z2 - self.block(z1)
            z = torch.cat((z1, z2), dim=1)
        else:
            z2 = z2 + self.block(z1)
            z = torch.cat((z1, z2), dim=1)
        return z, logdet


class Affine(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, device):
        super().__init__()
        self.block = convNet.ConvNet(in_channels, out_channels, hidden_channels, device)

    def get_param(self, x):
        z1, z2 = split_feature(x, "split")
        h = self.block(z1)
        s, t = split_feature(h, "cross")
        s = torch.sigmoid(s + 2.0)
        return s, t, z1, z2

    def forward(self, x, logdet, reverse=False):
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

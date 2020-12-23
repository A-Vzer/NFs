import torch
import torch.nn as nn
from Glow import tools
from Glow import layers as l
from utils import Rescale
from ConvNet import convNet, openai_conv

import torch.nn.functional as F


class Additive(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, flow):
        super().__init__()
        self.block = convNet.ConvNet(in_channels, out_channels, hidden_channels, flow.device)

    def forward(self, x, logdet, reverse=False):
        z1, z2 = tools.split_feature(x, 'split')
        if reverse:
            z2 = z2 - self.block(z1)
            z = torch.cat((z1, z2), dim=1)
        else:
            z2 = z2 + self.block(z1)
            z = torch.cat((z1, z2), dim=1)
        return z, logdet


class Affine(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, flow):
        super().__init__()
        self.device = flow.device
        self.level = flow.level
        self.conditional = flow.conditional
        self.modelName = flow.modelName
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.width = flow.widths[self.level]
        self.k = flow.k
        self.bias = flow.biases[self.level]
        self.log_scale_factor = flow.logscale_factor
        self.use_logscale = flow.use_logscale
        self.block = convNet.ConvNet(in_channels, out_channels, hidden_channels, flow.device)

    def get_param(self, x, conditioning=None):
        z1, z2 = tools.split_feature(x, "split")
        if self.modelName == 'waveletglow':
            if conditioning is not None:
                self.block = openai_conv.OpenAiConvZero(self.k, z2.size(1) * 2, 1, self.bias,
                                                        self.use_logscale, self.log_scale_factor, self.device)
                z1_conditioned = torch.cat([z1, conditioning], dim=1)
                h = self.block(z1_conditioned)
            else:
                self.block = openai_conv.OpenAiConvZero(self.k, z2.size(1) * 2, 1, self.bias, self.use_logscale,
                                                        self.log_scale_factor, self.device)
                h = self.block(z1)
        else:
            h = self.block(z1)
        s, t = tools.split_feature(h, "cross")
        s = torch.sigmoid(s + 2.0)
        return s, t, z1, z2

    def forward(self, x, logdet, conditioning=None, reverse=False):
        s, t, z1, z2 = self.get_param(x, conditioning)
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
        s, t = tools.split_feature(h, "cross")
        s = self.rescale(torch.tanh(s))
        s = s * (1 - mask)
        t = t * (1 - mask)
        return s, t, z1, z2, mask

    def forward(self, x, logdet, reverse=False):
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
    
    

import torch
import torch.nn as nn
from models.ConvNet.openai_conv import *
from utilities.utils import split_feature, Rescale


class Residual(nn.Module):
    def __init__(self, params, conditional, level):
        super().__init__()
        self.conditional = conditional
        self.edge_bias = params.spatialBiasing[level]
        self.normalize = params.normalize
        self.log_scale_factor = params.zero_logscale_factor
        self.use_logscale = params.zero_use_logscale
        self.rescale = nn.utils.weight_norm(Rescale(2))
        self.k = params.kernel
        n_res_blocks = params.n_res_blocks
        width = params.convWidth[level]

        self.conv_in = OpenAiConv(1, width, 1, self.edge_bias, self.normalize)
        self.res_blocks = []
        for n in range(n_res_blocks):
            self.res_blocks.append(ResidualBlock(self.k, width, self.edge_bias, self.normalize))

        self.conv_out = None

    def get_param(self, x, conditioning=None):
        z1, z2 = split_feature(x, "split")
        z2_out = z2.size(1) * 2
        if self.conv_out is None:
            self.conv_out = OpenAiConv(self.k, z2_out, 1, self.edge_bias, self.normalize)

        if conditioning is not None:
            self.block = OpenAiZeroConv(self.k, z2_out, 1, self.edge_bias, self.use_logscale, self.log_scale_factor)
            z1_c = torch.cat([z1, conditioning], dim=1)
        else:
            z1_c = z1
            self.block = OpenAiZeroConv(self.k, z2_out, 1, self.edge_bias, self.use_logscale, self.log_scale_factor)
        h = self.block(z1_c)
        s, t = split_feature(h, "cross")
        s = torch.tanh(s)
        return s, t, z1, z2

    def forward(self, x, logdet, cond, reverse=False):
        s, t, z1, z2 = self.get_param(x, cond)

        exp_s = s.exp()
        if reverse:
            z2 = z2 * exp_s - t
            logdet = -torch.sum(s, dim=[1, 2, 3]) + logdet
        else:
            z2 = (z2 + t) * exp_s
            logdet = torch.sum(s, dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)
        return z, logdet


class ResidualBlock(nn.Module):
    def __init__(self, k, width, edge_bias, normalize):
        super().__init__()
        self.conv1 = OpenAiConv(k, width, 1, edge_bias, normalize)
        self.conv2 = OpenAiConv(1, width, 1, edge_bias, 'none', normalize)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv2 = nn.relu(x + conv2)

        return conv2


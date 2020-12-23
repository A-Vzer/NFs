import torch
import torch.nn as nn
import torch.nn.functional as F
from Glow.layers import ActNorm2d
from WaveletGlow.tools import edge_bias
import math


class OpenAiConvZero(nn.Module):
    """
    open ai style conv
    act norm will be included in the future
    """

    def __init__(self, k_size, n_out, stride, edge_bias, use_logscale, logscale_f, device):
        # setup naming and scoping
        super().__init__()

        self.w = None
        self.b = None
        self.actnorm = None
        self.device = device
        self.k_size = k_size
        self.n_out = n_out
        self.stride = stride
        self.edge_bias = edge_bias
        self.use_logscale = use_logscale
        self.logscale_factor = logscale_f

    def __call__(self, x, init=False):
        # setup edge bias
        padding = 'SAME'
        if self.edge_bias:
            x = edge_bias(x, self.k_size)
            padding = 'VALID'

        # setup variables if needed
        if self.w is None:
            c_in = x.size(1)
            self.w = torch.zeros([self.n_out, c_in, self.k_size, self.k_size], requires_grad=True, device=self.device)
            self.b = torch.zeros([self.n_out], requires_grad=True, device=self.device)
            if self.use_logscale:
                self.logscale = torch.zeros([1, self.n_out, 1, 1], requires_grad=True, device=self.device)

        if padding == 'SAME':
            p = math.ceil(((self.stride - 1) * x.size(2) - self.stride + self.k_size) / 2)
            z = F.conv2d(x, self.w, stride=self.stride, padding=p)
        else:
            z = F.conv2d(x, self.w, stride=self.stride)
        if self.use_logscale:
            z *= torch.exp(self.logscale * self.logscale_factor)

        return z

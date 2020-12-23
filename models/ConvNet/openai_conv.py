import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities.utils import edge_bias
from modules.layers import ActNorm2d
import math


class OpenAiConv(nn.Module):
    def __init__(self, k_size, n_out, stride, edge_bias, activation='relu', norm=False):
        # setup naming and scoping
        super().__init__()
        self.w = None
        self.b = None
        self.actnorm = None
        self.k_size = k_size
        self.n_out = n_out
        self.stride = stride
        self.edge_bias = edge_bias
        self.activation = activation
        self.normalize = norm

    def __call__(self, x, init=False):
        # setup edge bias
        pad = True
        if self.edge_bias:
            x = edge_bias(x, self.k_size)
            pad = False

        # setup variables if needed
        if self.w is None:
            C = x.size(1)
            self.actnorm = ActNorm2d(C)
            self.w = torch.normal(0.0, 0.05, size=[self.n_out, C, self.k_size, self.k_size], requires_grad=True,
                                  device=x.device)
            if not self.normalize:
                self.b = torch.zeros([self.n_out], requires_grad=True, device=x.device)
        if pad:
            p = math.ceil(((self.stride - 1) * x.size(2) - self.stride + self.k_size) / 2)
            z = F.conv2d(x, self.w, stride=self.stride, padding=p)
        else:
            z = F.conv2d(x, self.w, stride=self.stride)

        if self.normalize:
            z, _ = self.actnorm(x)
        else:
            z += self.b

        if self.activation == 'relu':
            z = nn.relu(z)

        return z


class OpenAiZeroConv(nn.Module):
    def __init__(self, k_size, n_out, stride, edge_bias, use_logscale, logscale_f):
        # setup naming and scoping
        super().__init__()
        self.w = None
        self.b = None
        self.actnorm = None
        self.k_size = k_size
        self.n_out = n_out
        self.stride = stride
        self.edge_bias = edge_bias
        self.use_logscale = use_logscale
        self.logscale_factor = logscale_f

    def __call__(self, x, init=False):
        # setup edge bias
        pad = True
        if self.edge_bias:
            x = edge_bias(x, self.k_size)
            pad = False

        # setup variables if needed
        if self.w is None:
            c_in = x.size(1)
            self.w = torch.zeros([self.n_out, c_in, self.k_size, self.k_size], requires_grad=True, device=x.device)
            self.b = torch.zeros([self.n_out], requires_grad=True, device=x.device)
            if self.use_logscale:
                self.logscale = torch.zeros([1, self.n_out, 1, 1], requires_grad=True, device=x.device)

        if pad:
            p = math.ceil(((self.stride - 1) * x.size(2) - self.stride + self.k_size) / 2)
            z = F.conv2d(x, self.w, stride=self.stride, padding=p)
        else:
            z = F.conv2d(x, self.w, stride=self.stride)
        if self.use_logscale:
            z *= torch.exp(self.logscale * self.logscale_factor)

        return z

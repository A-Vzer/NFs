import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Squeeze(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.cached_kernel = None
        self.cached_conv_kernel = None
        self.device = device

    def get_haar_kernel(self, n_channels):
        '''
        generates kernel for haar wavelet downsampling
        kernel should be inverse of itself
        '''
        if self.cached_kernel is not None:
            return self.cached_kernel

        kernel = []
        for n in range(n_channels):
            front_padding = [0.0, 0.0, 0.0, 0.0] * n
            back_padding = [0.0, 0.0, 0.0, 0.0] * (n_channels - n - 1)

            row = front_padding + [0.5, 0.5, 0.5, 0.5] + back_padding
            kernel.append(row)
            row = front_padding + [0.5, -0.5, 0.5, -0.5] + back_padding
            kernel.append(row)
            row = front_padding + [0.5, 0.5, -0.5, -.5] + back_padding
            kernel.append(row)
            row = front_padding + [0.5, -0.5, -0.5, 0.5] + back_padding
            kernel.append(row)

        # invert to prepare for conversion to 1x1 conv kernel
        kernel = torch.transpose(kernel, 0, 1)

        # expand to valid 1x1 conv kernel
        kernel = torch.unsqueeze(kernel, 0)
        kernel = torch.unsqueeze(kernel, 0)
        # cache
        self.cached_kernel = kernel

        return self.cached_kernel

    def get_conv_haar_kernel(self, n_channels):
        if self.cached_conv_kernel is not None:
            return self.cached_conv_kernel

        k = np.zeros(shape=[4 * n_channels, n_channels, 2, 2])

        for i in range(n_channels):
            k[i * 4 + 0, i, :, :] = [[0.5, 0.5], [0.5, 0.5]]
            k[i * 4 + 1, i, :, :] = [[0.5, -0.5], [0.5, -0.5]]
            k[i * 4 + 2, i, :, :] = [[0.5, 0.5], [-0.5, -0.5]]
            k[i * 4 + 3, i, :, :] = [[0.5, -0.5], [-0.5, 0.5]]

        self.cached_conv_kernel = k

        return self.cached_conv_kernel

    def forward(self, x):
        return self.forward_conv(x)

    def forward_reshape(self, x):
        factor = 2
        C = x.size(1)
        H = x.size(2)
        W = x.size(3)

        # make sure we can divide nicely
        assert H % factor == 0 and W % factor == 0, '({},{}) not dividing by {} nicely'.format(H, W, factor)

        # get kernel for haar
        haar_kernel = self.get_haar_kernel(C)
        haar_kernel = torch.tensor(haar_kernel, dtype=torch.float, requires_grad=True, device=self.device)
        # reshape to add two auxillary dimensions
        y = torch.reshape(x, [-1, H // factor, factor, W // factor, factor, C])

        # transpose to move auxillary dimentions near the channel dimensions
        y = torch.transpose(y, [0, 5, 1, 3, 2, 4])

        # collapse auxillary dimensions into channel dimension
        y = torch.reshape(y, [-1, C * factor * factor, H // factor, W // factor])

        # apply haar downsampling
        y = F.conv2d(y, haar_kernel, stride=(1, 1), padding=0)

        return y, 0

    def forward_conv(self, x):
        factor = 2
        C = x.size(1)
        H = x.size(2)
        W = x.size(3)

        # make sure we can divide nicely
        assert H % factor == 0 and W % factor == 0, '({},{}) not dividing by {} nicely'.format(H, W, factor)

        # get kernel for haar
        haar_kernel = self.get_conv_haar_kernel(C)
        haar_kernel = torch.tensor(haar_kernel, dtype=torch.float, requires_grad=True, device=self.device)
        y = F.conv2d(x, haar_kernel, stride=(2, 2), padding=0)

        return y, 0

    def inverse(self, x):
        factor = 2
        C = x.size(1)
        H = x.size(2)
        W = x.size(3)

        # make sure n channels is divisible by 4
        assert C >= 4 and C % 4 == 0, '({}) channels must be divisible by 4'.format(c)

        # get kernel for haar
        haar_kernel = self.get_haar_kernel(C // 4)

        # apply haar downsampling inverse
        x = F.conv2d(x, haar_kernel, stride=(1, 1), padding=1)

        x = F.reshape(x, [-1, H, W, int(C / factor ** 2), factor, factor])
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = nn.reshape(x, [-1, int(H * factor), int(W * factor), int(C / (factor * factor))])
        return x

import torch.nn as nn
from modules.layers import Conv2d, Conv2dZeros


class ConvNet:
    def __init__(self, in_channels, out_channels, hidden_channels, device):
        self.cnn = self.st_net(in_channels, out_channels, hidden_channels).to(device)

    def st_net(self, in_channels, out_channels, hidden_channels):
        block = nn.Sequential(Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False),
                              Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1)), nn.ReLU(inplace=False),
                              Conv2dZeros(hidden_channels, out_channels))
        return block

    def __call__(self, x):
        return self.cnn(x)

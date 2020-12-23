import torch
import torch.nn as nn


class Split(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        C = x.size(1)

        assert C % 4 == 0, 'channels must be divisible by 4'

        n_actual = C // 4

        averages = []
        details = []

        for it in range(n_actual):
            idx = it * 4
            averages.append(x[:, idx:idx + 1, :, :])
            details.append(x[:, idx + 1:idx + 4, :, :])

        averages = torch.cat(averages, dim=1)
        details = torch.cat(details, dim=1)

        return averages, details

    def inverse(self, averages, details):
        slices = []
        n_actual = averages.size(1)
        for it in range(n_actual):
            threes = it * 3
            slices.append(averages[:, it:it + 1, :])
            slices.append(details[:, threes:threes + 3, :])

        reconstructed = torch.cat(slices, dim=1)
        return reconstructed
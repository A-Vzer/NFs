import torch


def standardize(x):
    n_bits = 8
    x = x * 255  # undo ToTensor scaling to [0,1]
    n_bins = 2 ** n_bits
    if n_bits < 8:
        x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5  # Scaled such that x lies in-between -0.5 and 0.5
    return x
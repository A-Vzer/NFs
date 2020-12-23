import torch
import torch.nn as nn
import torch.nn.functional as F


def edge_bias(x, k_size):
    """
    injects a mask into a tensor for 2d convs to indicate padding locations
    """

    pad_size = k_size // 2

    # manually pad data for conv
    x_padded = F.pad(x, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])

    # generate mask to indicated padded pixels
    # x_mask_inner = tf.zeros(shape=tf.shape(x))
    x_mask_inner = torch.zeros(x)
    x_mask_inner = x_mask_inner[:, 0:1, :, :]
    x_mask = F.pad(x_mask_inner, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], value=1.0)

    # combine into 1 tensor
    x_augmented = torch.cat([x_padded, x_mask], axis=1)

    return x_augmented

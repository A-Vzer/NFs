from Glow import model
from Glow import tools
import torch


class Parameters:
    def __init__(self):
        self.hiddenUnits = 512
        self.K = 32
        self.L = 3
        self.actNormScale = 1.0
        self.perm = "invconv"
        self.coupling = "affine"
        self.LU = True
        self.warmup = 5
        # zero means clipping off
        self.max_grad_clip = 0
        self.max_grad_norm = 0
        self.lr = 5e-4
        self.n_init_batch = 8
        self.y_classes = 10
        self.y_learn_top = False
        self.y_condition = False
        self.y_weight = 0.01
        self.loss = tools.loss
        self.weigh_decay = 5e-5
        self.lr_lambda = lambda epoch: min(1.0, (epoch + 1) / self.warmup)  # noqa
        self.initialize = True

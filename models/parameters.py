from models.WaveletFlow import model
from utilities.utils import loss
import torch
import torch.optim as optim
from models.WaveletFlow.conditionNet import ConditioningNetwork


class Parameters:
    def __init__(self, imShape, device):
        self.modelName = None
        self.model = None
        self.imShape = imShape
        self.device = device
        # zero means clipping is off
        self.max_grad_clip = 0
        self.max_grad_norm = 0
        self.LU = True
        self.warmup = 10
        self.n_init_batch = 32
        self.actNormScale = 1.0
        self.perm = "invconv"

    def compute_loss(self, x, y):
        return loss(self, x, y)

    def clip_gradients(self):
        if self.max_grad_clip > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.max_grad_clip)
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

    def sampler(self, bs, temp, n=10):
        pass
from models.Glow import model, tools
import torch
import torch.optim as optim
from models.model import Parameters


class GlowParameters(Parameters):
    def __init__(self, imShape, device):
        super().__init__(imShape, device)
        self.imShape = imShape
        self.modelName = 'glow'
        self.model = model.Glow(self).to(device)
        self.inChannels = 3
        self.hiddenChannels = 512
        self.K = 32
        self.L = 3
        self.perm = "invconv"
        self.coupling = "affine"
        self.lr = 1e-4
        self.y_classes = 10
        self.y_learn_top = False
        self.y_condition = False
        self.y_weight = 0.01
        self.weigh_decay = 1e-3
        self.lr_lambda = lambda epoch: min(1.0, (epoch + 1) / self.warmup)  # noqa
        self.initialize = True
        self.device = device
        self.optimizer = optim.Adamax(self.model.parameters(), lr=self.lr, weight_decay=self.weigh_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

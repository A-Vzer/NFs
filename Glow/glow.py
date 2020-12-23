from Glow import model
from Glow import tools
import torch
import torch.optim as optim


class Parameters:
    def __init__(self, imShape, device):
        self.imShape = imShape
        self.modelName = 'glow'
        self.inChannels = 3
        self.hiddenChannels = 512
        self.K = 32
        self.L = 3  # change dizz
        self.actNormScale = 1.0
        self.perm = "invconv"
        self.coupling = "affine"
        self.LU = True
        self.warmup = 10
        # zero means clipping off
        self.max_grad_clip = 0
        self.max_grad_norm = 0
        self.lr = 1e-4
        self.n_init_batch = 32
        self.y_classes = 10
        self.y_learn_top = False
        self.y_condition = False
        self.y_weight = 0.01
        self.loss = tools.loss
        self.weigh_decay = 1e-3
        self.lr_lambda = lambda epoch: min(1.0, (epoch + 1) / self.warmup)  # noqa
        self.initialize = True
        self.squeezeSplit = True  # toggle false for no split, not for conventional GLOW
        self.device = device
        self.model = model.Glow(self)
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adamax(self.model.parameters(), lr=self.lr, weight_decay=self.weigh_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

    def compute_loss(self, x, y):
        return tools.loss(self, x, y)

    def clip_gradients(self):
        if self.max_grad_clip > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.max_grad_clip)
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

    def sampler(self, bs, temp, n=10):
        if self.y_condition:
            y = torch.eye(n)
            y = y.repeat(bs // n)
            y = y[:32, :].to(self.device)
        else:
            y = None

        images = self.model(y_onehot=y, temperature=temp, reverse=True)

        return images

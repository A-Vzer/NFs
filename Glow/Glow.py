from Glow import model
from Glow import tools
import torch
import torch.optim as optim


class Parameters:
    def __init__(self, imShape, device):
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
        self.lr = 1e-3
        self.n_init_batch = 8
        self.y_classes = 10
        self.y_learn_top = False
        self.y_condition = False
        self.y_weight = 0.01
        self.loss = tools.loss
        self.weigh_decay = 5e-5
        self.lr_lambda = lambda epoch: min(1.0, (epoch + 1) / self.warmup)  # noqa
        self.initialize = True
        self.device = device
        self.model = model.Glow(imShape, self.hiddenUnits, self.K, self.L, self.actNormScale, self.perm,
                                self.coupling, self.LU, self.y_classes, self.y_learn_top, self.y_condition)
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

    def sampler(self, bs, n=10):
        if self.y_condition:
            y = torch.eye(n)
            y = y.repeat(bs // n)
            y = y[:32, :].to(self.device)
        else:
            y = None

        images = self.model(y_onehot=y, temperature=1, reverse=True)

        return images

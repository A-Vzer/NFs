from GLOW import Glow
import torch
import torch.optim as optim
from train_helper import check_manual_seed, load_dataset, loss_glow

class Adapter:
    def __init__(self, modelName, imShape):
        if modelName == 'glow':
            self.hiddenUnits = None
            self.K = None
            self.L = None
            self.actNormScale = None
            self.perm = None
            self.coupling = None
            self.LU = None
            self.warmup = None
            self.max_grad_clip = None
            self.max_grad_norm = None
            self.lr = None
            self.n_init_batch = 0
            self.compute_loss = loss_glow
            self.scheduler = None
            self.model = Glow.Glow(imShape, self.hiddenUnits, self.K, self.L, self.actNormScale, self.perm,
                                   self.coupling, self.LU)
            self.optimizer = optim.Adamax(self.model.parameters(), lr=self.lr, weight_decay=5e-5)
            # Interesting way of dealing with learning rate...
            self.lr_lambda = lambda epoch: min(1.0, (epoch + 1) / self.warmup)  # noqa
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)
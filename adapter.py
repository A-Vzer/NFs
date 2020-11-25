from Glow import model, Glow, tools
import torch.optim as optim
import torch



class Adapter:
    def __init__(self, modelName, imShape, device):
        if modelName == 'glow':
            self.modelName = modelName
            self.p = Glow.Parameters()
            self.model = model.Glow(imShape, self.p.hiddenUnits, self.p.K, self.p.L, self.p.actNormScale, self.p.perm,
                                    self.p.coupling, self.p.LU, self.p.y_classes, self.p.y_learn_top, self.p.y_condition)
            self.model = self.model.to(device)
            self.optimizer = optim.Adamax(self.model.parameters(), lr=self.p.lr, weight_decay=self.p.weigh_decay)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.p.lr_lambda)
            self.initialize = self.p.initialize
            self.n_init = self.p.n_init_batch
            self.y_condition = self.p.y_condition

    def compute_loss(self, x, y):
        if self.modelName == 'glow':
            return tools.loss(self, x, y)

    def clip_gradients(self):
        if self.p.max_grad_clip > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.p.max_grad_clip)
        if self.p.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.p.max_grad_norm)

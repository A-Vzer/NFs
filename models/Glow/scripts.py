import torch
from utilities.utils import check_manual_seed
from itertools import islice
import os
from barbar import Bar
dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz2.44.1/bin/'


class RunGlow:
    def __init__(self, flow, device):
        self.device = device
        self.flow = flow

    def initialize(self, ds, bs, seed):
        check_manual_seed(seed)
        if self.flow.initialize:
            print('Initializing...')
            self.flow.model.train()

            init_batches = []
            init_targets = []

            with torch.no_grad():
                for batch, target in islice(ds.train_loader, None, self.flow.n_init_batch):
                    init_batches.append(batch)
                init_batches = torch.cat(init_batches).to(self.device)
                assert init_batches.shape[0] == self.flow.n_init_batch * bs
                self.flow.model(init_batches)

        return ds.train_loader, ds.test_loader

    def train_step(self, train_loader, loss):
        for idx, (x, y) in enumerate(Bar(train_loader)):
            x = x.to(self.device)
            y = y.to(self.device)
            self.flow.optimizer.zero_grad()
            losses = self.flow.compute_loss(x, y)
            losses["total_loss"].backward()
            self.flow.clip_gradients()
            self.flow.optimizer.step()
            loss.append(losses["total_loss"].item())
        return loss

    def eval_step(self, test_loader, eval_loss):
        self.flow.scheduler.step()
        for idx, (x, y) in enumerate(Bar(test_loader)):
            self.flow.model.eval()
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                eval_loss_ = self.flow.compute_loss(x, y)
                eval_loss.append(eval_loss_["total_loss"].item())
        return eval_loss

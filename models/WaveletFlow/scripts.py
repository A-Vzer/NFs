import torch
from Datasets.datasets import Dataset
from models.adapter import Adapter
from utilities.utils import check_manual_seed
from itertools import islice
import os
from barbar import Bar
dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz2.44.1/bin/'


class RunWaveletFlow:
    def __init__(self, model, device):
        self.device = device
        self.model = model

    def initialize(self, ds, bs, seed):
        check_manual_seed(seed)
        if self.model.initialize:
            print('Initializing...')
            self.model.train()

            init_batches = []
            init_targets = []

            with torch.no_grad():
                for batch, target in islice(ds.train_loader, None, self.model.n_init_batch):
                    init_batches.append(batch)
                    init_targets.append(target)

                init_batches = torch.cat(init_batches).to(self.device)
                assert init_batches.shape[0] == self.model.n_init_batch * bs
                self.model(init_batches)

        return ds.train_loader, ds.test_loader

    def train_step(self, train_loader, loss):
        for idx, (x, y) in enumerate(Bar(train_loader)):
            x = x.to(self.device)
            y = y.to(self.device)
            self.model.optimizer.zero_grad()
            losses = self.model.compute_loss(x, y)
            losses["total_loss"].backward()
            self.model.clip_gradients()
            self.model.optimizer.step()
            loss.append(losses["total_loss"].item())
        return loss

    def eval_step(self, test_loader, eval_loss):
        for idx, (x, y) in enumerate(Bar(test_loader)):
            self.model.eval()
            x = x.to(self.device)
            with torch.no_grad():
                eval_loss_ = self.model.compute_loss(x, None)
                eval_loss.append(eval_loss_["total_loss"].item())
        return eval_loss

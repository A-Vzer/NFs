import torch
from modules.Wavelets.Haar.squeezeSplit import SqueezeSplit
from utilities.utils import check_manual_seed
from itertools import islice
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz2.44.1/bin/'


class RunWaveletFlow:
    def __init__(self, flow, device):
        self.device = device
        self.flow = flow
        self.haar_squeeze_split = SqueezeSplit(compensate=True, device=self.device)

    def initialize(self, ds, bs, level, seed):
        check_manual_seed(seed)
        if self.flow.initialize:
            print('Initializing...')
            self.flow.model.train()

            init_batches = []

            with torch.no_grad():
                for batch, target in islice(ds.train_loader, None, self.flow.n_init_batch):
                    init_batches.append(batch)

                init_batches = torch.cat(init_batches).to(self.device)
                assert init_batches.shape[0] == self.flow.n_init_batch * bs
                self.flow.model(init_batches, partial_level=level)

        return ds.train_loader, ds.test_loader

    def train_step(self, train_loader, level, loss):
        for idx, (x, y) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            self.flow.optimizer.zero_grad()
            losses = self.flow.compute_loss(x, y, level)
            losses["total_loss"].backward()
            self.flow.clip_gradients()
            self.flow.optimizer.step()
            loss.append(losses["total_loss"].item())
        return loss

    def eval_step(self, test_loader, level, eval_loss):
        for idx, (x, y) in enumerate(test_loader):
            self.flow.model.eval()
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                eval_loss_ = self.flow.compute_loss(x, y, level)
                eval_loss.append(eval_loss_["total_loss"].item())
        return eval_loss

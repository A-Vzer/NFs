from Glow import Glow, tools
import torch

import torch.optim as optim
from datasets import get_CIFAR10
from train_helper import check_manual_seed
from pathlib import Path
from utils import standardize
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets


class Dataset:
    def __init__(self, nameDataset, dataroot, dataAugment, download):
        if nameDataset == 'cifar10':
            self.imDim = (32, 32, 3)
            self.num_classes = 10
            test_transform = transforms.Compose([transforms.ToTensor(), standardize])
            if dataAugment:
                transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1)), transforms.RandomHorizontalFlip(), ]
            else:
                transformations = []

            transformations.extend([transforms.ToTensor(), standardize])
            train_transform = transforms.Compose(transformations)
            one_hot_encode = lambda target: F.one_hot(torch.tensor(target), self.num_classes)
            path = Path(dataroot) / "data" / "CIFAR10"
            self.train_dataset = datasets.CIFAR10(path, train=True, transform=train_transform, target_transform=one_hot_encode,
                                             download=download)
            self.test_dataset = datasets.CIFAR10(path, train=False, transform=test_transform, target_transform=one_hot_encode,
                                            download=download)


class Adapter:
    def __init__(self, modelName, imShape):
        if modelName == 'glow':
            self.hiddenUnits = 512
            self.K = 32
            self.L = 3
            self.actNormScale = 1.0
            self.perm = "invconv"
            self.coupling = "affine"
            self.LU = True
            self.warmup = 5
            self.max_grad_clip = 0
            self.max_grad_norm = 0
            self.lr = 5e-4
            self.n_init_batch = 8
            self.compute_loss = tools.loss_glow
            self.model = Glow.Glow(imShape, self.hiddenUnits, self.K, self.L, self.actNormScale, self.perm,
                                   self.coupling, self.LU)
            self.optimizer = optim.Adamax(self.model.parameters(), lr=self.lr, weight_decay=5e-5)
            # Interesting way of dealing with learning rate...
            self.lr_lambda = lambda epoch: min(1.0, (epoch + 1) / self.warmup)  # noqa
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)


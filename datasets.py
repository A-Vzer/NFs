from Datesets import Cifar
from pathlib import Path
from utils import standardize
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets


class Dataset:
    def __init__(self, nameDataset, dataroot, dataAugment, download):
        if nameDataset == 'cifar10':
            self.nameDataset = nameDataset
        self.data = Cifar.Cifar10(dataAugment, dataroot, download)

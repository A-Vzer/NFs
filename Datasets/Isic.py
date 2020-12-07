from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import numpy as np
from pathlib import Path
from utils import standardize
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
import os.path


class Isic:
    def __init__(self, dataAugment, dataroot, dim=32):
        test_transform = transforms.Compose([transforms.Resize([dim, dim]), transforms.ToTensor(), standardize])
        if dataAugment:
            transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1)), transforms.RandomHorizontalFlip(), ]
        else:
            transformations = []
        transformations.extend([transforms.Resize([32, 32]), transforms.ToTensor(), standardize])
        train_transform = transforms.Compose(transformations)
        path = Path(dataroot) / "data" / "ISIC"
        self.benign_dataset = ISICDataset(path, transform=train_transform, malignant=False)
        self.malignent_dataset = ISICDataset(path, transform=test_transform, malignant=True)
        self.imDim = (dim, dim, 3)
        self.num_classes = 2


class ISICDataset(Dataset):
    def __init__(self, img_path, transform, malignant=False):
        self.img_path = img_path
        if malignant:
            self.img_path = os.path.join(self.img_path, "benign/")
            self.ims = os.listdir(self.img_path)
            self.target = 1
        else:
            self.img_path = os.path.join(self.img_path, "malignant/")
            self.ims = os.listdir(self.img_path)
            self.target = 0
        self.transform = transform

    def __getitem__(self, index):
        img_name = os.path.join(self.img_path, self.ims[index])
        img = Image.open(img_name)
        img = self.transform(img)
        return {'image': img, 'label': self.target}

    def __len__(self):
        return len(self.ims)

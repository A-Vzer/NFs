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
    def __init__(self, dataroot, dim=32):
        transform = transforms.Compose([transforms.Resize([dim, dim]), transforms.ToTensor(), standardize])
        path = Path(dataroot) / "data" / "ISIC"
        self.train_dataset = ISICDataset(path, transform=transform, malignant=False)
        self.test_dataset = ISICDataset(path, transform=transform, malignant=True)
        self.imDim = (dim, dim, 3)
        self.num_classes = 2


class ISICDataset(Dataset):
    def __init__(self, img_path, transform, malignant=False):
        self.img_path = img_path
        if malignant:
            self.img_path = os.path.join(self.img_path, "malignant/")
            self.ims = os.listdir(self.img_path)
            self.target = 1
        else:
            self.img_path = os.path.join(self.img_path, "benign/")
            self.ims = os.listdir(self.img_path)
            self.target = 0
        self.transform = transform

    def __getitem__(self, index):
        img_name = os.path.join(self.img_path, self.ims[index])
        img = Image.open(img_name)
        img = self.transform(img)
        return img, self.target

    def __len__(self):
        return len(self.ims)

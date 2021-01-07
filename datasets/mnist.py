from pathlib import Path
from utilities.utils import standardize
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets


class Mnist:
    def __init__(self, dataAugment, dataroot, download, classNo):
        self.imDim = (3, 3, 28)
        self.num_classes = 10
        test_transform = transforms.Compose([transforms.ToTensor(), standardize])
        if dataAugment:
            transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1)), transforms.RandomHorizontalFlip(), ]
        else:
            transformations = []

        transformations.extend([transforms.ToTensor(), standardize])
        train_transform = transforms.Compose(transformations)
        one_hot_encode = lambda target: F.one_hot(torch.tensor(target), self.num_classes)
        path = Path(dataroot) / "data" / "MNIST"
        self.train_dataset = datasets.MNIST(path, train=True, transform=train_transform,
                                              target_transform=one_hot_encode, download=download)
        self.test_dataset = datasets.MNIST(path, train=False, transform=test_transform,
                                             target_transform=one_hot_encode, download=download)
        if classNo != False or classNo == 0:
            idx_train = torch.tensor(self.train_dataset.targets) == classNo
            idx_test = torch.tensor(self.test_dataset.targets) == classNo
            self.train_dataset.data = self.train_dataset.data[idx_train]
            self.test_dataset.data = self.test_dataset.data[idx_test]

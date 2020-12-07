from Datasets import Cifar, Svhn, Mnist, Isic


class Dataset:
    def __init__(self, nameDataset, dataroot, dataAugment, download=None, classNo=None):
        if nameDataset == 'cifar10':
            self.nameDataset = nameDataset
            self.data = Cifar.Cifar10(dataAugment, dataroot, download, classNo)
        if nameDataset == 'svhn':
            self.nameDataset = nameDataset
            self.data = Svhn.Svhn(dataAugment, dataroot, download, classNo)
        if nameDataset == 'mnist':
            self.nameDataset = nameDataset
            self.data = Mnist.Mnist(dataAugment, dataroot, download, classNo)
        if nameDataset == 'isic':
            self.nameDataset = nameDataset
            self.data = Isic.Isic(dataAugment, dataroot)


from Datesets import Cifar


class Dataset:
    def __init__(self, nameDataset, dataroot, dataAugment, download, classNo):
        if nameDataset == 'cifar10':
            self.nameDataset = nameDataset
        self.data = Cifar.Cifar10(dataAugment, dataroot, download, classNo)


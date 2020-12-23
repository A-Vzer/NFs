from Datasets import cifar, svhn, mnist, isic, celeb


class Dataset:
    def __init__(self, nameDataset, dataroot, dataAugment=None, download=None, classNo=None):
        if nameDataset == 'cifar10':
            self.nameDataset = nameDataset
            self.data = cifar.Cifar10(dataAugment, dataroot, download, classNo)
        if nameDataset == 'svhn':
            self.nameDataset = nameDataset
            self.data = svhn.Svhn(dataAugment, dataroot, download, classNo)
        if nameDataset == 'mnist':
            self.nameDataset = nameDataset
            self.data = mnist.Mnist(dataAugment, dataroot, download, classNo)
        if nameDataset == 'isic':
            self.nameDataset = nameDataset
            self.data = isic.Isic(dataroot)
        if nameDataset == 'celeb':
            self.nameDataset = nameDataset
            self.data = celeb.CelebA(dataroot)


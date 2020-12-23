from Datasets import cifar, svhn, mnist, isic, celeb
from torch.utils.data import DataLoader


class Dataset:
    def __init__(self, nameDataset, dataroot, bs, eval_bs, n_workers, dataAugment=None, download=None, classNo=None):
        self.bs = bs
        self.eval_bs = eval_bs
        self.n_workers = n_workers

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

        self.train_loader, self.test_loader = self.get_loaders()

    def get_loaders(self):
        train_loader = DataLoader(self.data.train_dataset, batch_size=self.bs, shuffle=True, num_workers=self.n_workers,
                                  drop_last=True)
        test_loader = DataLoader(self.data.test_dataset, batch_size=self.eval_bs, shuffle=False,
                                 num_workers=self.n_workers, drop_last=False)
        return train_loader, test_loader

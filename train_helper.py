from datasets import get_CIFAR10, get_SVHN
import torch
import random
from datasets import get_cifar10
import torch.nn.functional as F


def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    print("Using seed: {seed}".format(seed=seed))


def load_dataset(dataset, dataroot, dataAugment, download):
    if dataset == "cifar10":
        cifar10 = get_CIFAR10(dataAugment, dataroot, download)
        input_size, num_classes, train_dataset, test_dataset = cifar10
    if dataset == "svhn":
        svhn = get_SVHN(dataAugment, dataroot, download)
        input_size, num_classes, train_dataset, test_dataset = svhn

    return input_size, num_classes, train_dataset, test_dataset



import torch
from utils import postprocess
from datasets import Dataset
from adapter import Adapter
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from barbar import Bar
import numpy as np
import seaborn as sns
dir_path = os.path.dirname(os.path.realpath(__file__))


def sampler(modelName, modelDir, ds, n, temp, post=True):
    adapter = Adapter(modelName, ds.data.imDim, device)
    adapter.flow.model.set_actnorm_init()
    adapter.flow.model.load_state_dict(torch.load(modelDir)['model_state_dict'])
    images = adapter.flow.sampler(n, temp)
    if post:
        images = postprocess(images).cpu()
    return images


def likelihoodEst(adapter, temp=None, n=None, ds=None, sampled=True, train=False):
    if not sampled:
        nlls = []
        if train:
            data_loader = torch.utils.data.DataLoader(ds.data.train_dataset, batch_size=n, num_workers=0)
        else:
            data_loader = torch.utils.data.DataLoader(ds.data.test_dataset, batch_size=n, num_workers=0)
        for x, y in Bar(data_loader):
            with torch.no_grad():
                _, nll, _ = adapter.flow.model.forward(x.to(device))
                nlls.append(nll)
        return torch.cat(nlls)
    else:
        ims, adapter = sampler(modelName, modelSave, ds, n=n, temp=temp, post=False)
        with torch.no_grad():
            _, nll, _ = adapter.flow.model.forward(ims)
    return nll


def filter_threshold(llarray, threshold):
    print(np.sum(llarray < threshold), "outliers")
    return llarray[llarray > threshold]


def plot_samples(ims):
    grid = make_grid(ims[:30], nrow=6).permute(1, 2, 0)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid)
    plt.axis('off')
    plt.show()


def plot_hist(nllts, dataset):
    plt.figure()
    for nllt, label in nllts:
        sns.distplot(filter_threshold(-nllt, -10), norm_hist=True, label=label)

    plt.title(f"Trained on {dataset.upper()} {classNo1}")
    plt.xlabel("Negative bits per dimension")
    plt.legend(fontsize=16)
    plt.show()


if __name__ == "__main__":
    cuda = True
    device = "cpu" if (not torch.cuda.is_available() or not cuda) else "cuda:0"
    print(device)
    modelName = 'glow'
    data_in = 'svhn'
    data_out = 'cifar10'
    classNo1 = 1
    classNo2 = 1
    output_dir = "saves\\"
    modelSave = "saves\\glow-cifar10-bs64-ep1000-lr001-class1_Final-L=3.pt"
    sample = False
    likelihood = True
    losscape = False
    ds_in = Dataset(data_in, dataroot=dir_path, dataAugment=True, download=True, classNo=classNo1)
    ds_out = Dataset(data_out, dataroot=dir_path, dataAugment=True, download=True, classNo=classNo2)
    adapter = Adapter(modelName, ds_in.data.imDim, device)
    if sample:
        ims = sampler(modelName, modelSave, ds_in, n=512, temp=0.6)
        plot_samples(ims)
    if likelihood:
        nll_in_train = likelihoodEst(adapter, n=n, ds=ds_in, sampled=False, train=True).cpu().detach().numpy()
        nll_in_test = likelihoodEst(adapter, n=n, ds=ds_in, sampled=False, train=False).cpu().detach().numpy()
        nll_out = likelihoodEst(adapter, n=n, ds=ds_out, sampled=False, train=False).cpu().detach().numpy()
        nllts = [-nll_in_train, -nll_in_test, -nll_out]
        labels = ["1 train", "1 test", "CIFAR10: Car"]
        plot_hist(zip(nllts, labels), data_in)
    if losscape:
        y = torch.load(modelSave)['trainLoss']
        print(y)


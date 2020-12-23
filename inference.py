import torch
from utilities.utils import postprocess
from Datasets.datasets import Dataset
from models.adapter import Adapter
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


def plot_hist(nllts, dataset, classTrained):
    plt.figure()
    for nllt, label in nllts:
        sns.distplot(filter_threshold(nllt, -10), norm_hist=True, label=label)

    plt.title(f"Trained on {dataset.upper()} {classTrained}")
    plt.xlabel("Negative bits per dimension")
    plt.ylabel("Relative frequency")
    plt.legend(fontsize=16)
    plt.tick_params('y', left=False, labelleft=False)
    plt.show()


if __name__ == "__main__":
    cuda = True
    device = "cpu" if (not torch.cuda.is_available() or not cuda) else "cuda:0"
    print(device)
    modelName = 'waveletglow'
    data_in = 'isic'
    data_out = 'isic'
    classNo1 = 1
    classNo2 = 9
    output_dir = "saves\\"
    modelSave = "saves\\waveletglow-isic-bs16-ep500-lr0001-classbenign.pt"
    sample = False  # remember 0-1 or -0.5-05 !!
    likelihood = False
    losscape = True
    ds_in = Dataset(data_in, dataroot=dir_path)
    ds_out = Dataset(data_out, dataroot=dir_path)
    adapter = Adapter(modelName, ds_in.data.imDim, device)
    if sample:
        ims = sampler(modelName, modelSave, ds_in, n=512, temp=1)
        plot_samples(ims)
    if likelihood:
        nll_in_train = likelihoodEst(adapter, n=512, ds=ds_in, sampled=False, train=True).cpu().detach().numpy()
        # nll_in_test = likelihoodEst(adapter, n=512, ds=ds_in, sampled=False, train=False).cpu().detach().numpy()
        nll_out = likelihoodEst(adapter, n=512, ds=ds_out, sampled=False, train=False).cpu().detach().numpy()
        nllts = [-nll_in_train, -nll_out]
        labels = ["Benign", "Malignant"]
        plot_hist(zip(nllts, labels), data_in, 'benign')
    if losscape:
        y = torch.load(modelSave)['evalLoss']
        plt.plot(y, c='r')
        plt.title("Losscape of training 32x32 ISIC benign data")
        plt.xlabel("Epochs")
        plt.tick_params('y', left=False, labelleft=False)
        plt.ylabel("Loss")
        plt.show()

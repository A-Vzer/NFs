import torch
from utilities.utils import postprocess
from datasets.datasets import Dataset
from adapter import Adapter
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from barbar import Bar
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import pickle
dir_path = os.path.dirname(os.path.realpath(__file__))


def sampler(modelName, modelDir, ds, n, temp, post=True):
    adapter = Adapter(modelName, ds.data.imDim, device)
    adapter.flow.model.set_actnorm_init()
    adapter.flow.model.load_state_dict(torch.load(modelDir)['model_state_dict'])
    images = adapter.flow.sampler(n, temp)
    if post:
        images = postprocess(images).cpu()
    return images


def likelihoodEst(adapter, temp=None, n=None, ds=None, sampled=True, train=False, out=False):
    train_loader, test_loader, out_loader = ds.get_loaders()

    if not sampled:
        nlls = []
        if out:
            data_loader = out_loader
        elif train:
            data_loader = train_loader
        else:
            data_loader = test_loader

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


def plot_hist(nllts):
    plt.figure()
    lengths = []
    logs = []
    for nllt, label in nllts:
        sns.distplot(filter_threshold(nllt, -10), norm_hist=True, label=label)
        lengths.append(len(nllt))
        logs.append(nllt)

    # plt.title(f"Trained on {dataset.upper()} {classTrained}")
    plt.xlabel("Negative bits per dimension")
    plt.ylabel("Relative frequency")
    plt.xlim([-7, -3.7])
    plt.legend(fontsize=16)
    plt.tick_params('y', left=False, labelleft=False)
    plt.show()

    labels_positive = [1] * lengths[0]
    labels_negative = [0] * lengths[2]
    test_labels = labels_positive + labels_negative
    test_scores = np.concatenate((logs[0], logs[2]))
    fpr, tpr, _ = roc_curve(test_labels, test_scores)

    labels_positive2 = [1] * lengths[1]
    labels_negative2 = [0] * lengths[2]
    test_labels2 = labels_positive2 + labels_negative2
    test_scores2 = np.concatenate((logs[1], logs[2]))
    fpr2, tpr2, _ = roc_curve(test_labels2, test_scores2)

    with open("roc\\roc-train-out.pickle", "wb+") as output_file:
        pickle.dump([fpr, tpr], output_file)
    with open("roc\\roc-test-out.pickle", "wb+") as output_file:
        pickle.dump([fpr2, tpr2], output_file)


if __name__ == "__main__":
    cuda = True
    device = "cpu" if (not torch.cuda.is_available() or not cuda) else "cuda:0"
    print(device)
    modelName = 'glow'
    data = 'isic'
    modelSave = "saves\\Glow-isic-bs16-ep1000-lr0005-classbenign_Final.pt"
    sample = False  # remember 0-1 or -0.5-05 !!
    likelihood = False  # for probability curves
    losscape = False  # for training loss plot
    roc = True  # for roc curves, need
    ds = Dataset(data, dataroot=dir_path, bs=512, eval_bs=512)
    adapter = Adapter(modelName, ds.data.imDim, device)
    if sample:
        ims = sampler(modelName, modelSave, ds, n=512, temp=1)
        plot_samples(ims)
    if likelihood:
        nll_in_train = likelihoodEst(adapter, n=512, ds=ds, sampled=False, train=True).cpu().detach().numpy()
        nll_in_test = likelihoodEst(adapter, n=512, ds=ds, sampled=False, train=False).cpu().detach().numpy()
        nll_out = likelihoodEst(adapter, n=512, ds=ds, sampled=False, train=False, out=True).cpu().detach().numpy()
        nllts = [-nll_in_train, -nll_in_test, -nll_out]
        labels = ["Benign train", "Benign test", "Malignant"]
        plot_hist(zip(nllts, labels))
    if losscape:
        y = torch.load(modelSave)['evalLoss']
        plt.plot(y, c='r')
        plt.title("Losscape of training 32x32 ISIC benign data")
        plt.xlabel("Epochs")
        plt.tick_params('y', left=False, labelleft=False)
        plt.ylabel("Loss")
        plt.show()
    if roc:
        with open("roc\\roc-train-out.pickle", "rb") as output_file:
            train = pickle.load(output_file)
        with open("roc\\roc-test-out.pickle", "rb") as output_file:
            test = pickle.load(output_file)

        plt.figure()
        plt.plot(train[0], train[1], linewidth=2, color='navy', label='ROC curve train (area: %0.2f)' % auc(train[0], train[1]))
        plt.plot(test[0], test[1], linewidth=2, color='darkred', label='ROC curve test (area: %0.2f)' % auc(test[0], test[1]))
        x = np.linspace(0, 1, 3)
        plt.plot(x, x, 'r--', linewidth=2, label='No skill')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('1-Specificity')
        plt.ylabel('Sensitivity')
        plt.legend(loc="lower right", fontsize=8)
        plt.show()

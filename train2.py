import torch
import torch.utils.data as data
from train_helper import check_manual_seed
from datasets import Dataset
from adapter import Adapter
from itertools import islice
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage, Loss
import os
from barbar import Bar
dir_path = os.path.dirname(os.path.realpath(__file__))


def execute(modelName, dataset, dataroot, download, dataAugment, bs, eval_bs, eps, modelSave, seed, n_workers,
            cuda, output_dir, optimSave):
    device = "cpu" if (not torch.cuda.is_available() or not cuda) else "cuda:0"
    print(device)
    check_manual_seed(seed)
    ds = Dataset(dataset, dataroot, dataAugment, download)
    train_loader = data.DataLoader(ds.data.train_dataset, batch_size=bs, shuffle=True, num_workers=n_workers,
        drop_last=True)
    test_loader = data.DataLoader(ds.data.test_dataset, batch_size=eval_bs, shuffle=False, num_workers=n_workers,
        drop_last=False)
    flow = Adapter(modelName, ds.data.imDim, device)

    loss = []
    avgEval = []
    if flow.initialize:
        print('Initializing...')
        flow.model.train()

        init_batches = []
        init_targets = []

        with torch.no_grad():
            for batch, target in islice(train_loader, None, flow.n_init):
                init_batches.append(batch)
                init_targets.append(target)

            init_batches = torch.cat(init_batches).to(device)
            assert init_batches.shape[0] == flow.n_init * bs

            flow.model(init_batches)

    for i in range(eps):
        print(f"Epoch: {i}")
        for idx, (x, y) in enumerate(Bar(train_loader)):
            x = x.to(device)
            y = y.to(device)
            flow.model.train()
            flow.optimizer.zero_grad()
            losses = flow.compute_loss(x, y)
            losses["total_loss"].backward()
            flow.clip_gradients()
            flow.optimizer.step()
            loss.append(losses["total_loss"].item())
        print("\n Evaluating...")
        eval_loss = []
        for idx, (x, y) in enumerate(Bar(test_loader)):
            flow.model.eval()
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                eval_loss_ = flow.compute_loss(x, y)
                eval_loss.append(eval_loss_["total_loss"].item())
        avgEval.append(sum(eval_loss)/len(eval_loss))
        print(f'Avg eval loss: {avgEval[-1]}')

        torch.save({'epoch': i, 'model_state_dict': flow.model.state_dict(),
                    'optimizer_state_dict': flow.optimizer.state_dict(), 'trainLoss': loss[-1],
                    'evalLoss': avgEval[-1]}, output_dir + f"model.pt")


if __name__ == "__main__":
    modelName = 'glow'
    dataset = 'cifar10'
    dataroot = dir_path
    download = True
    dataAugment = True
    bs = 32
    eval_bs = 512
    eps = 20
    seed = 42069
    n_workers = 0
    cuda = True
    output_dir = "saves/"
    modelSave = None
    optimSave = None
    print(f"Model: {modelName}, Dataset: {dataset}, bs: {bs}, eps: {eps}")
    execute(modelName, dataset, dataroot, download, dataAugment, bs, eval_bs, eps, modelSave, seed, n_workers,
            cuda, output_dir, optimSave)
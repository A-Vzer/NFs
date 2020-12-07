import torch
import torch.utils.data as data
from train_helper import check_manual_seed
from datasets import Dataset
from adapter import Adapter
from itertools import islice
import os
from barbar import Bar
dir_path = os.path.dirname(os.path.realpath(__file__))


def execute(modelName, ds, bs, eval_bs, eps, seed, n_workers, device, output_dir):
    check_manual_seed(seed)
    train_loader = data.DataLoader(ds.data.train_dataset, batch_size=bs, shuffle=True, num_workers=n_workers,
        drop_last=True)
    test_loader = data.DataLoader(ds.data.test_dataset, batch_size=eval_bs, shuffle=False, num_workers=n_workers,
        drop_last=False)
    adapter = Adapter(modelName, ds.data.imDim, device)

    loss = []
    avgEval = []
    if adapter.flow.initialize:
        print('Initializing...')
        adapter.flow.model.train()

        init_batches = []
        init_targets = []

        with torch.no_grad():
            for batch, target in islice(train_loader, None, adapter.flow.n_init_batch):
                init_batches.append(batch)
                init_targets.append(target)

            init_batches = torch.cat(init_batches).to(device)
            assert init_batches.shape[0] == adapter.flow.n_init_batch * bs

            adapter.flow.model(init_batches)

    for i in range(eps):
        print(f"Epoch: {i}")
        for idx, (x, y) in enumerate(Bar(train_loader)):
            x = x.to(device)
            y = y.to(device)
            adapter.flow.model.train()
            adapter.flow.optimizer.zero_grad()
            losses = adapter.flow.compute_loss(x, y)
            losses["total_loss"].backward()
            adapter.flow.clip_gradients()  # if set
            adapter.flow.optimizer.step()
            loss.append(losses["total_loss"].item())
        print("\n Evaluating...")
        eval_loss = []
        for idx, (x, y) in enumerate(Bar(test_loader)):
            adapter.flow.model.eval()
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                eval_loss_ = adapter.flow.compute_loss(x, y)
                eval_loss.append(eval_loss_["total_loss"].item())
        avgEval.append(sum(eval_loss)/len(eval_loss))
        print(f'Avg eval loss: {avgEval[-1]}')

        if i == eps-1:
            directory = f"{modelName}-{ds.nameDataset}-bs{bs}-ep{eps}-lr{str(adapter.flow.lr)[2:]}-class{classNo}_Final.pt"
        else:
            directory = f"{modelName}-{ds.nameDataset}-bs{bs}-ep{eps}-lr{str(adapter.flow.lr)[2:]}-class{classNo}.pt"

        torch.save({'epoch': i, 'model_state_dict': adapter.flow.model.state_dict(),
                    'optimizer_state_dict': adapter.flow.optimizer.state_dict(), 'trainLoss': loss,
                    'evalLoss': avgEval}, dir_path + '\\' + output_dir + directory)

    return dir_path + '\\' + output_dir + directory


if __name__ == "__main__":
    cuda = True
    device = "cpu" if (not torch.cuda.is_available() or not cuda) else "cuda:0"
    print(device)
    modelName = 'glow'
    dataset = 'mnist'
    classNo = 8
    dataroot = dir_path
    download = True
    dataAugment = True
    bs = 64
    eval_bs = 512
    eps = 40
    seed = 42069
    n_workers = 0
    output_dir = "saves\\"
    modelSave = "saves\\model-bs32-ep40-lr001_Final.pt"
    optimSave = None
    train = True
    sample = False
    likelihood = False
    ds = Dataset(dataset, dataroot, dataAugment, download, classNo)
    if train:
        print(f"Model: {modelName}, Dataset: {dataset}, bs: {bs}, eps: {eps}, classNo: {classNo}")
        modelSave = execute(modelName, ds, bs, eval_bs, eps, seed, n_workers, device, output_dir)





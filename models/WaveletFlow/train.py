import torch
from models.adapter import Adapter
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
from Datasets.datasets import Dataset


def execute(eval_bs, eps, seed, n_workers, device):
    adapter = Adapter('WaveletGlow', imShape=[3, 64, 64], device=device)
    bs = adapter.flow.n_batch[level]
    crop = adapter.flow.cropFactor[level]
    ds = Dataset(dataset, dir_path, bs, eval_bs, n_workers, crop=crop)
    train_loader, test_loader = adapter.scripts.initialize(ds, bs, eval_bs, seed, n_workers, crop)

    loss = []
    avgEval = []
    lastEval = 999999
    for i in range(eps):
        print(f"Epoch: {i}")
        loss = adapter.scripts.train_step(train_loader, loss, adapter.flow)
        print("\n Evaluating...")
        eval_loss = []
        eval_loss = adapter.scripts.eval_step(test_loader, eval_loss, adapter.flow)
        lastEval = eval_loss
        avgEval.append(sum(eval_loss)/len(eval_loss))
        print(f'Avg eval loss: {avgEval[-1]}')
        save_name(adapter, i, loss, lastEval, avgEval)


def save_name(adapter, idx, loss, lastEval, avgEval):
    if idx == eps - 1:
        directory = f"WaveletGlow-{dataset}-bs{bs}-ep{eps}-lr{str(adapter.flow.lr)[2:]}-class{classNo}_Final.pt"
        torch.save({'epoch': idx, 'model_state_dict': adapter.flow.model.state_dict(),
                    'optimizer_state_dict': adapter.flow.optimizer.state_dict(), 'trainLoss': loss,
                    'evalLoss': avgEval}, dir_path + '\\' + output_dir + directory)

    elif lastEval > avgEval[-1]:
        print('Saving model...')
        directory = f"WaveletGlow-{dataset}-bs{bs}-ep{eps}-lr{str(adapter.flow.lr)[2:]}-class{classNo}.pt"
        lastEval = avgEval[-1]

        torch.save({'epoch': idx, 'model_state_dict': adapter.flow.model.state_dict(),
                    'optimizer_state_dict': adapter.flow.optimizer.state_dict(), 'trainLoss': loss,
                    'evalLoss': avgEval}, dir_path + '\\' + output_dir + directory)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    level = 0
    cuda = True
    device = "cpu" if (not torch.cuda.is_available() or not cuda) else "cuda:0"
    dataset = 'isic'
    classNo = 'benign'
    bs = 16
    eval_bs = 512
    eps = 500
    seed = 42069
    n_workers = 0
    output_dir = "saves\\"
    modelSave = "saves\\model-bs32-ep40-lr001_Final.pt"
    print(f"Model: WaveletGlow Level: {level}, Dataset: {dataset}, bs: {bs}, eps: {eps}, classNo: {classNo}")
    execute(eval_bs, eps, seed, n_workers, device)
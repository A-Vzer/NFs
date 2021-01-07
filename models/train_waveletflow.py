import torch
from adapter import Adapter
import os
import argparse
from datasets.datasets import Dataset
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def execute():
    adapter = Adapter('waveletglow', imShape=[3, 64, 64], device=device, level=level)
    bs = adapter.flow.n_batch[level]
    crop = adapter.flow.cropFactor[level]
    ds = Dataset(dataset, dir_path, bs, eval_bs, n_workers, crop=crop)
    train_loader, test_loader = adapter.scripts.initialize(ds, bs, level, seed)

    loss = []
    avgEval = []
    lastEval = 999999
    for i in range(eps):
        print(f"Epoch level {level}: {i}")
        loss = adapter.scripts.train_step(train_loader, level, loss)
        eval_loss = []
        eval_loss = adapter.scripts.eval_step(test_loader, level, eval_loss)
        avgEval.append(sum(eval_loss)/len(eval_loss))
        print(f'Avg eval loss level {level}: {avgEval[-1]}')
        save_name(adapter, i, loss, lastEval, avgEval)


def save_name(adapter, idx, loss, lastEval, avgEval):
    if idx == eps - 1:
        directory = f"WaveletGlow-level{level}-{dataset}-ep{eps}-lr{str(adapter.flow.lr)[2:]}-class{classNo}_Final.pt"
        torch.save({'epoch': idx, 'model_state_dict': adapter.flow.model.state_dict(),
                    'optimizer_state_dict': adapter.flow.optimizer.state_dict(), 'trainLoss': loss,
                    'evalLoss': avgEval}, dir_path + '\\' + output_dir + directory)

    elif lastEval > avgEval[-1]:
        print(f'Saving model level {level}...')
        directory = f"WaveletGlow-level{level}-{dataset}-ep{eps}-lr{str(adapter.flow.lr)[2:]}-class{classNo}.pt"
        lastEval = avgEval[-1]

        torch.save({'epoch': idx, 'model_state_dict': adapter.flow.model.state_dict(),
                    'optimizer_state_dict': adapter.flow.optimizer.state_dict(), 'trainLoss': loss,
                    'evalLoss': avgEval}, dir_path + '\\' + output_dir + directory)
    return lastEval


torch.cuda.empty_cache()
argParser = argparse.ArgumentParser(description="start/resume training")
argParser.add_argument("-l","--level",dest="level",action="store",default=0,type=int)
cmd_args = argParser.parse_args()
cuda = True
device = "cpu" if (not torch.cuda.is_available() or not cuda) else "cuda:0"
dataset = 'isic'
classNo = 'benign'
level = cmd_args.level
eval_bs = 512
eps = 50000
seed = 42069
n_workers = 0
output_dir = "saves\\"
modelSave = "saves\\model-bs32-ep40-lr001_Final.pt"
print(f"Model: WaveletGlow Level: {level}, Dataset: {dataset}, eps: {eps}, classNo: {classNo}")
execute()

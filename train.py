import torch
import torch.utils.data as data
from utils import standardize
from train_helper import check_manual_seed
import os
import torch.nn.functional as F
from pathlib import Path
from adapter import Adapter, Dataset
from torchvision import transforms, datasets
from itertools import islice
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage, Loss
import torch.optim as optim
dir_path = os.path.dirname(os.path.realpath(__file__))


def execute(modelName, dataset, dataroot, download, dataAugment, bs, eval_bs, eps, lr, modelSave, seed, n_workers,
            cuda, output_dir, optimSave):
    device = "cpu" if (not torch.cuda.is_available() or not cuda) else "cuda:0"
    print(device)
    check_manual_seed(seed)
    ds = Dataset(dataset, dataroot, dataAugment, download)
    train_loader = data.DataLoader(ds.train_dataset, batch_size=bs, shuffle=True, num_workers=n_workers,
        drop_last=True, )
    test_loader = data.DataLoader(ds.test_dataset, batch_size=eval_bs, shuffle=False, num_workers=n_workers,
        drop_last=False, )
    p = Adapter(modelName, ds.imDim)
    p.optimizer = optim.Adamax(p.model.parameters(), lr=lr, weight_decay=5e-5)
    # Interesting way of dealing with learning rate...
    model = p.model.to(device)

    def step(engine, batch):
        model.train()
        p.optimizer.zero_grad()

        x, y = batch
        x = x.to(device)
        z, nll, y_logits = model(x, None)
        losses = p.compute_loss(modelName, nll)

        losses["total_loss"].backward()

        if p.max_grad_clip > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), p.max_grad_clip)
        if p.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), p.max_grad_norm)

        p.optimizer.step()

        return losses

    def eval_step(engine, batch):
        model.eval()

        x, y = batch
        x = x.to(device)

        with torch.no_grad():
            z, nll, y_logits = model(x, None)
            losses = p.compute_loss(modelName, nll, reduction="none")

        return losses

    trainer = Engine(step)
    checkpoint_handler = ModelCheckpoint(output_dir, "glow", n_saved=2, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {"model": model, "optimizer": p.optimizer})
    monitoring_metrics = ["total_loss"]
    RunningAverage(output_transform=lambda x: x["total_loss"]).attach(trainer, "total_loss")

    evaluator = Engine(eval_step)
    Loss(lambda x, y: torch.mean(x), output_transform=lambda x: (x["total_loss"], torch.empty(
            x["total_loss"].shape[0]))).attach(evaluator, "total_loss")
    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    if modelSave:
        model.load_state_dict(torch.load(modelSave))
        model.set_actnorm_init()

        if optimSave:
            p.optimizer.load_state_dict(torch.load(optimSave))

        file_name, ext = os.path.splitext(modelSave)
        resume_epoch = int(file_name.split("_")[-1])

        @trainer.on(Events.STARTED)
        def resume_training(engine):
            engine.state.epoch = resume_epoch
            engine.state.iteration = resume_epoch * len(engine.state.dataloader)

    @trainer.on(Events.STARTED)
    def init(engine):
        model.train()

        init_batches = []
        init_targets = []

        with torch.no_grad():
            for batch, target in islice(train_loader, None, p.n_init_batch):
                init_batches.append(batch)
                init_targets.append(target)

            init_batches = torch.cat(init_batches).to(device)
            assert init_batches.shape[0] == p.n_init_batch * bs

            model(init_batches)

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate(engine):
        evaluator.run(test_loader)
        if p.scheduler:
            p.scheduler.step()
        metrics = evaluator.state.metrics

        losses = ", ".join([f"{key}: {value:.2f}" for key, value in metrics.items()])

        print(f"Validation Results - Epoch: {engine.state.epoch} {losses}")

    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED, )

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        pbar.log_message(f"Epoch {engine.state.epoch} done. Time per batch: {timer.value():.3f}[s]")
        timer.reset()

    trainer.run(train_loader, eps)


if __name__ == "__main__":
    modelName = 'glow'
    dataset = 'cifar10'
    dataroot = dir_path
    download = True
    dataAugment = True
    bs = 64
    eval_bs = 512
    eps = 200
    lr = 5e-4
    seed = 42069
    n_workers = 0
    cuda = True
    output_dir = "output/"
    modelSave = None
    optimSave = None
    print(f"Model: {modelName}, Dataset: {dataset}, bs: {bs}, eps: {eps}, lr: {lr}")
    execute(modelName, dataset, dataroot, download, dataAugment, bs, eval_bs, eps, lr, modelSave, seed, n_workers,
            cuda, output_dir, optimSave)
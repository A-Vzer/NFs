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
dir_path = os.path.dirname(os.path.realpath(__file__))


def execute(modelName, dataset, dataroot, download, dataAugment, bs, eval_bs, eps, lr, modelSave, seed, n_workers,
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

    def step(engine, batch):
        flow.model.train()
        flow.optimizer.zero_grad()

        x, y = batch
        x = x.to(device)
        losses = flow.compute_loss(x, y)
        losses["total_loss"].backward()
        flow.clip_gradients()
        flow.optimizer.step()

        return losses

    def eval_step(engine, batch):
        flow.model.eval()

        x, y = batch
        x = x.to(device)

        with torch.no_grad():
            losses = flow.compute_loss(x, y)

        return losses

    trainer = Engine(step)
    checkpoint_handler = ModelCheckpoint(output_dir, "glow", n_saved=2, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {"model": flow.model, "optimizer": flow.optimizer})
    monitoring_metrics = ["total_loss"]
    RunningAverage(output_transform=lambda x: x["total_loss"]).attach(trainer, "total_loss")

    evaluator = Engine(eval_step)
    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    if modelSave:
        flow.model.load_state_dict(torch.load(modelSave))
        flow.model.set_actnorm_init()

        if optimSave:
            flow.optimizer.load_state_dict(torch.load(optimSave))

        file_name, ext = os.path.splitext(modelSave)
        resume_epoch = int(file_name.split("_")[-1])

        @trainer.on(Events.STARTED)
        def resume_training(engine):
            engine.state.epoch = resume_epoch
            engine.state.iteration = resume_epoch * len(engine.state.dataloader)

    @trainer.on(Events.STARTED)
    def init(engine):
        if flow.initialize:
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

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate(engine):
        evaluator.run(test_loader)
        if flow.scheduler:
            flow.scheduler.step()
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
    bs = 32
    eval_bs = 512
    eps = 5
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
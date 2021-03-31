import albumentations as alb
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

from ignite import engine
from ignite import metrics
from ignite import handlers

from models import BiSeNetV2
from utils import AerialDataset
from utils import thresholded_transform, getFlopsandParams
from utils import init_logging, read_config
from utils import init_loss, init_optimizer, LossWithAux
from utils import attach_training_logger, attach_model_checkpoint


import argparse
import shutil
import os 
import logging

import matplotlib
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


def finetune_parse():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset",type=str, default="potsdam")
    parser.add_argument("--model",type=str, required=True)  # /home/admin/segmentation/task2/checkpoints/
    parser.add_argument("--name",type=str, required=True)

    parser.add_argument("--nocuda", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()



colors = {0:(255,255,255),1:(0,0,255),2:(0,255,255),3:(0,255,0),4:(255,255,0),5:(255,0,0)}

def attach_metric_logger(
    evaluator: engine.Engine,
    dataloader: torch.utils.data.DataLoader,
    data_name: str,
    writer: tensorboard.SummaryWriter,
):
    @trainer.on(engine.Events.EPOCH_COMPLETED)
    def log_metrics(engine):
        evaluator.run(dataloader)
        metrics = evaluator.state.metrics
        message = ''
        for metric_name, metric_value in metrics.items():
            message += f'{metric_name}: {metric_value} '

        logger.info(message)


if __name__ == "__main__":
    
    args = finetune_parse()
    config = read_config()
    try:
        if args.overwrite:
            shutil.rmtree(f"./logs/{args.name}", ignore_errors=True)
        os.mkdir(f"./logs/{args.name}")
    except:
        print(f"log folder {args.name} already exits.")

    init_logging(log_path = f"./logs/{args.name}")


    device = "cpu" if args.nocuda else "cuda"
    print(f"Using device: {device}")
    model = BiSeNetV2(config["n_classes"])
    model.load_state_dict(torch.load(args.model))
    model = nn.DataParallel(model, device_ids = [0, 1, 2, 3]).to(device)
    
    finetune_imgs = f"/home/admin/segmentation/task2/data/gen/images"
    finetune_masks = f"/home/admin/segmentation/task2/data/gen/masks"
    fintune_transform = alb.Compose([
        alb.Resize(512,512),
        ToTensorV2(),
    ])
    finetune_dataset = AerialDataset("train", "gen", finetune_imgs, finetune_masks, transform=fintune_transform)
    finetune_loader = DataLoader(finetune_dataset, batch_size=16, pin_memory=True, drop_last=True)

    eval_imgs = f"/home/admin/segmentation/task2/data/vaihingen/train/cropped/images/val"
    eval_masks = f"/home/admin/segmentation/task2/data/vaihingen/train/cropped/masks/val"
    eval_dataset = AerialDataset("val", "vaihingen", eval_imgs, eval_masks)
    eval_loader = DataLoader(eval_dataset, batch_size=16, pin_memory=True, drop_last=True)


    _loss_fn = init_loss(config["loss_fn"])
    loss_fn = LossWithAux(_loss_fn)
    _optimizer = init_optimizer(config)
    optimizer = _optimizer(model.parameters(), lr = config["learning_rate"])

    trainer = engine.create_supervised_trainer(
        model = model,
        optimizer = optimizer,
        loss_fn = loss_fn,
        device = device,
        non_blocking = True,
    )

    evaluator = engine.create_supervised_evaluator(
        model = model,
        metrics={
            "Loss": metrics.Loss(nn.CrossEntropyLoss()),
            "Accuracy@0.3": metrics.Accuracy(thresholded_transform(0.3)),
            "Precision@0.3": metrics.Accuracy(thresholded_transform(0.3)), 
            "IOU": metrics.IoU(metrics.ConfusionMatrix(num_classes = config["n_classes"])),
            "mIOU": metrics.mIoU(metrics.ConfusionMatrix(num_classes = config["n_classes"])),
            # "FPS": metrics.Frequency(output_transform=lambda x: x[0]),
        },
        device = device,
        non_blocking=True,
        output_transform = lambda x, y, y_pred: (torch.sigmoid(y_pred["out"]), y),
    )

    writer = tensorboard.SummaryWriter(log_dir=f'summary/{config["model_tag"]}')
    attach_metric_logger(evaluator, eval_loader, 'val', writer=writer)
    attach_training_logger(trainer, writer=writer, log_interval=1)
    attach_model_checkpoint(trainer, {config["model_tag"]: model.module}, args.name)

    trainer.run(finetune_loader, max_epochs=config["epochs"])
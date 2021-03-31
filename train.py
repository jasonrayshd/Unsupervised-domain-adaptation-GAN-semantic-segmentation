import argparse
import logging

import torch
from torch import nn
from torch import optim
from torch.utils import data
from ignite import engine
from ignite import metrics
from torch.utils import tensorboard

from utils import init_logging, read_config
from utils import init_optimizer, init_loss

from utils import LossWithAux
from utils import create_data_loaders
from utils import attach_lr_scheduler
from utils import attach_training_logger
from utils import attach_model_checkpoint
from utils import attach_metric_logger
from utils import thresholded_transform
from utils import AerialDataset
from models import BiSeNetV2

import os

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str,default="temp")
    parser.add_argument("--dataset", type=str,default="potsdam")
    parser.add_argument("--main_cuda", type=int,default=0)

    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def train():
    # initiate command line arguments, configuration file and logging block
    args = parse_args()
    config = read_config()
    try:
        if args.overwrite:
            shutil.rmtree(f"./logs/{args.name}", ignore_errors=True)
        os.mkdir(f"./logs/{args.name}")
    except:
        print(f"log folder {args.name} already exits.")

    init_logging(log_path = f"./logs/{args.name}")
    
    # determine train model on which device, cuda or cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"running training on {device}")
    device += f':{args.main_cuda}'

    # prepare training and validation datasets
    logger.info('creating dataset and data loaders')
    dataset = args.dataset

    train_dataset = AerialDataset("train", dataset, config[dataset]["train"]["image_path"], config[dataset]["train"]["mask_path"])
    val_dataset = AerialDataset("val", dataset, config[dataset]["val"]["image_path"], config[dataset]["val"]["mask_path"])
    train_loader, train_metrics_loader, val_metrics_loader = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_workers=config["num_workers"],
        batch_size=config["batchsize"],
    )

    # create model
    logger.info(f'creating BiseNetv2 and optimizer with initial lr of {config["learning_rate"]}')

    model = BiSeNetV2(config["n_classes"])
    model = nn.DataParallel(model, device_ids=[x for x in range(args.main_cuda, 4)]).to(device)

    # initiate loss function and optimizer
    optimizer_fn = init_optimizer(config)
    optimizer = optimizer_fn(model.parameters(), lr=config["learning_rate"])

    logger.info('creating trainer and evaluator engines')

    _loss_fn = init_loss(config["loss_fn"])
    loss_fn = LossWithAux(_loss_fn)

    # create trainer and evaluator wiht ignite.engine
    trainer = engine.create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        non_blocking=True,
    )
    
    evaluator = engine.create_supervised_evaluator(
        model = model,
        metrics={
            'loss': metrics.Loss(nn.CrossEntropyLoss()),
            "Accuracy@0.3": metrics.Accuracy(thresholded_transform(0.3)),
            "Precision@0.3": metrics.Accuracy(thresholded_transform(0.3)), 
            "IOU": metrics.IoU(metrics.ConfusionMatrix(num_classes = config["n_classes"])),
            "mIOU": metrics.mIoU(metrics.ConfusionMatrix(num_classes = config["n_classes"])),
        },
        device = device,
        non_blocking=True,
        output_transform=lambda x, y, y_pred: (torch.sigmoid(y_pred["out"]), y),
    )

    # attach event listener to do post process after each iteration and epoch
    
    logger.info(f'creating summary writer with tag {config["model_tag"]}')
    writer = tensorboard.SummaryWriter(log_dir=f'logs/{config["model_tag"]}')

    # logger.info('attaching lr scheduler')
    # lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # attach_lr_scheduler(trainer, lr_scheduler, writer)

    logger.info('attaching event driven calls')
    attach_model_checkpoint(trainer, {config["model_tag"]: model.module}, args.name)
    attach_training_logger(trainer, writer=writer)

    attach_metric_logger(trainer, evaluator, 'train', train_metrics_loader, writer)
    attach_metric_logger(trainer, evaluator, 'val', val_metrics_loader, writer)

    # start training (evaluation is included too)
    logger.info('training...')
    trainer.run(train_loader, max_epochs=config["epochs"])



if __name__ == '__main__':
    train()
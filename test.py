import torch
from torch.utils.data import DataLoader
from torch.utils import tensorboard
import torch.nn.functional as F
from torchvision import transforms

from ignite import engine
from ignite import metrics
from ignite import handlers

from models import BiSeNetV2
from utils import AerialDataset
from utils import thresholded_transform, getFlopsandParams
from utils import init_logging, read_config

import argparse
import shutil
import os 
import logging

import matplotlib
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


def test_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str, default="potsdam")
    parser.add_argument("--model",type=str, required=True)  # /home/admin/segmentation/task2/checkpoints/
    parser.add_argument("--name",type=str, required=True)

    parser.add_argument("--nocuda", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()



colors = {0:(255,255,255),1:(0,0,255),2:(0,255,255),3:(0,255,0),4:(255,255,0),5:(255,0,0)}


def attach_metric_logger(
    trainer: engine.Engine,
    data_name: str,
    writer: tensorboard.SummaryWriter,
):
    @trainer.on(engine.Events.EPOCH_COMPLETED)
    def log_metrics(engine):
        y_pred, y = engine.state.output

        metrics = engine.state.metrics
        message = ''
        for metric_name, metric_value in metrics.items():
            message += f'{metric_name}: {metric_value} '

        logger.info(message)


if __name__ == "__main__":
    
    args = test_parse()
    config = read_config()
    try:
        if args.overwrite:
            shutil.rmtree(f"./logs/{args.name}", ignore_errors=True)
        os.mkdir(f"./logs/{args.name}")
    except:
        print(f"log folder {args.name} already exits.")

    init_logging(log_path = f"./logs/{args.name}")

    n_classes = 6
    batch_size = 1
    num_workers = 0
    device = "cpu" if args.nocuda else "cuda"
    # overwrite the folder if exists
    if args.overwrite:
        shutil.rmtree(f"./results/{args.name}", ignore_errors=True)
        os.mkdir(f"./results/{args.name}")


    model = BiSeNetV2(n_classes)
    model.load_state_dict(torch.load(args.model)).to(device)

    dataset = args.dataset
    img_path = f"/home/admin/segmentation/task2/data/{dataset}/train/cropped/images/val"
    label_path = f"/home/admin/segmentation/task2/data/{dataset}/train/cropped/masks/val"

    testdataset = AerialDataset("val", dataset, img_path, label_path)
    testloader = DataLoader(testdataset, batch_size=16, pin_memory=True, drop_last=True,)

    evaluator = engine.create_supervised_evaluator(
        model = model,
        metrics={
            "Accuracy@0.3": metrics.Accuracy(thresholded_transform(0.3)),
            "Precision@0.3": metrics.Accuracy(thresholded_transform(0.3)), 
            "IOU": metrics.IoU(metrics.ConfusionMatrix(num_classes = n_classes)),
            "mIOU": metrics.mIoU(metrics.ConfusionMatrix(num_classes = n_classes)),
            # "FPS": metrics.Frequency(output_transform=lambda x: x[0]),
        },
        device = device,
        non_blocking=True,
        output_transform = lambda x, y, y_pred: (torch.sigmoid(y_pred["out"]), y),
    )
    writer = tensorboard.SummaryWriter(log_dir=f'summary/{config["model_tag"]}')

    attach_metric_logger(evaluator, 'val', writer)
    evaluator.run(testloader)

    getFlopsandParams(model, testdataset[0][0].unsqueeze(0))

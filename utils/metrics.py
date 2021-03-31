import torch
import torch.nn.functional  as F
from ignite import metrics

from typing import Callable, Dict, Sequence, Tuple, Union, cast
from thop import profile
import time

def thresholded_transform(threshold: float):
    def _fn(items):
        y_pred, y = items # y is Tensor.LongTensor target
        y_pred = F.threshold(y_pred, threshold, 0) # restrict the value to [threshold, 1] 
        y_pred = torch.round(y_pred).long()
        y = y.long()
        return y_pred, y

    return _fn


def getFlopsandParams(model, data):
    # data should have the shape of (batch_size, channels, width, height)
    flops, params = profile(model, inputs = (data,))
    print(f"FLOPS: {flops} PARAMS:{params}")
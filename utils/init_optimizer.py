from torch import optim
from functools import partial

def init_optimizer(option):

    if option["optimizer"] == "Adam":
        return  optim.Adam
    elif option["optimizer"] == "SGD":
        return partial(optim.SGD, weight_decay=option["weight_decay"], momentum=option["momentum"])  


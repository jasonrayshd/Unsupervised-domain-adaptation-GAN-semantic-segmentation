from torch import nn

def init_loss(loss_fn):
    if loss_fn == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    elif loss_fn == "BCELoss":
        return nn.BCELoss()
    elif loss_fn == "MSELoss":
        return nn.MSELoss()
    
class LossWithAux(nn.Module):
    def __init__(self, loss_fn: nn.Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, y_pred, y):

        y_pred_out = y_pred["out"]
        loss_output = self.loss_fn(y_pred_out, y)

        loss_aux = [self.loss_fn(y_pred_aux, y) for idx, y_pred_aux in enumerate(y_pred.values()) if idx!=0]
        loss_aux = sum(loss_aux)

        return loss_output + 0.5 * loss_aux
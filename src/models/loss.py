import torch
import torch.nn as nn
from torch.nn import functional as F


def get_loss_module(config):

    task = config['task']

    if (task == "imputation") or (task == "transduction"):
        return MaskedMSELoss(reduction='none')  # outputs loss for each batch element

    if task == "classification":
        return NoFussFocalLoss(reduction='none')  # outputs loss for each batch sample

    if task == "regression":
        return nn.MSELoss(reduction='none')  # outputs loss for each batch sample

    else:
        raise ValueError("Loss module for task '{}' does not exist".format(task))


def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""

    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))


class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

class NoFussFocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Always expects alpha to be a 2-element vector: [alpha_class0, alpha_class1].
    """

    def __init__(self,alpha=[0.1, 0.9], gamma=2.0, reduction="mean"):
        """
        alpha: list or tensor of 2 floats (class weights for 2 classes)
        gamma: focusing parameter
        reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inp, target):
        """
        inp: [batch_size, 2] logits
        target: [batch_size] class indices (0 or 1)
        """
        target = target.long().squeeze()

        log_probs = F.log_softmax(inp, dim=1)
        probs = torch.exp(log_probs)

        # one-hot encode targets
        targets_one_hot = F.one_hot(target, num_classes=inp.size(1)).float()

        # probability of the true class
        pt = (probs * targets_one_hot).sum(dim=1)

        # pick alpha per sample based on target class
        alpha_t = self.alpha[target]

        # focal weight
        focal_weight = alpha_t * (1 - pt) ** self.gamma

        # standard CE
        ce_loss = -(log_probs * targets_one_hot).sum(dim=1)

        loss = focal_weight * ce_loss

        # reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss



class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
    """

    def __init__(self, reduction: str = 'mean'):

        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)

import torch
from pytorch_metric_learning.losses import TripletMarginLoss


class HingeLoss(torch.nn.Module):

    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, output, target):
        hinge_loss = 1 - torch.mul(output, target)
        hinge_loss[hinge_loss < 0] = 0
        mean_hinge_loss = torch.mean(hinge_loss)
        return mean_hinge_loss


class LowerBoundLoss(torch.nn.Module):

    def __init__(self):
        super(LowerBoundLoss, self).__init__()

    def forward(self, output):
        max_loss = torch.clamp(output, min=0, max=None)
        mean_max_loss = torch.mean(max_loss)
        return mean_max_loss


class LogSumExpLoss(torch.nn.Module):
    def __init__(self):
        super(LogSumExpLoss, self).__init__()

    def forward(self, x, return_mean=True):
        if len(x.shape) == 0:
            final_loss = torch.mean(x)
        else:
            if x.shape[-1] == 1:
                dim = len(x.shape) - 1
            else:
                x = x.unsqueeze(len(x.shape))
                dim = len(x.shape) - 1
            zeros = torch.zeros(x.size(dim - 1), dtype=x.dtype, device=x.device).unsqueeze(dim)
            x = torch.cat([x, zeros], dim=dim)

            loss = torch.logsumexp(x, dim=dim, keepdim=True)
            if return_mean:
                final_loss = torch.mean(loss)
            else:
                final_loss = loss
        return final_loss


class TripletCustomMarginLoss(TripletMarginLoss):
    """
    Args:
        margin: The desired difference between the anchor-positive distance and the
                anchor-negative distance.
        swap: Use the positive-negative distance instead of anchor-negative distance,
              if it violates the margin more.
        smooth_loss: Use the log-exp version of the triplet loss
    """

    def __init__(
            self,
            margin=0.05,
            swap=False,
            smooth_loss=False,
            triplets_per_anchor="all",
            **kwargs
    ):
        super().__init__(margin=margin, swap=swap, smooth_loss=smooth_loss, triplets_per_anchor=triplets_per_anchor,
                         **kwargs)

    def set_margin(self, margin):
        self.margin = margin

"""L1/MAE pixel accuracy loss."""

import torch.nn as nn


class PixelAccuracyLoss(nn.Module):
    """L1 loss on [0,1]-normalized images. Input expected in [-1,1]."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        pred01 = (pred + 1.0) * 0.5
        tgt01 = (target + 1.0) * 0.5
        return self.l1(pred01, tgt01)

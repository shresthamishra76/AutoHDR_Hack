"""Weighted composite of all competition-aligned losses."""

import torch.nn as nn
from .edge_similarity import EdgeSimilarityLoss
from .line_straightness import LineStraightnessLoss
from .gradient_orientation import GradientOrientationLoss
from .ssim_loss import SSIMLoss
from .pixel_accuracy import PixelAccuracyLoss


class WeightedCompositeLoss(nn.Module):
    """Combines 5 competition-metric proxy losses with configurable weights.

    Returns (total_loss, details_dict) where details_dict maps loss names
    to their individual (unweighted) scalar values.
    """

    def __init__(self, w_edge=0.40, w_line=0.22, w_grad=0.18, w_ssim=0.15, w_pixel=0.05):
        super().__init__()
        self.w_edge = w_edge
        self.w_line = w_line
        self.w_grad = w_grad
        self.w_ssim = w_ssim
        self.w_pixel = w_pixel

        self.edge_loss = EdgeSimilarityLoss()
        self.line_loss = LineStraightnessLoss()
        self.grad_loss = GradientOrientationLoss()
        self.ssim_loss = SSIMLoss()
        self.pixel_loss = PixelAccuracyLoss()

    def forward(self, pred, target):
        l_edge = self.edge_loss(pred, target)
        l_line = self.line_loss(pred, target)
        l_grad = self.grad_loss(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        l_pixel = self.pixel_loss(pred, target)

        total = (self.w_edge * l_edge + self.w_line * l_line +
                 self.w_grad * l_grad + self.w_ssim * l_ssim +
                 self.w_pixel * l_pixel)

        details = {
            "edge": l_edge.item(),
            "line": l_line.item(),
            "grad": l_grad.item(),
            "ssim": l_ssim.item(),
            "pixel": l_pixel.item(),
        }
        return total, details

"""Differentiable multi-scale edge F1 loss (proxy for Canny edge similarity)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeSimilarityLoss(nn.Module):
    """Soft multi-scale edge F1 loss using Sobel gradients and sigmoid thresholding."""

    def __init__(self, steepness_values=(5.0, 10.0, 20.0)):
        super().__init__()
        self.steepness_values = steepness_values
        # Sobel kernels (3x3)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        # Shape: (1, 1, 3, 3) for F.conv2d
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _to_gray(self, img):
        """Convert [-1,1] RGB to [0,1] grayscale."""
        img01 = (img + 1.0) * 0.5
        return 0.299 * img01[:, 0:1] + 0.587 * img01[:, 1:2] + 0.114 * img01[:, 2:3]

    def _edge_magnitude(self, gray):
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return (gx ** 2 + gy ** 2).sqrt()

    def _soft_f1(self, pred_edges, tgt_edges, eps=1e-7):
        intersection = (pred_edges * tgt_edges).sum()
        return 2.0 * intersection / (pred_edges.sum() + tgt_edges.sum() + eps)

    def forward(self, pred, target):
        gray_pred = self._to_gray(pred)
        gray_tgt = self._to_gray(target)

        mag_pred = self._edge_magnitude(gray_pred)
        mag_tgt = self._edge_magnitude(gray_tgt)

        f1_scores = []
        for s in self.steepness_values:
            # Soft threshold at mean edge magnitude
            pred_edges = torch.sigmoid(s * (mag_pred - mag_pred.mean()))
            tgt_edges = torch.sigmoid(s * (mag_tgt - mag_tgt.mean()))
            f1_scores.append(self._soft_f1(pred_edges, tgt_edges))

        mean_f1 = torch.stack(f1_scores).mean()
        return 1.0 - mean_f1

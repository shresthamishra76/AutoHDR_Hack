"""Gradient direction histogram cosine similarity loss."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientOrientationLoss(nn.Module):
    """Soft gradient histogram loss using Gaussian kernel binning.

    Builds a magnitude-weighted orientation histogram for pred and target,
    then computes 1 - cosine_similarity.
    """

    def __init__(self, n_bins=36, sigma=0.15):
        super().__init__()
        self.n_bins = n_bins
        self.sigma = sigma
        # Bin centers from -pi to pi
        centers = torch.linspace(-math.pi, math.pi, n_bins + 1)[:-1]
        centers = centers + (math.pi / n_bins)  # center of each bin
        self.register_buffer("centers", centers)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _to_gray(self, img):
        img01 = (img + 1.0) * 0.5
        return 0.299 * img01[:, 0:1] + 0.587 * img01[:, 1:2] + 0.114 * img01[:, 2:3]

    def _soft_histogram(self, theta, mag):
        """Build a magnitude-weighted soft orientation histogram."""
        # theta: (N, 1, H, W), mag: (N, 1, H, W)
        # centers: (n_bins,)
        N = theta.shape[0]
        theta_flat = theta.view(N, -1, 1)      # (N, HW, 1)
        mag_flat = mag.view(N, -1, 1)           # (N, HW, 1)
        centers = self.centers.view(1, 1, -1)   # (1, 1, n_bins)

        # Gaussian kernel: weight of each pixel to each bin
        # Handle circular distance
        diff = theta_flat - centers
        # Wrap to [-pi, pi]
        diff = diff - 2 * math.pi * torch.round(diff / (2 * math.pi))
        weights = torch.exp(-0.5 * (diff / self.sigma) ** 2)  # (N, HW, n_bins)
        hist = (weights * mag_flat).sum(dim=1)                 # (N, n_bins)
        # Normalize to sum=1
        hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)
        return hist

    def forward(self, pred, target):
        gray_pred = self._to_gray(pred)
        gray_tgt = self._to_gray(target)

        gx_p = F.conv2d(gray_pred, self.sobel_x, padding=1)
        gy_p = F.conv2d(gray_pred, self.sobel_y, padding=1)
        gx_t = F.conv2d(gray_tgt, self.sobel_x, padding=1)
        gy_t = F.conv2d(gray_tgt, self.sobel_y, padding=1)

        mag_p = (gx_p ** 2 + gy_p ** 2).sqrt()
        mag_t = (gx_t ** 2 + gy_t ** 2).sqrt()
        theta_p = torch.atan2(gy_p, gx_p)
        theta_t = torch.atan2(gy_t, gx_t)

        hist_p = self._soft_histogram(theta_p, mag_p)
        hist_t = self._soft_histogram(theta_t, mag_t)

        cos_sim = F.cosine_similarity(hist_p, hist_t, dim=1).mean()
        return 1.0 - cos_sim

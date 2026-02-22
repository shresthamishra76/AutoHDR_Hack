"""Windowed SSIM loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """Structural Similarity Index loss: 1 - SSIM.

    Uses a Gaussian window for local statistics.
    Input expected in [-1, 1] range (converted to [0, 1] internally).
    """

    def __init__(self, window_size=11, sigma=1.5, C1=(0.01) ** 2, C2=(0.03) ** 2):
        super().__init__()
        self.C1 = C1
        self.C2 = C2
        # Build 2D Gaussian window
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) * g.unsqueeze(0)  # outer product
        # Shape: (1, 1, K, K) — will be expanded to match channels
        self.register_buffer("window", window.unsqueeze(0).unsqueeze(0))
        self.window_size = window_size

    def forward(self, pred, target):
        # Convert [-1,1] to [0,1]
        pred01 = (pred + 1.0) * 0.5
        tgt01 = (target + 1.0) * 0.5

        C = pred01.shape[1]  # channels
        # Expand window to match channel count
        window = self.window.expand(C, 1, -1, -1)
        pad = self.window_size // 2

        mu_p = F.conv2d(pred01, window, padding=pad, groups=C)
        mu_t = F.conv2d(tgt01, window, padding=pad, groups=C)

        mu_p_sq = mu_p ** 2
        mu_t_sq = mu_t ** 2
        mu_pt = mu_p * mu_t

        sigma_p_sq = F.conv2d(pred01 ** 2, window, padding=pad, groups=C) - mu_p_sq
        sigma_t_sq = F.conv2d(tgt01 ** 2, window, padding=pad, groups=C) - mu_t_sq
        sigma_pt = F.conv2d(pred01 * tgt01, window, padding=pad, groups=C) - mu_pt

        ssim_map = ((2 * mu_pt + self.C1) * (2 * sigma_pt + self.C2)) / \
                   ((mu_p_sq + mu_t_sq + self.C1) * (sigma_p_sq + sigma_t_sq + self.C2))

        return 1.0 - ssim_map.mean()

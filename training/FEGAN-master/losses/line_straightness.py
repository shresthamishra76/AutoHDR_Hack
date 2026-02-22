"""Edge curvature proxy for Hough line straightness."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LineStraightnessLoss(nn.Module):
    """Penalize curvature at edge locations as a differentiable proxy for Hough straightness.

    Straight lines have consistent gradient orientation along their length.
    This loss measures spatial variation of gradient angle, weighted by edge magnitude.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _to_gray(self, img):
        img01 = (img + 1.0) * 0.5
        return 0.299 * img01[:, 0:1] + 0.587 * img01[:, 1:2] + 0.114 * img01[:, 2:3]

    def forward(self, pred, target):
        """Loss is computed on pred only; target is unused (kept for API consistency)."""
        gray = self._to_gray(pred)

        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)

        mag = (gx ** 2 + gy ** 2 + 1e-8).sqrt()
        theta = torch.atan2(gy, gx)  # [-pi, pi]

        # Use sin/cos to avoid angle wrapping issues at +/-pi
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)

        # Spatial variation of orientation via finite differences
        dsin_dx = (sin_t[:, :, :, 1:] - sin_t[:, :, :, :-1]).abs()
        dsin_dy = (sin_t[:, :, 1:, :] - sin_t[:, :, :-1, :]).abs()
        dcos_dx = (cos_t[:, :, :, 1:] - cos_t[:, :, :, :-1]).abs()
        dcos_dy = (cos_t[:, :, 1:, :] - cos_t[:, :, :-1, :]).abs()

        # Angular variation at each pixel (average of x and y directions)
        # Trim magnitude to match finite difference sizes
        mag_x = torch.min(mag[:, :, :, 1:], mag[:, :, :, :-1])
        mag_y = torch.min(mag[:, :, 1:, :], mag[:, :, :-1, :])

        curvature_x = (dsin_dx + dcos_dx) * mag_x
        curvature_y = (dsin_dy + dcos_dy) * mag_y

        loss = curvature_x.mean() + curvature_y.mean()
        return loss

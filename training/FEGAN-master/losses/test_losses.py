"""Smoke tests for competition-aligned losses."""

import sys
import torch

sys.path.insert(0, "..")
from losses.edge_similarity import EdgeSimilarityLoss
from losses.line_straightness import LineStraightnessLoss
from losses.gradient_orientation import GradientOrientationLoss
from losses.ssim_loss import SSIMLoss
from losses.pixel_accuracy import PixelAccuracyLoss
from losses.weighted_composite import WeightedCompositeLoss


def test_scalar_output():
    """Each loss returns a scalar tensor."""
    x = torch.randn(2, 3, 64, 64)
    y = torch.randn(2, 3, 64, 64)
    for name, Loss in [
        ("EdgeSimilarity", EdgeSimilarityLoss),
        ("LineStraightness", LineStraightnessLoss),
        ("GradientOrientation", GradientOrientationLoss),
        ("SSIM", SSIMLoss),
        ("PixelAccuracy", PixelAccuracyLoss),
    ]:
        loss = Loss()(x, y)
        assert loss.dim() == 0, f"{name}: expected scalar, got shape {loss.shape}"
        assert torch.isfinite(loss), f"{name}: non-finite loss {loss.item()}"
        print(f"  {name}: {loss.item():.6f} (scalar, finite)")


def test_gradient_flow():
    """Gradients flow through all losses back to input."""
    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    y = torch.randn(2, 3, 64, 64)
    for name, Loss in [
        ("EdgeSimilarity", EdgeSimilarityLoss),
        ("LineStraightness", LineStraightnessLoss),
        ("GradientOrientation", GradientOrientationLoss),
        ("SSIM", SSIMLoss),
        ("PixelAccuracy", PixelAccuracyLoss),
    ]:
        loss = Loss()(x, y)
        loss.backward()
        assert x.grad is not None, f"{name}: no gradient"
        assert torch.isfinite(x.grad).all(), f"{name}: non-finite gradient"
        print(f"  {name}: gradient flows (grad norm = {x.grad.norm().item():.6f})")
        x.grad = None


def test_identical_near_zero():
    """Identical inputs should produce near-zero loss (except line straightness)."""
    x = torch.randn(2, 3, 64, 64)
    for name, Loss in [
        ("EdgeSimilarity", EdgeSimilarityLoss),
        ("GradientOrientation", GradientOrientationLoss),
        ("SSIM", SSIMLoss),
        ("PixelAccuracy", PixelAccuracyLoss),
    ]:
        loss = Loss()(x, x)
        assert loss.item() < 0.05, f"{name}: identical inputs gave loss {loss.item():.6f} (expected ~0)"
        print(f"  {name}: {loss.item():.6f} (near zero for identical inputs)")


def test_composite():
    """WeightedCompositeLoss returns (scalar, dict)."""
    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    y = torch.randn(2, 3, 64, 64)
    composite = WeightedCompositeLoss()
    total, details = composite(x, y)
    assert total.dim() == 0, f"Composite: expected scalar, got {total.shape}"
    assert torch.isfinite(total), f"Composite: non-finite total {total.item()}"
    total.backward()
    assert x.grad is not None, "Composite: no gradient"
    print(f"  Composite total: {total.item():.6f}")
    for k, v in details.items():
        print(f"    {k}: {v:.6f}")


if __name__ == "__main__":
    print("Test 1: Scalar output")
    test_scalar_output()
    print("\nTest 2: Gradient flow")
    test_gradient_flow()
    print("\nTest 3: Identical inputs -> near-zero loss")
    test_identical_near_zero()
    print("\nTest 4: Composite loss")
    test_composite()
    print("\nAll tests passed!")

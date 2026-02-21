"""
FEGAN Performance Evaluation
=============================
Computes image-quality metrics for a results directory produced by
run_inference.py (or the patched train.py) and generates a markdown report.

Usage:
    python evaluate.py --results_dir results/my_exp

    # Auto-find the latest results directory:
    python evaluate.py

Output:
    results/{name}/report.md    -- full markdown performance report
    results/{name}/loss_curve.png  -- loss curve (if loss_log.txt exists)
"""

import argparse
import os
import sys
import re
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "training", "FEGAN-master"))
from util.metric import evaluate as fegan_evaluate


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def find_latest_results(results_root="./results"):
    root = Path(results_root)
    if not root.exists():
        return None
    candidates = [d for d in root.iterdir() if d.is_dir() and (d / "output").is_dir()]
    if not candidates:
        return None
    return str(max(candidates, key=lambda d: d.stat().st_mtime))


def find_images(directory):
    images = []
    for entry in sorted(Path(directory).iterdir()):
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(entry)
    return images


def compute_metrics(output_dir, gt_dir):
    output_images = find_images(output_dir)
    gt_images = {p.stem.lower(): p for p in find_images(gt_dir)}

    results = []
    for out_path in output_images:
        gt_key = out_path.stem.lower()
        if gt_key not in gt_images:
            print(f"  Warning: no ground truth for {out_path.name}, skipping")
            continue

        out_img = cv2.imread(str(out_path))
        gt_img = cv2.imread(str(gt_images[gt_key]))

        if out_img is None or gt_img is None:
            print(f"  Warning: could not read {out_path.name} or its GT, skipping")
            continue

        h, w = out_img.shape[:2]
        gt_img = cv2.resize(gt_img, (w, h))

        score = fegan_evaluate(gt_img, out_img)
        score["name"] = out_path.stem
        results.append(score)
        print(f"  {out_path.name}: RMSE={score['rmse']:.3f}  PSNR={score['psnr']:.2f}  "
              f"SSIM={score['ssim']:.4f}  Hough={score['hough']:.4f}")

    return results


def aggregate_metrics(results):
    if not results:
        return {}
    metrics = ["rmse", "psnr", "ssim", "hough"]
    agg = {}
    for m in metrics:
        vals = [r[m] for r in results]
        agg[m] = {
            "mean": np.mean(vals),
            "std": np.std(vals),
            "min": np.min(vals),
            "max": np.max(vals),
        }
    return agg


def parse_loss_log(log_path):
    """Parse FEGAN loss_log.txt into structured data for plotting."""
    entries = []
    with open(log_path) as f:
        for line in f:
            match = re.match(r"\(epoch:\s*(\d+),\s*iters:\s*(\d+),\s*time:\s*[\d.]+\)\s*(.*)", line.strip())
            if not match:
                continue
            epoch = int(match.group(1))
            iters = int(match.group(2))
            loss_str = match.group(3)
            losses = {}
            for pair in re.findall(r"(\S+):\s*([\d.]+)", loss_str):
                losses[pair[0]] = float(pair[1])
            entries.append({"epoch": epoch, "iters": iters, "losses": losses})
    return entries


def plot_loss_curve(entries, output_path):
    if not entries:
        return False

    loss_keys = list(entries[0]["losses"].keys())
    x = list(range(len(entries)))

    plt.figure(figsize=(10, 6))
    for key in loss_keys:
        y = [e["losses"].get(key, 0) for e in entries]
        plt.plot(x, y, label=key, alpha=0.8)

    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def parse_config(results_dir):
    """Look for opt_train.txt in the results directory or checkpoints."""
    for candidate in [
        os.path.join(results_dir, "opt_train.txt"),
        os.path.join(results_dir, "checkpoints", "opt_train.txt"),
    ]:
        if os.path.isfile(candidate):
            config = {}
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("---") or not line:
                        continue
                    parts = line.split(": ", 1)
                    if len(parts) == 2:
                        config[parts[0]] = parts[1]
            return config
    return None


def generate_report(results_dir, metrics, agg, loss_plotted, config):
    name = Path(results_dir).name
    lines = []
    lines.append(f"# Performance Report: {name}")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Summary table
    lines.append("## Summary Metrics")
    lines.append("")
    lines.append("| Metric | Mean | Std | Min | Max | Ideal |")
    lines.append("|--------|------|-----|-----|-----|-------|")
    if agg:
        metric_info = {
            "rmse":  ("RMSE",  "Lower"),
            "psnr":  ("PSNR",  "Higher"),
            "ssim":  ("SSIM",  "Higher (max 1.0)"),
            "hough": ("Hough", "Lower"),
        }
        for key, (label, ideal) in metric_info.items():
            if key in agg:
                a = agg[key]
                lines.append(f"| {label} | {a['mean']:.4f} | {a['std']:.4f} | "
                             f"{a['min']:.4f} | {a['max']:.4f} | {ideal} |")
    lines.append("")

    # Per-image table
    if metrics:
        lines.append("## Per-Image Breakdown")
        lines.append("")
        lines.append("| Image | RMSE | PSNR | SSIM | Hough |")
        lines.append("|-------|------|------|------|-------|")
        for m in metrics:
            lines.append(f"| {m['name']} | {m['rmse']:.4f} | {m['psnr']:.2f} | "
                         f"{m['ssim']:.4f} | {m['hough']:.4f} |")
        lines.append("")

    # Sample comparisons
    input_dir = os.path.join(results_dir, "input")
    output_dir = os.path.join(results_dir, "output")
    gt_dir = os.path.join(results_dir, "ground_truth")

    sample_images = []
    if os.path.isdir(output_dir):
        sample_images = find_images(output_dir)[:5]

    if sample_images:
        lines.append("## Sample Comparisons")
        lines.append("")
        for img_path in sample_images:
            name_stem = img_path.name
            lines.append(f"### {img_path.stem}")
            lines.append("")
            row = []
            if os.path.isfile(os.path.join(input_dir, name_stem)):
                row.append(f"| Input | ![](input/{name_stem}) |")
            if os.path.isfile(os.path.join(output_dir, name_stem)):
                row.append(f"| Output | ![](output/{name_stem}) |")
            if os.path.isfile(os.path.join(gt_dir, name_stem)):
                row.append(f"| Ground Truth | ![](ground_truth/{name_stem}) |")
            if row:
                lines.append("| | |")
                lines.append("|---|---|")
                lines.extend(row)
                lines.append("")

    # Loss curve
    if loss_plotted:
        lines.append("## Training Loss Curve")
        lines.append("")
        lines.append("![Training Loss](loss_curve.png)")
        lines.append("")

    # Training config
    if config:
        lines.append("## Training Configuration")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        important_keys = [
            "model", "which_model_netG", "fineSize", "loadSize", "batchSize",
            "niter", "niter_decay", "lr", "lambda_AB", "lambda_crossflow",
            "lambda_radial", "lambda_smooth", "lambda_rot", "upsample_flow",
            "norm", "init_type", "gpu_ids",
        ]
        for key in important_keys:
            if key in config:
                lines.append(f"| {key} | {config[key]} |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FEGAN results and generate a performance report.")
    parser.add_argument("--results_dir", default=None,
                        help="Path to the results directory (auto-detects latest if omitted)")
    parser.add_argument("--results_root", default="./results",
                        help="Root results directory for auto-detection")
    args = parser.parse_args()

    results_dir = args.results_dir
    if results_dir is None:
        results_dir = find_latest_results(args.results_root)
        if results_dir is None:
            print("No results directory found. Run inference first or specify --results_dir.")
            sys.exit(1)
        print(f"Auto-detected results directory: {results_dir}")

    output_dir = os.path.join(results_dir, "output")
    gt_dir = os.path.join(results_dir, "ground_truth")

    if not os.path.isdir(output_dir):
        print(f"Error: output directory not found at {output_dir}")
        sys.exit(1)

    metrics = []
    agg = {}
    if os.path.isdir(gt_dir) and find_images(gt_dir):
        print("Computing metrics...")
        metrics = compute_metrics(output_dir, gt_dir)
        agg = aggregate_metrics(metrics)
        print(f"\nEvaluated {len(metrics)} image pairs.")
    else:
        print("No ground truth directory found -- skipping metric computation.")
        print("To compute metrics, provide ground truth images via run_inference.py --gt_dir")

    # Loss curve
    loss_plotted = False
    loss_log = os.path.join(results_dir, "loss_log.txt")
    if os.path.isfile(loss_log):
        print("Parsing training loss log...")
        entries = parse_loss_log(loss_log)
        curve_path = os.path.join(results_dir, "loss_curve.png")
        loss_plotted = plot_loss_curve(entries, curve_path)
        if loss_plotted:
            print(f"  Loss curve saved to {curve_path}")

    config = parse_config(results_dir)

    print("Generating report...")
    report = generate_report(results_dir, metrics, agg, loss_plotted, config)
    report_path = os.path.join(results_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nReport saved to {report_path}")

    if agg:
        print("\n--- Quick Summary ---")
        print(f"  RMSE:  {agg['rmse']['mean']:.4f} (lower is better)")
        print(f"  PSNR:  {agg['psnr']['mean']:.2f} dB (higher is better)")
        print(f"  SSIM:  {agg['ssim']['mean']:.4f} (higher is better, max 1.0)")
        print(f"  Hough: {agg['hough']['mean']:.4f} (lower is better)")


if __name__ == "__main__":
    main()

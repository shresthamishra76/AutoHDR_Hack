# FEGAN Colab Scripts Guide

Three copy-paste-ready scripts for Google Colab: **Preprocessing**, **Training**, and **Submission**.

---

## Script 1: Preprocessing (Multi-Worker)

This script clones the repo, installs deps, and runs the preprocessing pipeline with parallel workers to convert raw Kaggle data into the FEGAN directory structure.

```python
# ============================================================
# CELL 1: Setup & Clone
# ============================================================
!git clone https://github.com/<YOUR_USERNAME>/AutoHDR_Hack.git /content/AutoHDR_Hack
%cd /content/AutoHDR_Hack

# Install preprocessing dependencies
!pip install -q Pillow numpy PyYAML

# Install FEGAN training + submission dependencies (install now to save time)
!pip install -q torch torchvision numpy matplotlib Pillow opencv-python \
    scikit-image sewar dominate beautifulsoup4 requests tensorboard \
    realesrgan basicsr

# ============================================================
# CELL 2: Mount Google Drive (optional - for saving checkpoints)
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================================
# CELL 3: Verify Kaggle dataset is available
# ============================================================
import os

# Option A: If using Kaggle competition data directly
# Upload kaggle.json first, then:
# !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# !kaggle competitions download -c automatic-lens-correction -p /content/kaggle_data
# !unzip -q /content/kaggle_data/*.zip -d /content/kaggle_data

# Option B: If dataset is already in a known path
INPUT_DIR = "/content/lens-correction-train-cleaned"   # <-- ADJUST THIS
TEST_DIR  = "/content/test-originals"                  # <-- ADJUST THIS
OUTPUT_DIR = "/content/datasets/fisheye"

assert os.path.isdir(INPUT_DIR), f"Input dir not found: {INPUT_DIR}"
print(f"Input directory: {INPUT_DIR}")
print(f"Files found: {len(os.listdir(INPUT_DIR))}")

# ============================================================
# CELL 4: Run Preprocessing with Parallel Workers
# ============================================================
import multiprocessing
NUM_WORKERS = multiprocessing.cpu_count()  # Colab typically has 2 CPUs
print(f"Using {NUM_WORKERS} workers")

!python preprocessing/preprocess.py \
    --config preprocessing/config.yaml \
    --input_dir "{INPUT_DIR}" \
    --output_dir "{OUTPUT_DIR}" \
    --target_size 320 \
    --output_format png \
    --train_split 0.9 \
    --val_split 0.1 \
    --seed 42 \
    --workers {NUM_WORKERS}

# ============================================================
# CELL 5: Verify output structure
# ============================================================
for split in ["trainA", "trainB", "trainBtoA", "valA", "valB", "valBtoA"]:
    d = os.path.join(OUTPUT_DIR, split)
    count = len(os.listdir(d)) if os.path.isdir(d) else 0
    print(f"  {split}: {count} images")

# Quick-test: limit to small subset for debugging (optional)
# !python preprocessing/preprocess.py \
#     --config preprocessing/config.yaml \
#     --input_dir "{INPUT_DIR}" \
#     --output_dir "/content/datasets/fisheye_small" \
#     --target_size 320 \
#     --max_pairs 100 \
#     --workers {NUM_WORKERS}
```

---

## Script 2: Training (All Hyperparameter Flags)

Run FEGAN training with the full set of configurable hyperparameters. Annotated so you can tune each flag.

```python
# ============================================================
# CELL 1: Set paths
# ============================================================
import os

REPO_ROOT   = "/content/AutoHDR_Hack"
FEGAN_ROOT  = f"{REPO_ROOT}/training/FEGAN-master"
DATAROOT    = "/content/datasets/fisheye"        # Output from preprocessing
RESULTS_DIR = "/content/results"                 # Where checkpoints/logs go
EXP_NAME    = "fisheye_rectification"            # Experiment name

os.makedirs(RESULTS_DIR, exist_ok=True)

# Verify dataset exists
for d in ["trainA", "trainB"]:
    assert os.path.isdir(os.path.join(DATAROOT, d)), f"Missing {d}/ in {DATAROOT}"
print("Dataset verified.")

# ============================================================
# CELL 2: Training - FULL CONFIG
# ============================================================
# All flags documented. Uncomment/modify as needed.

%cd {FEGAN_ROOT}

!python train.py \
    --dataroot "{DATAROOT}" \
    --name "{EXP_NAME}" \
    --results_root "{RESULTS_DIR}" \
    \
    `# ── Model Architecture ──` \
    --model gc_gan_cross \
    --which_model_netG unet_128 \
    --which_model_netD Fusion \
    --which_direction BtoA \
    --ngf 64 \
    --ndf 64 \
    --input_nc 3 \
    --output_nc 3 \
    --norm instance \
    --init_type xavier \
    --use_att \
    --upsample_flow 2 \
    \
    `# ── Image Sizing ──` \
    --loadSize 288 \
    --fineSize 256 \
    \
    `# ── Training Schedule ──` \
    --niter 1 \
    --niter_decay 0 \
    --epoch_count 1 \
    --batchSize 64 \
    \
    `# ── Optimizer ──` \
    --lr 0.0002 \
    --beta1 0.5 \
    --lr_policy lambda \
    \
    `# ── Loss Weights ──` \
    --lambda_G 1.0 \
    --lambda_gc 1.0 \
    --lambda_AB 10.0 \
    --lambda_smooth 2.0 \
    --lambda_crossflow 2.0 \
    --lambda_radial 0.5 \
    --lambda_rot 0.1 \
    --identity 0 \
    \
    `# ── Competition Metric Loss (0=disabled) ──` \
    --lambda_metric 0.0 \
    --w_edge 0.40 \
    --w_line 0.22 \
    --w_grad 0.18 \
    --w_ssim 0.15 \
    --w_pixel 0.05 \
    \
    `# ── GAN Type ──` \
    `# --no_lsgan              # Uncomment to use vanilla GAN instead of LSGAN` \
    \
    `# ── Data Augmentation ──` \
    `# --no_flip               # Uncomment to disable random horizontal flips` \
    `# --no_dropout            # Uncomment to disable dropout in generator` \
    `# --no_rot                # Uncomment to disable rotation augmentation` \
    --resize_or_crop resize_and_crop \
    \
    `# ── Saving & Logging ──` \
    --save_epoch_freq 1 \
    --save_latest_freq 5000 \
    --print_freq 10 \
    --display_freq 10 \
    --nThreads 2 \
    --gpu_ids 0 \
    --tensorboard

# ============================================================
# CELL 3: Monitor training with TensorBoard (run in parallel)
# ============================================================
%load_ext tensorboard

# TensorBoard logs are in the results or checkpoints directory
%tensorboard --logdir "{RESULTS_DIR}/{EXP_NAME}"

# ============================================================
# CELL 4: Resume training from a checkpoint (if interrupted)
# ============================================================
# Uncomment and adjust epoch_count to resume:

# !python train.py \
#     --dataroot "{DATAROOT}" \
#     --name "{EXP_NAME}" \
#     --results_root "{RESULTS_DIR}" \
#     --model gc_gan_cross \
#     --which_model_netG unet_128 \
#     --which_model_netD Fusion \
#     --which_direction BtoA \
#     --ngf 64 --ndf 64 --input_nc 3 --output_nc 3 \
#     --norm instance --init_type xavier --use_att --upsample_flow 2 \
#     --loadSize 288 --fineSize 256 \
#     --niter 20 --niter_decay 0 \
#     --batchSize 64 --lr 0.0002 --beta1 0.5 --lr_policy lambda \
#     --lambda_G 1.0 --lambda_gc 1.0 --lambda_AB 10.0 \
#     --lambda_smooth 2.0 --lambda_crossflow 2.0 \
#     --lambda_radial 0.5 --lambda_rot 0.1 --identity 0 \
#     --lambda_metric 0.0 \
#     --w_edge 0.40 --w_line 0.22 --w_grad 0.18 --w_ssim 0.15 --w_pixel 0.05 \
#     --save_epoch_freq 1 --print_freq 10 --nThreads 2 \
#     --gpu_ids 0 --tensorboard \
#     --continue_train \
#     --which_epoch latest \
#     --epoch_count 10           # <-- set to epoch where training stopped

# ============================================================
# CELL 5: Copy checkpoints to Google Drive (backup)
# ============================================================
# !cp -r {RESULTS_DIR}/{EXP_NAME}/checkpoints /content/drive/MyDrive/fegan_checkpoints/

# ============================================================
# CELL 6: Verify training completed (check saved checkpoints)
# ============================================================
import os
ckpt_dir = f"{RESULTS_DIR}/{EXP_NAME}/checkpoints"
if os.path.isdir(ckpt_dir):
    ckpts = sorted(os.listdir(ckpt_dir))
    print(f"Checkpoints in {ckpt_dir}:")
    for c in ckpts:
        size_mb = os.path.getsize(os.path.join(ckpt_dir, c)) / (1024 * 1024)
        print(f"  {c} ({size_mb:.1f} MB)")
else:
    print(f"No checkpoints found at {ckpt_dir}")
```

### Hyperparameter Tuning Quick Reference

| Flag | Default | What it controls | Try |
|------|---------|-----------------|-----|
| `--batchSize` | 64 | Batch size. H100 = 32-64, T4 = 4-8 | 32 if OOM on H100 |
| `--lr` | 0.0002 | Adam learning rate | 1e-4 for finer convergence |
| `--niter` | 20 | Epochs at constant LR | 10 for quick test |
| `--niter_decay` | 0 | Epochs to linearly decay LR to 0 | 50 for gradual cooldown |
| `--lambda_AB` | 10.0 | Reconstruction weight (dominant loss) | 5.0-20.0 |
| `--lambda_smooth` | 2.0 | Flow smoothness (reduces artifacts) | 1.0-5.0 |
| `--lambda_crossflow` | 2.0 | Cross-flow consistency | 1.0-5.0 |
| `--lambda_radial` | 0.5 | Radial flow constraint | 0.1-2.0 |
| `--lambda_rot` | 0.1 | Rotation consistency | 0.05-0.5 |
| `--which_model_netG` | unet_128 | Generator arch | `unet_256` for higher cap |
| `--ngf` | 64 | Generator width | 32 (lighter) or 128 (heavier) |
| `--lambda_metric` | 0.0 | Competition metric loss weight | 1.0 after phase 1 |
| `--w_edge` | 0.40 | Edge similarity weight in metric loss | Competition: 40% |
| `--w_line` | 0.22 | Line straightness weight | Competition: 22% |
| `--w_grad` | 0.18 | Gradient orientation weight | Competition: 18% |
| `--w_ssim` | 0.15 | SSIM weight | Competition: 15% |
| `--w_pixel` | 0.05 | Pixel accuracy weight | Competition: 5% |

### Phased Training Strategy

Train in phases to get stable GAN convergence first, then align with competition metrics:

```python
# Ensure CWD is FEGAN-master (already set by Cell 2 above, but safe to repeat)
%cd {FEGAN_ROOT}

# Phase 1 (epochs 1-5): GAN + geometry only
!python train.py \
    --dataroot "{DATAROOT}" --name "{EXP_NAME}" --results_root "{RESULTS_DIR}" \
    --model gc_gan_cross --which_model_netG unet_128 --which_model_netD Fusion \
    --which_direction BtoA --ngf 64 --ndf 64 --input_nc 3 --output_nc 3 \
    --norm instance --init_type xavier --use_att --upsample_flow 2 \
    --loadSize 288 --fineSize 256 --niter 5 --niter_decay 0 --batchSize 64 \
    --lr 0.0002 --beta1 0.5 --lr_policy lambda \
    --lambda_G 1.0 --lambda_gc 1.0 --lambda_AB 10.0 \
    --lambda_smooth 2.0 --lambda_crossflow 2.0 --lambda_radial 0.5 --lambda_rot 0.1 \
    --identity 0 --lambda_metric 0.0 \
    --save_epoch_freq 1 --print_freq 10 --nThreads 2 --gpu_ids 0 --tensorboard

# Phase 2 (epochs 6-15): introduce metric loss
!python train.py \
    --dataroot "{DATAROOT}" --name "{EXP_NAME}" --results_root "{RESULTS_DIR}" \
    --model gc_gan_cross --which_model_netG unet_128 --which_model_netD Fusion \
    --which_direction BtoA --ngf 64 --ndf 64 --input_nc 3 --output_nc 3 \
    --norm instance --init_type xavier --use_att --upsample_flow 2 \
    --loadSize 288 --fineSize 256 --niter 15 --niter_decay 0 --batchSize 4 \
    --lr 0.0002 --beta1 0.5 --lr_policy lambda \
    --lambda_G 1.0 --lambda_gc 1.0 --lambda_AB 10.0 \
    --lambda_smooth 2.0 --lambda_crossflow 2.0 --lambda_radial 0.5 --lambda_rot 0.1 \
    --identity 0 --lambda_metric 1.0 \
    --w_edge 0.40 --w_line 0.22 --w_grad 0.18 --w_ssim 0.15 --w_pixel 0.05 \
    --save_epoch_freq 1 --print_freq 10 --nThreads 2 --gpu_ids 0 --tensorboard \
    --continue_train --which_epoch latest --epoch_count 6

# Phase 3 (epochs 16-20): metric-focused fine-tuning
!python train.py \
    --dataroot "{DATAROOT}" --name "{EXP_NAME}" --results_root "{RESULTS_DIR}" \
    --model gc_gan_cross --which_model_netG unet_128 --which_model_netD Fusion \
    --which_direction BtoA --ngf 64 --ndf 64 --input_nc 3 --output_nc 3 \
    --norm instance --init_type xavier --use_att --upsample_flow 2 \
    --loadSize 288 --fineSize 256 --niter 20 --niter_decay 0 --batchSize 4 \
    --lr 0.0001 --beta1 0.5 --lr_policy lambda \
    --lambda_G 1.0 --lambda_gc 1.0 --lambda_AB 10.0 \
    --lambda_smooth 2.0 --lambda_crossflow 2.0 --lambda_radial 0.5 --lambda_rot 0.1 \
    --identity 0 --lambda_metric 2.0 \
    --w_edge 0.40 --w_line 0.22 --w_grad 0.18 --w_ssim 0.15 --w_pixel 0.05 \
    --save_epoch_freq 1 --print_freq 10 --nThreads 2 --gpu_ids 0 --tensorboard \
    --continue_train --which_epoch latest --epoch_count 16
```

---

## Script 2.5: Evaluation & Visualization (Validation Set)

Run inference on the validation set and visually compare predictions vs ground truth. Re-run the visualization cell to see a different random sample.

```python
# ============================================================
# CELL 1: Set paths (same as training)
# ============================================================
import os

REPO_ROOT   = "/content/AutoHDR_Hack"
DATAROOT    = "/content/datasets/fisheye"
RESULTS_DIR = "/content/results"
EXP_NAME    = "fisheye_rectification"

CHECKPOINT = f"{RESULTS_DIR}/{EXP_NAME}/checkpoints/latest_net_G_AB.pth"
VAL_INPUT  = os.path.join(DATAROOT, "valB")       # distorted validation images
VAL_GT     = os.path.join(DATAROOT, "valA")        # clean ground truth images
VAL_OUTPUT = "/content/val_predictions"             # where predictions go

assert os.path.isfile(CHECKPOINT), f"Checkpoint not found: {CHECKPOINT}"
assert os.path.isdir(VAL_INPUT), f"valB not found: {VAL_INPUT}"
assert os.path.isdir(VAL_GT), f"valA not found: {VAL_GT}"
print(f"Checkpoint: {CHECKPOINT}")
print(f"Val images: {len(os.listdir(VAL_INPUT))} in {VAL_INPUT}")

# ============================================================
# CELL 2: Run inference on validation set
# ============================================================
%cd {REPO_ROOT}

!python submit.py \
    --checkpoint "{CHECKPOINT}" \
    --test_dir "{VAL_INPUT}" \
    --output_dir "{VAL_OUTPUT}" \
    --zip_path "/content/_val_dummy.zip" \
    --size 256 \
    --ngf 64 \
    --which_model unet_128 \
    --use_att \
    --upsample_flow 2.0 \
    --gpu_ids 0 \
    --jpeg_quality 95 \
    --upsampler bicubic

print(f"Predictions: {len(os.listdir(VAL_OUTPUT))} files in {VAL_OUTPUT}")

# ============================================================
# CELL 3: Visualize predictions vs ground truth (re-run for new sample)
# ============================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import glob

pred_dir = VAL_OUTPUT
gt_dir   = VAL_GT

# Find matching filenames between predictions and ground truth
pred_files = {os.path.splitext(f)[0]: os.path.join(pred_dir, f)
              for f in os.listdir(pred_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
gt_files   = {os.path.splitext(f)[0]: os.path.join(gt_dir, f)
              for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}

common = sorted(set(pred_files.keys()) & set(gt_files.keys()))
assert len(common) > 0, "No matching filenames between predictions and ground truth!"

N = min(5, len(common))
sample = random.sample(common, N)
print(f"Showing {N} random samples out of {len(common)} matched images")

fig, axes = plt.subplots(N, 3, figsize=(15, 5 * N))
if N == 1:
    axes = axes[np.newaxis, :]  # ensure 2D indexing

for i, name in enumerate(sample):
    # Load images as RGB
    pred = cv2.imread(pred_files[name])
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)

    gt = cv2.imread(gt_files[name])
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

    # Resize GT to match prediction dimensions if needed
    if gt.shape[:2] != pred.shape[:2]:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Compute difference overlay
    diff = np.abs(pred.astype(np.float32) - gt.astype(np.float32))
    diff_gray = np.mean(diff, axis=2) / 255.0                     # [0, 1]
    heatmap = plt.cm.hot(diff_gray)[:, :, :3]                     # RGB from colormap
    overlay = (0.6 * gt.astype(np.float32) / 255.0
             + 0.4 * heatmap)
    overlay = np.clip(overlay, 0, 1)

    # Plot
    axes[i, 0].imshow(pred)
    axes[i, 0].set_title(f"Prediction: {name}", fontsize=10)
    axes[i, 0].axis("off")

    axes[i, 1].imshow(gt)
    axes[i, 1].set_title(f"Ground Truth: {name}", fontsize=10)
    axes[i, 1].axis("off")

    axes[i, 2].imshow(overlay)
    axes[i, 2].set_title(f"Diff Overlay: {name}", fontsize=10)
    axes[i, 2].axis("off")

plt.suptitle("Prediction vs Ground Truth (hot colormap: black=match, red/yellow=error)", fontsize=13)
plt.tight_layout()
plt.show()
```

---

## Script 3: Submission (Creates submission.zip)

Runs the trained generator on the test set and creates a Kaggle-ready `submission.zip`.

```python
# ============================================================
# CELL 1: Set paths
# ============================================================
import os

REPO_ROOT  = "/content/AutoHDR_Hack"
FEGAN_ROOT = f"{REPO_ROOT}/training/FEGAN-master"
RESULTS_DIR = "/content/results"
EXP_NAME    = "fisheye_rectification"

# Path to the trained generator checkpoint
# After training, checkpoints are at:
#   {RESULTS_DIR}/{EXP_NAME}/checkpoints/latest_net_G_AB.pth
#   {RESULTS_DIR}/{EXP_NAME}/checkpoints/{epoch}_net_G_AB.pth
CHECKPOINT = f"{RESULTS_DIR}/{EXP_NAME}/checkpoints/1_net_G_AB.pth"

# Test images directory
TEST_DIR = "/content/test-originals"     # <-- ADJUST THIS

# Output
SUBMISSION_DIR = "/content/submission"
ZIP_PATH       = "/content/submission.zip"

assert os.path.isfile(CHECKPOINT), f"Checkpoint not found: {CHECKPOINT}"
assert os.path.isdir(TEST_DIR), f"Test dir not found: {TEST_DIR}"
print(f"Checkpoint: {CHECKPOINT}")
print(f"Test images: {len(os.listdir(TEST_DIR))} files in {TEST_DIR}")

# ============================================================
# CELL 2: Run submit.py
# ============================================================
%cd {REPO_ROOT}

!python submit.py \
    --checkpoint "{CHECKPOINT}" \
    --test_dir "{TEST_DIR}" \
    --output_dir "{SUBMISSION_DIR}" \
    --zip_path "{ZIP_PATH}" \
    --size 256 \
    --ngf 64 \
    --which_model unet_128 \
    --use_att \
    --upsample_flow 2.0 \
    --gpu_ids 0 \
    --jpeg_quality 95 \
    --upsampler bicubic
# For better upscaling, use: --upsampler realesrgan (requires: pip install realesrgan basicsr)

# ============================================================
# CELL 3: Verify submission
# ============================================================
import zipfile

with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    names = zf.namelist()
    print(f"submission.zip contains {len(names)} files")
    print(f"First 5: {names[:5]}")
    print(f"Last 5:  {names[-5:]}")

zip_size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
print(f"Zip size: {zip_size_mb:.1f} MB")

# Sanity check: all files are .jpg
non_jpg = [n for n in names if not n.lower().endswith('.jpg')]
if non_jpg:
    print(f"WARNING: {len(non_jpg)} non-JPEG files found: {non_jpg[:5]}")
else:
    print("All files are .jpg")

# ============================================================
# CELL 4: Download submission.zip
# ============================================================
from google.colab import files
files.download(ZIP_PATH)

# ============================================================
# CELL 5: (Alternative) Copy to Google Drive
# ============================================================
# !cp {ZIP_PATH} /content/drive/MyDrive/submissions/submission.zip
# print("Copied to Google Drive")

# ============================================================
# CELL 6: (Optional) Use a specific epoch checkpoint instead
# ============================================================
# EPOCH = 100
# CHECKPOINT_EPOCH = f"{RESULTS_DIR}/{EXP_NAME}/checkpoints/{EPOCH}_net_G_AB.pth"
#
# !python submit.py \
#     --checkpoint "{CHECKPOINT_EPOCH}" \
#     --test_dir "{TEST_DIR}" \
#     --output_dir "/content/submission_epoch{EPOCH}" \
#     --zip_path "/content/submission_epoch{EPOCH}.zip" \
#     --size 256 --ngf 64 --which_model unet_128 \
#     --use_att --upsample_flow 2.0 --gpu_ids 0 --jpeg_quality 95 \
#     --upsampler bicubic
```

---

## Quick Full Pipeline (All 3 Steps in One)

For a fast end-to-end run (e.g., debugging with a small subset):

```python
# Clone + install
!git clone https://github.com/<YOUR_USERNAME>/AutoHDR_Hack.git /content/AutoHDR_Hack
!pip install -q torch torchvision numpy matplotlib Pillow opencv-python \
    scikit-image sewar dominate beautifulsoup4 requests tensorboard PyYAML \
    realesrgan basicsr

# Preprocess (small subset for testing)
!python /content/AutoHDR_Hack/preprocessing/preprocess.py \
    --config /content/AutoHDR_Hack/preprocessing/config.yaml \
    --input_dir "/content/kaggle_data/lens-correction-train-cleaned" \
    --output_dir "/content/datasets/fisheye" \
    --target_size 320 --max_pairs 200 --workers 2

# Train (short run)
%cd /content/AutoHDR_Hack/training/FEGAN-master
!python train.py \
    --dataroot /content/datasets/fisheye \
    --name quick_test --results_root /content/results \
    --model gc_gan_cross --which_model_netG unet_128 --which_model_netD Fusion \
    --which_direction BtoA --ngf 64 --ndf 64 --input_nc 3 --output_nc 3 \
    --norm instance --init_type xavier --use_att --upsample_flow 2 \
    --loadSize 288 --fineSize 256 --niter 5 --niter_decay 0 --batchSize 64 \
    --lr 0.0002 --beta1 0.5 --lr_policy lambda \
    --lambda_G 1.0 --lambda_gc 1.0 --lambda_AB 10.0 \
    --lambda_smooth 2.0 --lambda_crossflow 2.0 --lambda_radial 0.5 --lambda_rot 0.1 \
    --identity 0 --save_epoch_freq 1 --print_freq 10 --nThreads 2 \
    --gpu_ids 0 --tensorboard

# Submit
%cd /content/AutoHDR_Hack
!python submit.py \
    --checkpoint /content/results/quick_test/checkpoints/latest_net_G_AB.pth \
    --test_dir "/content/kaggle_data/test-originals" \
    --output_dir /content/submission --zip_path /content/submission.zip \
    --size 256 --ngf 64 --which_model unet_128 --use_att --upsample_flow 2.0 \
    --gpu_ids 0 --jpeg_quality 95 --upsampler bicubic

from google.colab import files
files.download("/content/submission.zip")
```

---

## Notes

- **GPU**: H100 (80GB VRAM) handles `batchSize 64` at 256x256 easily. Drop to 32 if OOM. For T4 (16GB), use 4-8.
- **Runtime disconnects**: Use `--continue_train --which_epoch latest --epoch_count N` to resume.
- **Checkpoints**: Saved every `save_epoch_freq` epochs and every `save_latest_freq` iterations. Back up to Drive.
- **submit.py flags must match training**: `--ngf`, `--which_model`, `--use_att`, `--upsample_flow` must be identical to what was used during training, or the checkpoint won't load.
- **nThreads**: Set to 2 on Colab (only 2 CPU cores). Higher values cause warnings.

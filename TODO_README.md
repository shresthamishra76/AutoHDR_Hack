# FEGAN Edge Correction Optimization - TODO & Implementation Guide

## Overview
This document outlines the required modifications to the current codebase to optimize for the weighted multi-metric evaluation system used in the Kaggle competition. The primary focus is on aligning the loss functions with the evaluation metric weights and implementing efficient preprocessing/inference pipelines.

---

## 📊 Evaluation Metrics & Weights (Target Optimization)

| Metric | Weight | Measurement | Implementation |
|--------|--------|-------------|-----------------|
| Edge Similarity | 40% | Multi-scale Canny edge F1 score | Primary loss function |
| Line Straightness | 22% | Hough line angle distribution match | Secondary loss function |
| Gradient Orientation | 18% | Gradient direction histogram similarity | Tertiary loss function |
| SSIM | 15% | Structural similarity index | Quaternary loss function |
| Pixel Accuracy | 5% | Mean absolute pixel difference | Fine-tuning loss |

**Hard Fail Conditions:**
- Maximum regional difference exceeds threshold → Score 0.0
- Edge similarity below minimum threshold → Score 0.0

---

## 🎯 Critical Implementation Objectives

### 1. **Preprocessing Pipeline (Standardization)**
- [ ] **Downsampling**: Resize all training/validation images to **256×256**
  - Reduces computation overhead by 85%
  - Maintains edge structure quality
  - Use bicubic interpolation for downsampling
  
- [ ] **Parallel Processing**:
  - Implement `multiprocessing.Pool` or `torch.utils.data.DataLoader` with `num_workers > 0`
  - Process dataset in parallel batches
  - Cache preprocessed images to disk
  - Target: <2 seconds per batch on CPU-bound preprocessing

- [ ] **Normalization**:
  - Standardize intensity values (0-1 range)
  - Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for edge preservation

---

### 2. **Loss Function Implementation (Weighted Composite)**

Create a new file: `losses/weighted_loss.py`

```python
# Pseudo-structure for weighted loss
class WeightedMultiMetricLoss(nn.Module):
    def __init__(self):
        self.weights = {
            'edge_similarity': 0.40,
            'line_straightness': 0.22,
            'gradient_orientation': 0.18,
            'ssim': 0.15,
            'pixel_accuracy': 0.05
        }
    
    def forward(self, pred, target):
        loss = (
            0.40 * edge_similarity_loss(pred, target) +
            0.22 * line_straightness_loss(pred, target) +
            0.18 * gradient_orientation_loss(pred, target) +
            0.15 * ssim_loss(pred, target) +
            0.05 * l1_loss(pred, target)
        )
        return loss
```

#### Loss Function Details:

**a) Edge Similarity Loss (40%)**
- [ ] Implement multi-scale Canny edge detection
- [ ] Compute F1 score between predicted and target edges
- [ ] Use scales: [0.5, 1.0, 2.0] (corresponding to edge thresholds)
- [ ] Formula: `F1 = 2 * (precision * recall) / (precision + recall)`
- [ ] Loss = `1 - F1_score`

**b) Line Straightness Loss (22%)**
- [ ] Apply Hough line transform to predicted image
- [ ] Compute angle distribution of detected lines
- [ ] Compare angle histogram against ground truth
- [ ] Use histogram intersection or Wasserstein distance
- [ ] Loss = `1 - angle_distribution_match`

**c) Gradient Orientation Loss (18%)**
- [ ] Compute gradient using Sobel filters
- [ ] Extract angle maps (atan2(Gy, Gx))
- [ ] Build orientation histograms (36 bins for 10° resolution)
- [ ] Compare using cosine similarity
- [ ] Loss = `1 - cosine_similarity(hist_pred, hist_target)`

**d) SSIM Loss (15%)**
- [ ] Use `skimage.metrics.structural_similarity` or `torchvision.metrics.StructuralSimilarityIndexMeasure`
- [ ] Compute SSIM with `data_range=1.0`
- [ ] Loss = `1 - SSIM_score`

**e) Pixel Accuracy Loss (5%)**
- [ ] Simple L1/MAE: `mean(|pred - target|)`
- [ ] Serves as fine-tuning regularizer

---

### 3. **FEGAN Model Modifications**

**File: `models/fegan.py`**

- [ ] **Input/Output channels**: Ensure model handles 256×256 input
- [ ] **Loss weighting**: Integrate weighted composite loss into training loop
- [ ] **Validation metrics**: Track all 5 metrics separately during training
- [ ] **Early stopping**: Monitor weighted composite loss, NOT individual metrics

Example training loop modification:
```python
# Instead of: loss = criterion(pred, target)
# Use:
loss = weighted_loss(pred, target)  # Returns: 0.40*edge + 0.22*line + ...

# Track metrics separately for logging
edge_sim = compute_edge_similarity(pred, target)
line_straight = compute_line_straightness(pred, target)
grad_orient = compute_gradient_orientation(pred, target)
ssim_score = compute_ssim(pred, target)
pixel_acc = compute_l1(pred, target)

wandb.log({
    'loss_weighted': loss.item(),
    'metric_edge_similarity': edge_sim,
    'metric_line_straightness': line_straight,
    'metric_gradient_orientation': grad_orient,
    'metric_ssim': ssim_score,
    'metric_pixel_accuracy': pixel_acc,
})
```

---

### 4. **AI Upsampling for Inference**

**File: `inference/upsampler.py`**

After inference at 256×256, upsample to match ground truth image sizes.

- [ ] **Method 1: Real-ESRGAN** (Recommended)
  - State-of-the-art super-resolution
  - Preserves edge structure
  - Maintains gradient information
  ```bash
  pip install realesrgan
  ```
  
- [ ] **Method 2: Bicubic Interpolation** (Fallback)
  - Fast, deterministic
  - Less artifacts than bilinear
  
- [ ] **Method 3: BSRGAN** (Alternative)
  - Handles degradation well
  - Good for edge-heavy images

**Inference Pipeline:**
```
Input Image (Original Size)
    ↓
Downsampling to 256×256
    ↓
FEGAN Edge Correction (256×256)
    ↓
Real-ESRGAN Upsampling (Original Size)
    ↓
Output (Original Size, Edge Corrected)
```

**Implementation:**
```python
# Pseudo-code
def inference_with_upsampling(image_path, original_size):
    # 1. Load original image
    img_original = cv2.imread(image_path)
    h_orig, w_orig = img_original.shape[:2]
    
    # 2. Downsampling
    img_256 = cv2.resize(img_original, (256, 256), interpolation=cv2.INTER_CUBIC)
    
    # 3. Edge correction via FEGAN
    corrected_256 = fegan_model(img_256)
    
    # 4. AI Upsampling (Real-ESRGAN)
    upsampler = RealESRGANer(scale=4, ...)  # Adjust scale as needed
    corrected_original = upsampler.enhance(corrected_256)[0]
    
    # 5. Resize to exact original dimensions (if needed)
    corrected_original = cv2.resize(corrected_original, (w_orig, h_orig))
    
    return corrected_original
```

---

## 📋 Implementation Checklist

### Phase 1: Data Preprocessing (Week 1)
- [ ] Create `preprocessing/data_loader.py`
  - [ ] Implement `PreprocessorPool` with `multiprocessing.Pool`
  - [ ] Add 256×256 downsampling with bicubic interpolation
  - [ ] Implement CLAHE normalization
  - [ ] Cache to disk (`.npy` format for speed)
  - [ ] Benchmark: Target <100ms per image

- [ ] Modify dataset class
  - [ ] Update `__getitem__` to load 256×256 images
  - [ ] Add multi-worker DataLoader support
  - [ ] Verify no data leakage between train/val/test

### Phase 2: Loss Functions (Week 1-2)
- [ ] Create `losses/` directory
  - [ ] `losses/edge_similarity.py` - Multi-scale Canny F1
  - [ ] `losses/line_straightness.py` - Hough angle matching
  - [ ] `losses/gradient_orientation.py` - Gradient histogram
  - [ ] `losses/weighted_loss.py` - Composite loss
  
- [ ] Unit tests for each loss function
  - [ ] Verify outputs are in [0, 1] range
  - [ ] Test gradient flow (backpropagation)
  - [ ] Benchmark computation time

### Phase 3: Model Integration (Week 2)
- [ ] Modify FEGAN training loop
  - [ ] Replace old loss with `WeightedMultiMetricLoss`
  - [ ] Update optimizer (recommend Adam with lr=1e-3)
  - [ ] Add metric tracking for all 5 metrics
  - [ ] Implement validation loop with metric computation
  
- [ ] Hyperparameter tuning
  - [ ] Loss weight sensitivity analysis
  - [ ] Learning rate scheduling (ReduceLROnPlateau)
  - [ ] Batch size optimization for 256×256

### Phase 4: Inference & Upsampling (Week 2-3)
- [ ] Install Real-ESRGAN
  ```bash
  pip install realesrgan
  ```
  
- [ ] Create `inference/inference_pipeline.py`
  - [ ] Load trained FEGAN model
  - [ ] Initialize Real-ESRGAN upsampler
  - [ ] Implement inference function with error handling
  - [ ] Add quality checks (hard fail detection)
  
- [ ] Create submission script
  - [ ] Batch process test set
  - [ ] Generate predictions at original sizes
  - [ ] Compute evaluation metrics (for validation)
  - [ ] Output CSV with per-image scores

### Phase 5: Validation & Testing (Week 3)
- [ ] Hard fail detection
  - [ ] Monitor maximum regional difference
  - [ ] Track minimum edge similarity
  - [ ] Flag images that would score 0.0
  
- [ ] Benchmark suite
  - [ ] Measure inference time per image (target: <5s with upsampling)
  - [ ] Memory usage profiling
  - [ ] Compare MAE against baseline
  
- [ ] Cross-validation
  - [ ] 5-fold CV on training set
  - [ ] Monitor metric stability

---

## 📁 Directory Structure

```
project_root/
├── README.md (this file)
├── designDoc.md (NEW - detailed technical design)
├── config/
│   └── config.yaml (NEW - hyperparameters, paths, loss weights)
├── data/
│   ├── raw/
│   ├── processed_256/          (NEW - cached 256×256 images)
│   ├── train.csv
│   └── test.csv
├── preprocessing/
│   ├── __init__.py
│   ├── data_loader.py          (MODIFIED - parallel processing)
│   └── preprocessor.py         (NEW - downsampling, CLAHE)
├── losses/
│   ├── __init__.py             (NEW directory)
│   ├── edge_similarity.py       (NEW)
│   ├── line_straightness.py     (NEW)
│   ├── gradient_orientation.py  (NEW)
│   └── weighted_loss.py         (NEW)
├── models/
│   ├── fegan.py                (MODIFIED - loss integration)
│   └── ...
├── inference/
│   ├── __init__.py             (NEW directory)
│   ├── upsampler.py            (NEW - Real-ESRGAN wrapper)
│   └── inference_pipeline.py   (NEW)
├── train.py                    (MODIFIED - new loss functions)
├── validate.py                 (NEW - metric computation)
└── submit.py                   (NEW - test set inference)
```

---

## 🚀 Quick Start Commands

```bash
# 1. Setup environment
pip install -r requirements.txt
pip install realesrgan  # For upsampling
pip install opencv-python scikit-image scipy  # For loss functions

# 2. Preprocess dataset (parallel)
python preprocessing/preprocessor.py --input-dir data/raw --output-dir data/processed_256 --workers 8

# 3. Train with new loss functions
python train.py --config config/config.yaml --experiment-name fegan_weighted_loss

# 4. Validate and compute metrics
python validate.py --checkpoint models/best_model.pt --data-dir data/processed_256

# 5. Generate test predictions with upsampling
python submit.py --checkpoint models/best_model.pt --test-dir data/test --output submission.csv

# 6. Analyze hard fail conditions
python utils/hard_fail_analysis.py --predictions predictions/ --thresholds config/thresholds.yaml
```

---

## 🔍 Monitoring & Evaluation

### Key Metrics to Track During Training

```
Epoch 1:
  Loss (Weighted): 0.523
    ├─ Edge Similarity: 0.65 (40%)
    ├─ Line Straightness: 0.72 (22%)
    ├─ Gradient Orientation: 0.58 (18%)
    ├─ SSIM: 0.81 (15%)
    └─ Pixel Accuracy (L1): 0.12 (5%)
  
  Val MAE: 0.18 (against reference 1.0 scores)
```

### Submission Evaluation

The Kaggle system will compute:
```
Submission MAE = mean(|your_score - 1.0|) across all test images

Where your_score per image = 
  0.40 * edge_sim_score +
  0.22 * line_straightness_score +
  0.18 * gradient_orientation_score +
  0.15 * ssim_score +
  0.05 * pixel_accuracy_score

Each component score ∈ [0, 1]
Your final MAE target: < 0.10 (excellent), < 0.15 (competitive)
```

---

## ⚠️ Critical Considerations

### Hard Fail Detection
Before submission, verify NO images trigger hard fails:
```python
# Flag these conditions:
if max_regional_diff > THRESHOLD:  # 0.5 recommended
    score = 0.0  # AUTO FAIL
    
if edge_similarity < MIN_EDGE_SIM:  # 0.3 recommended
    score = 0.0  # AUTO FAIL
```

### Upsampling Quality Trade-offs
- **Real-ESRGAN 4x scale**: Best quality, 3-5s per image
- **Bicubic interpolation**: Baseline, <100ms per image
- **Recommendation**: Use Real-ESRGAN for final submission, bicubic for rapid iteration

### Validation Strategy
- Reserve 20% of training data for validation
- Compute all 5 metrics + composite loss
- Monitor for overfitting to individual metrics (e.g., edge similarity at cost of SSIM)
- Track MAE progression

---

## 📚 References

- Canny Edge Detection: OpenCV docs, `cv2.Canny()`
- Hough Transform: OpenCV, `cv2.HoughLines()`
- SSIM: `skimage.metrics.structural_similarity()`
- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
- Gradient Orientation: NumPy `arctan2()`, histogram binning

---

## 🎓 Success Criteria

✅ **Preprocessing**
- All images resized to 256×256 without quality loss
- Parallel processing < 5 minutes for full dataset

✅ **Loss Functions**
- Each metric computable with <100ms inference time
- Gradients flow properly (no NaN/Inf in backprop)
- Validation metrics align with Kaggle evaluation

✅ **Model Training**
- Training converges with new weighted loss
- Validation MAE < 0.15 on holdout set
- No hard fail conditions triggered

✅ **Inference**
- Output images match ground truth dimensions
- Edge structure preserved after upsampling
- Submission CSV has correct format

---

## 📞 Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM with 256×256 batch size 32 | Reduce batch size to 16 or 8 |
| Loss becomes NaN | Check for division by zero in metric computation |
| Upsampling artifacts | Try BSRGAN or increase interpolation quality |
| Slow preprocessing | Increase `num_workers` or use SSD caching |
| Hard fails in submission | Adjust thresholds or retrain with penalty term |

---

**Last Updated**: February 2026  
**Status**: Ready for Implementation  
**Estimated Timeline**: 3 weeks (all phases)

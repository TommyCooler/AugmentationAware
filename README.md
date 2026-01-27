# Augmentation-Aware Anomaly Detection System

Hệ thống phát hiện dị thường (Anomaly Detection) hai giai đoạn sử dụng Contrastive Learning và Reconstruction-based Detection.

---

Checkpoints: https://drive.google.com/drive/folders/1ss7Ozt4PV3sXGDkFHZLMn_93be-82GWV?usp=drive_link


## 📋 Mục Lục

1. [Tổng Quan Kiến Trúc](#tổng-quan-kiến-trúc)
2. [Phase 1: Contrastive Learning Training](#phase-1-contrastive-learning-training)
3. [Phase 2: Supervised Reconstruction Training](#phase-2-supervised-reconstruction-training)
4. [Inference: Anomaly Detection](#inference-anomaly-detection)
5. [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
6. [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)

---

## 🏗️ Tổng Quan Kiến Trúc

Hệ thống được chia thành 2 phase training:

### Phase 1: Contrastive Learning
- **Mục đích**: Huấn luyện module **Augmentation** và **Encoder** để học biểu diễn (representations) tốt từ dữ liệu normal
- **Phương pháp**: Self-supervised contrastive learning với NTXent loss
- **Đầu vào**: Chỉ dùng **train data** (không có labels)
- **Đầu ra**: Checkpoint chứa Augmentation module đã được huấn luyện

### Phase 2: Supervised Reconstruction
- **Mục đích**: Huấn luyện module **AGF-TCN** để reconstruct augmented data
- **Phương pháp**: Supervised reconstruction với MSE loss
- **Đầu vào**: 
  - Augmentation module từ Phase 1 (đã được **freeze** - không train)
  - Train data (để huấn luyện AGF-TCN)
- **Đầu ra**: Checkpoint chứa cả Augmentation và AGF-TCN

### Inference
- **Mục đích**: Phát hiện dị thường trên test data
- **Cơ chế**: Tính reconstruction error → Anomaly score → Threshold → Prediction
- **Đầu vào**: Test data + labels (để đánh giá)
- **Đầu ra**: Anomaly scores, predictions, metrics, và visualization

---

## 🔷 Phase 1: Contrastive Learning Training

### Mục Tiêu
Huấn luyện **Augmentation module** và **Encoder** để học được biểu diễn tốt từ dữ liệu normal mà không cần labels.

### Modules Được Train

#### 1. **Augmentation Module** (`modules/augmentation.py`)
- **Kiến trúc**: 
  - Transformer-based augmentation với learnable transformations
  - Gumbel-Softmax sampling để tạo các augmentation strategies khác nhau
- **Input**: `(batch, n_channels, window_size)`
- **Output**: `(batch, n_channels, window_size)` - augmented version của input
- **Tham số trainable**: ✅ Có (được train trong Phase 1)

#### 2. **Encoder** (`phase1/encoder.py`)
- **Kiến trúc**: Có 2 loại
  - **MLPEncoder**: Multi-layer perceptron với projection head
  - **CNNEncoder**: Convolutional encoder với projection head
- **Input**: `(batch, n_channels, window_size)`
- **Output**: `(batch, projection_dim)` - embedding vector
- **Tham số trainable**: ✅ Có (được train trong Phase 1)

### Quy Trình Training

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1 Training Flow                                        │
└─────────────────────────────────────────────────────────────┘

1. Data Loading
   ├─ Load train data từ nhiều datasets (ucr, smd, etc.)
   ├─ Chia thành sliding windows (window_size, stride)
   └─ Tạo dataloader với batch_size

2. Forward Pass (mỗi batch)
   ├─ Input: batch_windows (batch, channels, window_size)
   ├─ Augmentation: augmented = augmentation(batch_windows)
   ├─ Encoding: 
   │   ├─ z_original = encoder(batch_windows)
   │   └─ z_augmented = encoder(augmented)
   └─ Loss: NTXentLoss(z_original, z_augmented)

3. Backward Pass
   ├─ Tính gradient cho cả encoder và augmentation
   ├─ Gradient clipping (optional)
   └─ Update weights

4. Checkpoint Saving
   ├─ Lưu encoder_state_dict
   ├─ Lưu augmentation_state_dict
   ├─ Lưu config (n_channels, window_size, etc.)
   └─ Lưu optimizer và scheduler state
```

### Chi Tiết Implementation

#### Loss Function: NTXent Loss
```python
# Tạo positive pairs
embeddings = [z_original, z_augmented]  # (2*batch, proj_dim)
labels = [0, 1, 2, ..., 0, 1, 2, ...]  # Mỗi cặp (i, i+batch) là positive pair

# NTXent loss: Pull positive pairs together, push negatives apart
loss = NTXentLoss(temperature=0.07)
```

#### Training Script: `phase1/train_phase1.py`
```bash
python phase1/train_phase1.py
```

**Cấu hình chính**:
- `window_size`: Kích thước window (ví dụ: 16)
- `stride`: Bước nhảy giữa các windows (ví dụ: 1)
- `batch_size`: Kích thước batch (ví dụ: 64)
- `num_epochs`: Số epoch (ví dụ: 1000)
- `encoder_type`: 'mlp' hoặc 'cnn'
- `projection_dim`: Chiều của embedding (ví dụ: 256)
- `transformer_d_model`: Hidden dimension của transformer (ví dụ: 128)
- `transformer_nhead`: Số attention heads (ví dụ: 2)

**Output**: 
- Checkpoint tại `phase1/checkpoints/best_model.pth`
- Chứa augmentation module đã được train (dùng cho Phase 2)

---

## 🔷 Phase 2: Supervised Reconstruction Training

### Mục Tiêu
Huấn luyện **AGF-TCN** để reconstruct augmented data. Module này sẽ được dùng để tính reconstruction error trong inference.

### Modules Được Train

#### 1. **Augmentation Module** (từ Phase 1)
- **Trạng thái**: ❌ **FROZEN** (không train)
  - Tất cả parameters: `requires_grad = False`
  - Đặt về `eval()` mode (disable dropout, BN uses running stats)
- **Vai trò**: Chỉ forward pass để tạo augmented data

#### 2. **AGF-TCN** (`phase2/agf_tcn.py`)
- **Kiến trúc**: 
  - Temporal Convolutional Network (TCN) với Adaptive Graph Fusion
  - Multi-scale feature extraction và fusion
  - Reconstruction head
- **Input**: `(batch, n_channels, window_size)` - augmented data
- **Output**: `(batch, n_channels, window_size)` - reconstructed data
- **Tham số trainable**: ✅ Có (chỉ train module này)

### Quy Trình Training

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 2 Training Flow                                        │
└─────────────────────────────────────────────────────────────┘

1. Load Phase 1 Checkpoint
   ├─ Load augmentation module (FROZEN)
   └─ Verify augmentation is in eval mode

2. Data Loading
   ├─ Load train data (chỉ dùng train, không dùng test)
   ├─ Chia thành sliding windows
   └─ Tạo train dataloader

3. Forward Pass (mỗi batch)
   ├─ Input: batch_data (batch, channels, window_size)
   ├─ Augmentation (FROZEN):
   │   └─ augmented = augmentation(batch_data)  # no_grad()
   ├─ Reconstruction:
   │   └─ reconstructed = agf_tcn(augmented)
   └─ Loss: MSE(reconstructed, augmented)

4. Backward Pass
   ├─ Tính gradient CHỈ cho agf_tcn (augmentation không có gradient)
   ├─ Gradient clipping (optional)
   └─ Update AGF-TCN weights

5. Checkpoint Saving
   ├─ Lưu augmentation_state_dict (frozen, từ Phase 1)
   ├─ Lưu agf_tcn_state_dict (trained)
   ├─ Lưu config (n_channels, window_size, agf_tcn config, etc.)
   └─ Lưu metrics (train_loss, test_loss nếu có)
```

### Chi Tiết Implementation

#### Loss Function: MSE Loss
```python
# Augmented data từ frozen augmentation
augmented = augmentation(batch_data)  # (batch, channels, window_size)

# Reconstruction
reconstructed = agf_tcn(augmented)    # (batch, channels, window_size)

# MSE loss
loss = MSE(reconstructed, augmented)
```

#### Freezing Augmentation
```python
# Freeze tất cả parameters
for param in augmentation.parameters():
    param.requires_grad = False

# Set to eval mode (disable dropout, BN uses running stats)
augmentation.eval()

# Verify: trong forward pass, dùng torch.no_grad()
with torch.no_grad():
    augmented = augmentation(batch_data)
```

#### Training Script: `phase2/train_phase2.py`
```bash
python phase2/train_phase2.py
```

**Cấu hình chính**:
- `phase1_checkpoint`: Đường dẫn đến checkpoint Phase 1
- `window_size`: Phải khớp với Phase 1
- `agf_tcn_channels`: Hidden channels của TCN (ví dụ: [64, 64])
- `dropout`: Dropout rate (ví dụ: 0.1)
- `activation`: Activation function (ví dụ: 'gelu')
- `fuse_type`: Loại fusion (ví dụ: 5 - TripConFusion)
- `batch_size`: Kích thước batch (ví dụ: 64)
- `num_epochs`: Số epoch (ví dụ: 50)

**Output**: 
- Checkpoint tại `phase2/checkpoints/phase2_<dataset>_<subset>_best.pt`
- Chứa cả augmentation (frozen) và agf_tcn (trained)
- Dùng cho inference

---

## 🔷 Inference: Anomaly Detection

### Mục Tiêu
Phát hiện dị thường trên test data sử dụng reconstruction error từ AGF-TCN.

### Quy Trình Inference

```
┌─────────────────────────────────────────────────────────────┐
│ Inference Flow                                               │
└─────────────────────────────────────────────────────────────┘

1. Load Phase 2 Checkpoint
   ├─ Load augmentation module (FROZEN, eval mode)
   ├─ Load agf_tcn (FROZEN, eval mode)
   └─ Load config (window_size, n_channels, etc.)

2. Data Loading
   ├─ Load test data và labels (per time-step)
   ├─ Chia thành sliding windows
   └─ Tạo test dataloader (không shuffle)

3. Forward Pass (mỗi batch)
   ├─ Input: batch_data (batch, channels, window_size)
   ├─ Augmentation (FROZEN):
   │   └─ augmented = augmentation(batch_data)  # no_grad()
   ├─ Reconstruction:
   │   └─ reconstructed = agf_tcn(augmented)  # no_grad()
   └─ Anomaly Score:
       └─ score = MSE(reconstructed, augmented) per time-step
           # Shape: (batch, window_size) - 1 score per time-step

4. Map Window Scores to Time Series
   ├─ Window 0: Map tất cả time-step scores → [0:window_size]
   ├─ Window i>0: Chỉ map score của time-step cuối → [start+window_size-1]
   └─ Forward fill NaN values
   
5. Threshold & Prediction
   └─ Always search for best threshold (maximize F1) → predictions = (anomaly_scores >= best_threshold).astype(int)

6. Evaluation (với Point Adjustment)
   ├─ Apply Point Adjustment: Nếu detect bất kỳ điểm nào trong segment
   │   → Mark toàn bộ segment là detected
   ├─ Compute metrics: F1, Precision, Recall, Accuracy
   └─ Compute segment-level metrics

7. Visualization (optional)
   ├─ Plot original, augmented, reconstructed data
   ├─ Plot anomaly scores với threshold
   ├─ Highlight ground truth và predicted anomalies
   └─ Save to results/visualizations/
```

### Chi Tiết Implementation

#### Anomaly Score Calculation
```python
# Trong mỗi window, tính score cho từng time-step
timestep_losses = torch.mean((reconstructed - augmented) ** 2, dim=1)
# Shape: (batch, window_size) - 1 score per time-step

# Sau khi map về time series:
anomaly_scores  # Shape: (n_time_steps,)
```

#### Mapping Strategy
```python
def map_window_scores_to_timeseries(...):
    # Window 0: Map tất cả
    timeseries_scores[0:window_size] = window_0_scores
    
    # Window i>0: Chỉ map time-step cuối
    last_idx = i * stride + window_size - 1
    timeseries_scores[last_idx] = window_i_scores[-1]
```

#### Threshold Search
```python
# Luôn tìm threshold tốt nhất (maximize F1) với Point Adjustment
# evaluate_with_pa() tự động search và trả về best_threshold
metrics = evaluate_with_pa(
    anomaly_scores=anomaly_scores,
    labels=labels
)
# metrics['best_threshold'] chứa threshold tốt nhất
# metrics['predictions'] chứa predictions tại best threshold
# Point Adjustment luôn được áp dụng tự động
```

#### Point Adjustment
```python
# Nếu có bất kỳ prediction = 1 trong một anomaly segment
# → Set toàn bộ segment = 1
# Giúp đánh giá fair hơn (chỉ cần detect 1 điểm trong segment)
```

#### Inference Script: `phase2/inference.py`
```bash
# Cơ bản
python phase2/inference.py --checkpoint phase2/checkpoints/phase2_ucr_135_best.pt

# Tùy chỉnh
python phase2/inference.py \
    --checkpoint <path> \
    --dataset ucr \
    --subset 135 \
    --no_viz                   # Disable visualization
```

**Arguments**:
- `--checkpoint`: Đường dẫn đến Phase 2 checkpoint (required)
- `--dataset`: Tên dataset (optional, lấy từ checkpoint nếu không có)
- `--subset`: Subset của dataset (optional, lấy từ checkpoint nếu không có)
- `--no_viz`: Tắt visualization (default: có visualization)
- `--batch_size`: Batch size (optional, lấy từ checkpoint nếu không có)
- `--device`: cuda/cpu (optional, auto-detect nếu không có)

**Lưu ý**: 
- Hệ thống luôn tự động search threshold tốt nhất (maximize F1). Không có option để tắt hoặc set threshold cố định.
- Point Adjustment luôn được áp dụng tự động trong quá trình đánh giá.

**Outputs**:
1. **Metrics**: F1, Precision, Recall, Accuracy, Confusion Matrix
2. **Saved results**: `results/inference_<dataset>_<subset>_results.npz`
   - `anomaly_scores`: Anomaly scores cho mỗi time-step
   - `labels`: Ground truth labels
   - `metrics`: Tất cả metrics (dict)
3. **Visualization**: `results/visualizations/viz_<dataset>_<subset>.png`
   - Original data, augmented data, reconstructed data
   - Anomaly scores với threshold line
   - Highlighted anomaly regions (ground truth và predictions)

---

## 📁 Cấu Trúc Dự Án

```
ACIIDS2025/
├── phase1/
│   ├── train_phase1.py          # Phase 1 training script
│   ├── encoder.py                # MLPEncoder, CNNEncoder
│   └── checkpoints/              # Phase 1 checkpoints
│       └── best_model.pth
│
├── phase2/
│   ├── train_phase2.py           # Phase 2 training script
│   ├── inference.py              # Inference script
│   ├── visualize.py              # Visualization utilities
│   ├── agf_tcn.py                # AGF-TCN model
│   ├── basicBlock.py             # TCN basic blocks
│   ├── FusionBlock.py            # Fusion blocks
│   └── checkpoints/              # Phase 2 checkpoints
│       └── phase2_*.pt
│
├── modules/
│   └── augmentation.py           # Augmentation module (Transformer-based)
│
├── data/
│   ├── dataloader.py             # Data loading functions
│   ├── phase1_dataloader.py      # Phase 1 data preparation
│   ├── phase2_dataloader.py      # Phase 2 data preparation
│   └── sliding_window.py         # Sliding window utilities
│
├── utils/
│   └── point_adjustment.py       # Point Adjustment evaluation
│
├── results/
│   ├── inference_*_results.npz   # Inference results
│   └── visualizations/           # Visualization images
│       └── viz_*.png
│
└── README.md                     # File này
```

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Phase 1 Training

```bash
cd <project_root>

# Chỉnh sửa config trong phase1/train_phase1.py
python phase1/train_phase1.py
```

**Config cần chỉnh**:
- `datasets_info`: Danh sách datasets để train
- `window_size`, `stride`
- `batch_size`, `num_epochs`
- `encoder_type`: 'mlp' hoặc 'cnn'
- `transformer_d_model`, `transformer_nhead`

**Output**: `phase1/checkpoints/best_model.pth`

### 2. Phase 2 Training

```bash
# Chỉnh sửa config trong phase2/train_phase2.py
python phase2/train_phase2.py
```

**Config cần chỉnh**:
- `phase1_checkpoint`: Đường dẫn đến Phase 1 checkpoint
- `dataset_name`, `subset`: Dataset để train
- `window_size`: Phải khớp với Phase 1
- `agf_tcn_channels`, `dropout`, `activation`, `fuse_type`

**Output**: `phase2/checkpoints/phase2_<dataset>_<subset>_best.pt`

### 3. Inference

```bash
# Cơ bản (lấy dataset từ checkpoint)
python phase2/inference.py --checkpoint phase2/checkpoints/phase2_ucr_135_best.pt

# Với visualization (default)
python phase2/inference.py --checkpoint <path>

# Không visualization
python phase2/inference.py --checkpoint <path> --no_viz
```

### 4. Xem Kết Quả

- **Metrics**: In ra console
- **Saved results**: `results/inference_<dataset>_<subset>_results.npz`
- **Visualization**: `results/visualizations/viz_<dataset>_<subset>.png`

---

## 📊 Metrics và Evaluation

### Metrics Được Tính

1. **F1-Score**: Harmonic mean của Precision và Recall
2. **Precision**: TP / (TP + FP)
3. **Recall**: TP / (TP + FN)
4. **Accuracy**: (TP + TN) / (TP + TN + FP + FN)

### Point Adjustment (PA)

- **Mục đích**: Đánh giá công bằng hơn cho time-series anomaly detection
- **Cơ chế**: Nếu detect bất kỳ điểm nào trong một anomaly segment → Mark toàn bộ segment là detected
- **Lý do**: Trong thực tế, chỉ cần phát hiện 1 điểm trong segment là đủ để biết segment đó là anomaly

### Threshold Search

- **Mục đích**: Tìm threshold tốt nhất để maximize F1-score
- **Cơ chế**: 
  - Duyệt qua nhiều threshold values
  - Tính F1 với Point Adjustment
  - Chọn threshold có F1 cao nhất

---

## 🔧 Lưu Ý Kỹ Thuật

### 1. Dropout và BatchNorm trong Inference
- Tất cả modules đều được set về `eval()` mode
- Dropout tự động tắt
- BatchNorm dùng running statistics (không update)

### 2. Gradient trong Inference
- Dùng `torch.no_grad()` để tắt gradient computation
- Tiết kiệm memory và tăng tốc

### 3. Augmentation Freezing trong Phase 2
- Tất cả parameters: `requires_grad = False`
- Luôn ở `eval()` mode
- Dùng `torch.no_grad()` khi forward

### 4. Window to Time-Series Mapping
- Window 0: Map tất cả time-steps
- Window i>0: Chỉ map time-step cuối (không có overlap vì chỉ lấy time-step cuối)
- NaN: Forward fill

### 5. Data Normalization
- Dữ liệu đã được normalize trong dataloader
- `normalized=True` khi load data

---

## 📝 Tóm Tắt

### Phase 1: Contrastive Learning
- **Train**: Augmentation + Encoder
- **Loss**: NTXent Loss
- **Input**: Train data (không cần labels)
- **Output**: Augmentation module đã train

### Phase 2: Supervised Reconstruction
- **Train**: AGF-TCN (Augmentation frozen)
- **Loss**: MSE Loss (reconstruction)
- **Input**: Train data + Phase 1 checkpoint
- **Output**: Augmentation (frozen) + AGF-TCN (trained)

### Inference
- **Forward**: Augmentation (frozen) → AGF-TCN (frozen) → Score
- **Evaluation**: Threshold search + Point Adjustment
- **Output**: Scores, predictions, metrics, visualization

---

## 📧 Liên Hệ

Nếu có câu hỏi hoặc vấn đề, vui lòng tạo issue hoặc liên hệ maintainer.

---

**Last Updated**: 2025


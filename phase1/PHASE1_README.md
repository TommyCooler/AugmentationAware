# Phase 1: Contrastive Learning - Chi Tiết Cơ Chế Hoạt Động

## 📋 Tổng Quan

**Phase 1** là giai đoạn **Self-Supervised Contrastive Learning** nhằm học các đặc trưng (features) từ dữ liệu time series thông qua việc tạo và học từ các positive pairs. Phase này train đồng thời 2 modules:
- **Augmentation Module**: Học cách tạo augmentations có ý nghĩa từ masked data
- **Encoder Module**: Học cách encode time series thành embeddings có ý nghĩa

### Mục Đích Chính
- **Feature Learning**: Học representations có ý nghĩa từ unlabeled time series data
- **Augmentation Learning**: Học cách tạo augmentations tự động thay vì dùng augmentations cố định
- **Contrastive Learning**: Sử dụng NTXent loss để pull positive pairs gần nhau, push negatives xa nhau
- **Foundation for Phase 2**: Tạo ra augmentation module đã được pre-train để sử dụng trong Phase 2

---

## 🏗️ Kiến Trúc Tổng Thể

```
Input: batch_windows [B, C, T] (original time series)
    ↓
┌─────────────────────────────────────┐
│   Random Time Masking               │
│   - Mask random time steps          │
│   - Ratio: mask_ratio (default 0.15)│
└─────────────────────────────────────┘
    ↓
masked_windows [B, C, T]
    ↓
┌─────────────────────────────────────┐
│   Augmentation Module                │
│   - 5 augmentation strategies       │
│   - Gumbel-Softmax selection        │
│   - Learnable weights (alpha)       │
└─────────────────────────────────────┘
    ↓
augmented_windows [B, C, T]
    ↓
┌─────────────────────────────────────┐
│   Encoder (MLP)                     │
│   - Flatten input                    │
│   - MLP layers + BatchNorm          │
│   - Projection head                  │
│   - L2 normalization                │
└─────────────────────────────────────┘
    ↓
z_augmented [B, projection_dim]
    ↓
┌─────────────────────────────────────┐
│   Encoder (MLP) - Same instance      │
│   (on original data)                 │
└─────────────────────────────────────┘
    ↓
z_original [B, projection_dim]
    ↓
┌─────────────────────────────────────┐
│   NTXent Loss                        │
│   - Positive pairs: (z_orig, z_aug) │
│   - Negative pairs: all others       │
│   - Temperature: 0.07                │
└─────────────────────────────────────┘
    ↓
Loss (scalar)
```

### Input/Output Format
- **Input Shape**: `(batch_size, num_channels, window_size)`
  - Ví dụ: `(64, 2, 16)` → 64 samples, 2 channels, 16 time steps
- **Augmentation Output**: `(batch_size, num_channels, window_size)` (giống input)
- **Encoder Output**: `(batch_size, projection_dim)`
  - Ví dụ: `(64, 256)` → 64 samples, 256-dimensional embeddings

---

## 🔧 Các Thành Phần Chi Tiết

### 1. **Augmentation Module (Core Component)**

Augmentation module sử dụng **multi-strategy approach** với **Gumbel-Softmax** để tự động chọn và kết hợp 5 augmentation strategies khác nhau.

#### Kiến Trúc Tổng Thể

```
Input [B, C, T]
    │
    ├─ LinearAugmentation ────────┐
    ├─ MLPaugmentation ───────────┤
    ├─ CNN1DAugmentation ─────────┤
    ├─ CNN1DCausalAugmentation ───┤
    └─ TransformerEncoderAug ─────┤
                                   │
                    Stack: [5, B, C, T]
                                   │
                    Gumbel-Softmax Selection
                    (learnable weights α)
                                   │
                    Weighted Sum
                                   ↓
                    Output [B, C, T]
```

#### 5 Augmentation Strategies

##### Strategy 1: LinearAugmentation
```python
# Channel-wise linear transformation
output = weights @ input + bias
# weights: [C, C]
# bias: [C]
```
- **Mô tả**: Linear transformation theo từng channel
- **Ưu điểm**: Đơn giản, nhanh, học được linear relationships
- **Use case**: Capture linear dependencies giữa các channels

##### Strategy 2: MLPaugmentation
```python
# Flatten → MLP → Reshape
x = input.view(B, C*T)  # [B, C*T]
x = MLP(x)              # [B, C*T]
x = x.view(B, C, T)     # [B, C, T]
```
- **Mô tả**: Multi-layer perceptron trên flattened input
- **Ưu điểm**: Học được non-linear transformations
- **Use case**: Capture complex non-linear patterns

##### Strategy 3: CNN1DAugmentation
```python
# Standard 1D Convolution (symmetric padding)
Conv1d(C, C, kernel_size, padding='same')
→ GELU → Dropout
```
- **Mô tả**: 1D convolution với symmetric padding
- **Ưu điểm**: Capture local temporal patterns
- **Use case**: Local feature extraction

##### Strategy 4: CNN1DCausalAugmentation
```python
# Causal 1D Convolution (left padding only)
F.pad(x, (kernel_size-1, 0))  # Left padding
Conv1d(C, C, kernel_size, padding=0)
→ GELU → Dropout
```
- **Mô tả**: Causal convolution (chỉ nhìn vào quá khứ)
- **Ưu điểm**: Phù hợp cho time series, không leak future info
- **Use case**: Temporal dependencies với causal constraint

##### Strategy 5: TransformerEncoderAugmentation
```python
# Transformer Encoder
Input Projection: [B, C, T] → [B, T, d_model]
Positional Encoding (learnable)
Transformer Encoder (num_layers)
Output Projection: [B, T, d_model] → [B, C, T]
```
- **Mô tả**: Transformer encoder với learnable positional encoding
- **Ưu điểm**: Capture long-range dependencies, attention mechanism
- **Use case**: Complex temporal patterns, global context

#### Gumbel-Softmax Selection

Sau khi có 5 outputs, module sử dụng **Gumbel-Softmax** để chọn và kết hợp:

**Training Mode**:
```python
probs = F.gumbel_softmax(alpha, tau=temperature, hard=hard, dim=0)
# probs: [5] - probability distribution over 5 strategies
```

**Inference Mode**:
```python
probs = F.softmax(alpha / temperature, dim=0)
# Soft selection (smooth combination)
```

**Weighted Combination**:
```python
weighted = outputs * probs.view(-1, 1, 1, 1)  # [5, B, C, T]
combined_output = weighted.sum(dim=0)          # [B, C, T]
```

#### Learnable Weights (α)

- **Initialization**: `alpha = ones(5) / 5.0` (uniform distribution)
- **Learning**: Được update qua backpropagation
- **Interpretation**: 
  - α[i] cao → strategy i được ưa chuộng
  - Model tự động học strategy nào tốt nhất cho dataset

#### Temperature Parameter

- **Training**: `tau = temperature` (default 1.0)
  - Cao → smooth distribution (exploration)
  - Thấp → sharp distribution (exploitation)
- **Inference**:**: `tau = temperature` (smooth combination)

---

### 2. **MLPEncoder**

Encoder chuyển đổi time series windows thành embeddings có ý nghĩa.

#### Kiến Trúc

```
Input [B, C, T]
    ↓
Flatten: [B, C*T]
    ↓
┌─────────────────────────────────────┐
│   MLP Layers                         │
│   - Linear → BatchNorm → ReLU       │
│   - Dropout                          │
│   - Repeat for each hidden_dim      │
└─────────────────────────────────────┘
    ↓
Projection Head: Linear(C*T → projection_dim)
    ↓
L2 Normalization
    ↓
Output [B, projection_dim]
```

#### Chi Tiết

**Hidden Layers**:
```python
for hidden_dim in hidden_dims:
    x = Linear(prev_dim, hidden_dim)(x)
    x = BatchNorm1d(hidden_dim)(x)
    x = ReLU(inplace=True)(x)
    x = Dropout(dropout)(x)
```

**Projection Head**:
```python
z = Linear(last_hidden_dim, projection_dim)(x)
z = L2_normalize(z, dim=1)  # Normalize to unit sphere
```

#### Tại Sao L2 Normalization?
- **Contrastive Learning**: Embeddings trên unit sphere dễ so sánh hơn
- **Stability**: Tránh embeddings quá lớn
- **Standard Practice**: Thường dùng trong contrastive learning (SimCLR, MoCo)

---

### 3. **RandomTimeMasking**

Tạo masked versions của input để tăng độ khó cho augmentation.

#### Cơ Chế

```python
# Randomly select time steps to mask
n_mask = int(T * mask_ratio)  # e.g., 15% of time steps
mask_indices = torch.randperm(T)[:n_mask]

# Create mask
time_mask = ones(T)
time_mask[mask_indices] = False  # Mask selected indices

# Apply mask
masked_x = x * time_mask  # Zero out masked positions
```

#### Đặc Điểm
- **Shared Mask**: Cùng một mask cho tất cả channels trong một sample
- **Random**: Mask indices được chọn ngẫu nhiên mỗi lần
- **Training Only**: Không mask trong eval mode (nếu mask_ratio > 0)

#### Tại Sao Masking?
- **Regularization**: Tăng độ khó, tránh overfitting
- **Robustness**: Model học cách xử lý missing data
- **Contrastive Learning**: Tạo variation giữa original và augmented

---

### 4. **NTXent Loss (Normalized Temperature-scaled Cross Entropy)**

Loss function cho contrastive learning.

#### Cơ Chế

**Positive Pairs**:
- `(z_original[i], z_augmented[i])` cho mỗi sample i
- Cùng label: `labels[i] = labels[i+batch_size] = i`

**Negative Pairs**:
- Tất cả các cặp khác trong batch
- `(z_original[i], z_augmented[j])` với i ≠ j

**Loss Calculation**:
```python
# Similarity matrix
sim_matrix = z @ z.T / temperature  # [2B, 2B]

# Positive pairs: (i, i+B) and (i+B, i)
# Negative pairs: all others

# NTXent loss
loss = -log(exp(sim_pos) / (exp(sim_pos) + sum(exp(sim_neg))))
```

#### Temperature Parameter
- **Default**: 0.07
- **Cao**: Smooth distribution, easier learning
- **Thấp**: Sharp distribution, harder learning (better separation)

---

## 🔄 Cơ Chế Hoạt Động Trong Phase 1

### Training Flow Chi Tiết

```
1. Input: batch_windows [B, C, T]
   - Load từ multiple datasets
   - All normal data (no labels needed)
   
2. Random Time Masking
   masked_windows = time_masking(batch_windows)
   - Randomly mask 15% of time steps
   - Same mask for all channels
   
3. Augmentation (TRAINABLE)
   augmented_windows = augmentation(masked_windows)
   - 5 strategies run in parallel
   - Gumbel-Softmax selection
   - Weighted combination
   - Both augmentation and alpha weights are updated
   
4. Encoding - Original
   z_original = encoder(batch_windows)
   - Flatten → MLP → Projection → L2 norm
   - Shape: [B, projection_dim]
   
5. Encoding - Augmented
   z_augmented = encoder(augmented_windows)
   - Same encoder, same process
   - Shape: [B, projection_dim]
   
6. Create Positive Pairs
   embeddings = concat([z_original, z_augmented], dim=0)
   # Shape: [2B, projection_dim]
   
   labels = [0, 1, 2, ..., B-1, 0, 1, 2, ..., B-1]
   # Each (i, i+B) is a positive pair
   
7. NTXent Loss
   loss = NTXentLoss(embeddings, labels)
   - Pull positive pairs together
   - Push negative pairs apart
   
8. Backward Pass
   - Update encoder parameters
   - Update augmentation parameters (all 5 strategies + alpha)
   - AdamW optimizer
```

### Data Flow

**Multi-Dataset Training**:
- Load data từ nhiều datasets (UCR, SMD, ECG, etc.)
- Combine tất cả windows lại
- Shuffle để mix datasets
- Model học features chung cho tất cả datasets

---

## 📊 Chi Tiết Forward Pass

### Ví Dụ Cụ Thể

**Input**: `batch_windows` shape `[64, 2, 16]`
- Batch size: 64
- Channels: 2
- Time steps: 16

#### Step 1: Random Time Masking

```python
mask_ratio = 0.15
n_mask = int(16 * 0.15) = 2  # Mask 2 time steps
mask_indices = [3, 11]  # Randomly selected

masked_windows[64, 2, 16]
# Time steps 3 and 11 are zeroed out
```

#### Step 2: Augmentation

**5 Strategies Run**:

1. **LinearAugmentation**:
   ```
   weights: [2, 2]
   output = weights @ masked_windows + bias
   → [64, 2, 16]
   ```

2. **MLPaugmentation**:
   ```
   Flatten: [64, 32]  # 2*16
   MLP: [64, 32] → [64, 32]
   Reshape: [64, 2, 16]
   ```

3. **CNN1DAugmentation**:
   ```
   Conv1d(2, 2, kernel=3, padding=1)
   → [64, 2, 16]
   ```

4. **CNN1DCausalAugmentation**:
   ```
   Pad left: [64, 2, 18]
   Conv1d(2, 2, kernel=3, padding=0)
   → [64, 2, 16]
   ```

5. **TransformerEncoderAugmentation**:
   ```
   Input proj: [64, 2, 16] → [64, 16, 128]
   Positional encoding
   Transformer encoder (2 layers)
   Output proj: [64, 16, 128] → [64, 2, 16]
   ```

**Gumbel-Softmax Selection**:
```python
alpha = [0.2, 0.15, 0.25, 0.2, 0.2]  # Learned weights
probs = gumbel_softmax(alpha, tau=1.0)
# probs = [0.18, 0.12, 0.32, 0.19, 0.19]  # Example

# Weighted combination
output = 0.18*linear + 0.12*mlp + 0.32*cnn + 0.19*cnn_causal + 0.19*transformer
→ [64, 2, 16]
```

#### Step 3: Encoding

**Original**:
```python
z_original = encoder(batch_windows)
# Flatten: [64, 32]
# MLP: [64, 32] → [64, 256] → [64, 256]
# L2 norm: [64, 256]
```

**Augmented**:
```python
z_augmented = encoder(augmented_windows)
# Same process
# Shape: [64, 256]
```

#### Step 4: NTXent Loss

```python
embeddings = concat([z_original, z_augmented], dim=0)
# Shape: [128, 256]

labels = [0,1,2,...,63, 0,1,2,...,63]
# Positive pairs: (0, 64), (1, 65), ..., (63, 127)

loss = NTXentLoss(embeddings, labels, temperature=0.07)
```

---

## 🎯 Tại Sao Phase 1 Hiệu Quả?

### 1. **Multi-Strategy Augmentation**
- 5 strategies khác nhau capture nhiều loại transformations
- Model tự động học strategy nào tốt nhất
- Robust với nhiều loại time series

### 2. **Contrastive Learning**
- Pull positive pairs gần nhau
- Push negative pairs xa nhau
- Học representations có ý nghĩa mà không cần labels

### 3. **Self-Supervised Learning**
- Không cần labeled data
- Có thể train trên nhiều datasets
- Học features chung (generalizable)

### 4. **Learnable Augmentation**
- Thay vì dùng augmentations cố định, model học cách augment
- Adapt với dataset cụ thể
- Tạo ra augmentations có ý nghĩa hơn

### 5. **Masking Strategy**
- Tăng độ khó cho model
- Học cách xử lý missing data
- Regularization effect

---

## ⚙️ Các Tham Số Quan Trọng

### Augmentation Module

#### `kernel_size`
- **Mô tả**: Kernel size cho CNN augmentations
- **Default**: 3
- **Ảnh hưởng**: 
  - Nhỏ → local patterns
  - Lớn → broader patterns

#### `num_layers`
- **Mô tả**: Số layers cho MLP/Transformer
- **Default**: 2
- **Ảnh hưởng**: 
  - Nhiều layers → deeper, more complex
  - Ít layers → simpler, faster

#### `dropout`
- **Mô tả**: Dropout rate trong augmentation strategies
- **Default**: 0.1
- **Ảnh hưởng**: Regularization

#### `temperature` (Gumbel-Softmax)
- **Mô tả**: Temperature cho Gumbel-Softmax
- **Default**: 1.0
- **Ảnh hưởng**: 
  - Cao → smooth selection (exploration)
  - Thấp → sharp selection (exploitation)

#### `hard` (Gumbel-Softmax)
- **Mô tả**: Hard vs soft Gumbel-Softmax
- **Default**: False
- **Ảnh hưởng**: 
  - False → soft (differentiable)
  - True → hard (discrete, non-differentiable)

#### `transformer_d_model`
- **Mô tả**: Hidden dimension của Transformer
- **Default**: 128
- **Ảnh hưởng**: 
  - Lớn → more capacity, slower
  - Nhỏ → less capacity, faster

#### `transformer_nhead`
- **Mô tả**: Số attention heads
- **Default**: 2
- **Lưu ý**: Phải là số chẵn
- **Ảnh hưởng**: 
  - Nhiều heads → more parallel attention
  - Ít heads → simpler attention

### Encoder Module

#### `hidden_dims`
- **Mô tả**: Số units trong mỗi hidden layer
- **Default**: [256]
- **Ví dụ**: [512, 512, 256] → 3 layers
- **Ảnh hưởng**: 
  - Nhiều layers → deeper network
  - Nhiều units → more capacity

#### `projection_dim`
- **Mô tả**: Dimension của output embedding
- **Default**: 256
- **Ảnh hưởng**: 
  - Lớn → more expressive, but more parameters
  - Nhỏ → less expressive, but fewer parameters

#### `dropout_mlp`
- **Mô tả**: Dropout rate trong encoder
- **Default**: 0.1
- **Ảnh hưởng**: Regularization

### Training

#### `temperature` (NTXent Loss)
- **Mô tả**: Temperature cho contrastive loss
- **Default**: 0.07
- **Ảnh hưởng**: 
  - Cao → easier learning
  - Thấp → harder learning (better separation)

#### `mask_ratio`
- **Mô tả**: Tỷ lệ time steps bị mask
- **Default**: 0.15 (15%)
- **Ảnh hưởng**: 
  - Cao → harder task
  - Thấp → easier task

#### `learning_rate`
- **Mô tả**: Learning rate cho optimizer
- **Default**: 1e-4
- **Ảnh hưởng**: Training speed và stability

#### `weight_decay`
- **Mô tả**: L2 regularization
- **Default**: 1e-6
- **Ảnh hưởng**: Regularization

---

## 📝 Ví Dụ Sử Dụng

### Khởi Tạo Model

```python
from modules.augmentation import Augmentation
from phase1.encoder import MLPEncoder

# Augmentation module
augmentation = Augmentation(
    in_channels=2,
    seq_len=16,
    kernel_size=3,
    num_layers=2,
    dropout=0.1,
    temperature=1.0,
    hard=False,
    transformer_d_model=128,
    transformer_nhead=2,
)

# Encoder module
encoder = MLPEncoder(
    input_channels=2,
    window_size=16,
    hidden_dims=[256],
    projection_dim=256,
    dropout=0.1,
)
```

### Training Loop

```python
from phase1.train_phase1 import Phase1Trainer

trainer = Phase1Trainer(
    encoder=encoder,
    augmentation=augmentation,
    temperature=0.07,
    learning_rate=1e-4,
    weight_decay=1e-6,
    mask_ratio=0.15,
)

# Train
trainer.train(
    dataloader=dataloader,
    num_epochs=1000,
    save_dir="checkpoints",
    checkpoint_filename="best_model.pth"
)
```

### Forward Pass Example

```python
# Training mode
encoder.train()
augmentation.train()

batch_windows = torch.randn(64, 2, 16)

# Masking
masked = time_masking(batch_windows)

# Augmentation
augmented = augmentation(masked)

# Encoding
z_original = encoder(batch_windows)      # [64, 256]
z_augmented = encoder(augmented)         # [64, 256]

# Loss
embeddings = torch.cat([z_original, z_augmented], dim=0)  # [128, 256]
labels = torch.arange(64).repeat(2)  # [0,1,...,63,0,1,...,63]
loss = ntxent_loss(embeddings, labels)
```

---

## 🔍 So Sánh Với Các Phương Pháp Khác

| Feature | Fixed Augmentation | Learnable Augmentation (Phase 1) |
|---------|-------------------|----------------------------------|
| **Augmentation** | Pre-defined (rotation, noise, etc.) | Learned (5 strategies + selection) |
| **Adaptability** | Fixed for all datasets | Adapts to dataset |
| **Selection** | Manual/random | Automatic (Gumbel-Softmax) |
| **Complexity** | Simple | More complex but more powerful |
| **Training** | Only encoder | Encoder + Augmentation |

---

## 🐛 Debug Tips

### 1. Kiểm Tra Augmentation Weights

```python
# Check learned alpha weights
alpha = augmentation.alpha
probs = F.softmax(alpha, dim=0)
print(f"Augmentation probabilities: {probs}")
# Should see which strategy is preferred
```

### 2. Kiểm Tra Embeddings

```python
z_original = encoder(batch_windows)
z_augmented = encoder(augmented_windows)

# Check if normalized
print(f"z_original norm: {z_original.norm(dim=1).mean()}")
print(f"z_augmented norm: {z_augmented.norm(dim=1).mean()}")
# Should be ~1.0 (L2 normalized)
```

### 3. Kiểm Tra Similarity

```python
# Positive pairs should be similar
pos_sim = (z_original * z_augmented).sum(dim=1).mean()
print(f"Positive pair similarity: {pos_sim:.4f}")
# Should be high (close to 1.0)
```

### 4. Kiểm Tra Loss

```python
# Loss should decrease over time
# If stuck, check:
# - Learning rate too high/low
# - Temperature too high/low
# - Mask ratio too high
```

---

## 📚 Tài Liệu Tham Khảo

- **Contrastive Learning**: SimCLR (Chen et al., 2020)
- **NTXent Loss**: SimCLR paper
- **Gumbel-Softmax**: Jang et al., 2016
- **Self-Supervised Learning**: Survey papers
- **Transformer**: Vaswani et al., 2017

---

## ⚠️ Lưu Ý Quan Trọng

1. **Multi-Dataset Training**: Phase 1 train trên nhiều datasets để học features chung
2. **No Labels Needed**: Tất cả data là normal, không cần anomaly labels
3. **Augmentation + Encoder**: Cả 2 modules được train đồng thời
4. **Checkpoint Saving**: Lưu cả encoder và augmentation để dùng trong Phase 2
5. **Normalization**: Dữ liệu đã được normalize về [0, 1] trước khi vào model
6. **Temperature**: Có 2 temperatures khác nhau:
   - Gumbel-Softmax temperature (augmentation selection)
   - NTXent temperature (contrastive loss)

---

## 🔗 Kết Nối Với Phase 2

Phase 1 tạo ra:
- **Augmentation Module**: Được freeze và sử dụng trong Phase 2
- **Learned Representations**: Encoder có thể được sử dụng cho downstream tasks
- **Augmentation Strategy**: Model đã học cách augment tốt nhất cho dataset

Trong Phase 2:
- Augmentation module được **FROZEN** (không update)
- Chỉ AGF-TCN được train
- Augmentation tạo augmented_data từ masked_data
- AGF-TCN học reconstruct augmented_data

---

## 🎓 Kết Luận

Phase 1 là giai đoạn **Self-Supervised Contrastive Learning** quan trọng:
- Học **features có ý nghĩa** từ unlabeled data
- Học **augmentation strategies** tự động
- Tạo **foundation** cho Phase 2
- **Generalizable** across multiple datasets

Module này kết hợp:
- **Multi-strategy augmentation** (5 strategies)
- **Learnable selection** (Gumbel-Softmax)
- **Contrastive learning** (NTXent loss)
- **Self-supervised learning** (no labels needed)

Tất cả nhằm mục đích học representations tốt để sử dụng trong Phase 2 cho anomaly detection.


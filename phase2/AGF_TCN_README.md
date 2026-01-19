# AGF-TCN Module: Chi Tiết Cơ Chế Hoạt Động

## 📋 Tổng Quan

**AGF-TCN (Attention-Guided Fusion Temporal Convolutional Network)** là một mô-đun mạng nơ-ron được thiết kế đặc biệt cho nhiệm vụ **reconstruction** trong Phase 2 của hệ thống Anomaly Detection. Module này học cách reconstruct lại dữ liệu đã được augment từ Augmentation module (đã được pre-train trong Phase 1).

### Mục Đích Chính
- **Reconstruction Task**: Học cách reconstruct lại `augmented_data` từ chính nó (autoencoder style)
- **Anomaly Detection**: Trong inference, reconstruction error cao → có thể là anomaly
- **Feature Learning**: Học các đặc trưng quan trọng từ augmented data để có thể reconstruct chính xác

---

## 🏗️ Kiến Trúc Tổng Thể

```
Input: augmented_data [B, C, T]
    ↓
┌─────────────────────────────────────┐
│   Encoder Network (TCN Layers)      │
│   - Multiple AttentionFusionBlocks   │
│   - Progressive dilation (2^i)      │
│   - Multi-scale feature extraction  │
└─────────────────────────────────────┘
    ↓
Encoded Features [B, num_channels[-1], T]
    ↓
┌─────────────────────────────────────┐
│   Decoder (CustomLinear)             │
│   - Channel-wise linear projection  │
│   - Restore original dimensions     │
└─────────────────────────────────────┘
    ↓
Output: reconstructed [B, C, T]
```

### Input/Output Format
- **Input Shape**: `(batch_size, num_channels, window_size)`
  - Ví dụ: `(64, 2, 16)` → 64 samples, 2 channels, 16 time steps
- **Output Shape**: `(batch_size, num_channels, window_size)` (giống input)
  - Ví dụ: `(64, 2, 16)`

---

## 🔧 Các Thành Phần Chi Tiết

### 1. **Agf_TCN (Main Module)**

#### Khởi Tạo
```python
Agf_TCN(
    num_inputs=2,              # Số channels đầu vào
    num_channels=[256],        # Số channels ở mỗi TCN layer
    dropout=0.1,              # Dropout rate
    activation="gelu",        # Activation function
    fuse_type=2,              # Loại fusion (1-5)
    window_size=16            # Độ dài time series
)
```

#### Cấu Trúc
1. **Encoder Network**: Stack các `AttentionFusionBlock` với dilation tăng dần
2. **Decoder**: `CustomLinear` để project về kích thước ban đầu

#### Forward Pass
```python
def forward(self, x):
    x = self.network(x)      # Encoder: extract features
    x = self.decoder(x)      # Decoder: reconstruct
    return x
```

---

### 2. **AttentionFusionBlock (Core Component)**

Đây là thành phần quan trọng nhất, thực hiện multi-scale feature extraction với 3 branches song song.

#### Kiến Trúc 3 Branches

```
Input [B, C_in, T]
    │
    ├─ Branch 1 (Kernel=3) ──┐
    ├─ Branch 2 (Kernel=5) ──┼─→ Fusion ─→ Output [B, C_out, T]
    └─ Branch 3 (Kernel=7) ──┘
```

#### Chi Tiết Mỗi Branch

**Branch 1 (Kernel Size = 3)**:
```
Input [B, C_in, T]
    ↓
Conv1d(kernel=3, dilation=d) + Chomp1d + Dropout
    ↓
Activation (GELU/ReLU/etc.)
    ↓
SEBlock (Squeeze-and-Excitation)
    ↓
Conv1d(kernel=3, dilation=d) + Chomp1d + Dropout
    ↓
Activation + SEBlock
    ↓
Output [B, C_out, T]
```

**Branch 2 (Kernel Size = 5)**: Tương tự nhưng kernel size = 5
**Branch 3 (Kernel Size = 7)**: Tương tự nhưng kernel size = 7

#### Tại Sao 3 Branches?
- **Multi-scale Feature Extraction**: Mỗi branch capture patterns ở scale khác nhau
  - Kernel 3: Local patterns (short-term dependencies)
  - Kernel 5: Medium-range patterns
  - Kernel 7: Long-range patterns
- **Robustness**: Kết hợp nhiều receptive fields giúp model robust hơn

#### Dilation Mechanism
- Dilation tăng theo cấp số nhân: `dilation = 2^i` (i là layer index)
- Layer 0: dilation = 1
- Layer 1: dilation = 2
- Layer 2: dilation = 4
- ...
- **Lợi ích**: Capture dependencies ở nhiều khoảng cách khác nhau mà không tăng số parameters

#### Chomp1d (Causal Padding)
```python
class Chomp1d(nn.Module):
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
```
- **Mục đích**: Loại bỏ padding ở cuối để giữ nguyên sequence length
- **Causal Convolution**: Đảm bảo không leak thông tin từ tương lai

---

### 3. **SEBlock (Squeeze-and-Excitation)**

#### Cơ Chế
```
Input [B, C, T]
    ↓
AdaptiveAvgPool1d(1)  → [B, C, 1]  (Global average pooling)
    ↓
Conv1d(C → C//reduction) + Activation
    ↓
Conv1d(C//reduction → C) + Sigmoid
    ↓
Scale [B, C, 1]
    ↓
Element-wise multiply: Input × Scale
    ↓
Output [B, C, T]
```

#### Mục Đích
- **Channel Attention**: Tự động học channel nào quan trọng
- **Feature Recalibration**: Tăng weight cho channels quan trọng, giảm cho channels ít quan trọng
- **Improve Representation**: Giúp model tập trung vào features quan trọng

---

### 4. **Fusion Mechanisms**

Sau khi có 3 outputs từ 3 branches, chúng được fusion lại bằng một trong các cách:

#### Fuse Type 1: AddFusion
```python
output = x1 + x2 + x3
```
- **Đơn giản nhất**: Cộng element-wise
- **Ưu điểm**: Nhanh, không có parameters

#### Fuse Type 2: AvgFusion (Default)
```python
output = (x1 + x2 + x3) / 3
```
- **Trung bình**: Chia đều cho 3 branches
- **Ưu điểm**: Ổn định, tránh giá trị quá lớn

#### Fuse Type 3: MulFusion
```python
output = x1 * x2 * x3
```
- **Nhân**: Element-wise multiplication
- **Ưu điểm**: Tăng cường features được đồng ý bởi cả 3 branches

#### Fuse Type 4: ConcatFusion
```python
x = concat([x1, x2, x3], dim=1)  # [B, 3*C, T]
output = Conv1d(3*C → C)(x)       # Project về C channels
```
- **Concatenation**: Nối channels, sau đó project về C channels
- **Ưu điểm**: Giữ được nhiều thông tin nhất

#### Fuse Type 5: TripConFusion
```python
h1 = Conv1d(C → C)(x1)
h2 = Conv1d(C → C)(x2)
h3 = Conv1d(C → C)(x3)
output = h1 * h2 * h3
```
- **Projected Multiplication**: Project mỗi branch trước, rồi nhân
- **Ưu điểm**: Kết hợp ưu điểm của projection và multiplication

---

### 5. **CustomLinear (Decoder)**

#### Mục Đích
Project từ encoded features về kích thước ban đầu (reconstruction).

#### Cơ Chế

**Trường hợp 1: Sequence length không đổi**
```python
# Input: [B, C_encoded, T]
# Output: [B, C_original, T]

output = weights1 @ input + bias1
# weights1: [C_original, C_encoded]
# bias1: [C_original, T]
```

**Trường hợp 2: Sequence length thay đổi**
```python
# Input: [B, C_encoded, T_in]
# Output: [B, C_original, T_out]

x = weights1 @ input + bias1      # [B, C_original, T_in]
output = x @ weights2 + bias2     # [B, C_original, T_out]
# weights2: [T_in, T_out]
```

#### Weight Initialization
- Weights: Normal distribution (mean=0, std=0.001)
- Bias: Uniform distribution (bound = 1/sqrt(fan_in))

---

## 🔄 Cơ Chế Hoạt Động Trong Phase 2

### Training Flow

```
1. Input: batch_data [B, C, T] (original time series)
    ↓
2. Random Time Masking
   masked_data = time_masking(batch_data)
    ↓
3. Augmentation (FROZEN, eval mode)
   augmented_data = augmentation(masked_data)  # [B, C, T]
    ↓
4. AGF-TCN Forward Pass
   reconstructed = agf_tcn(augmented_data)  # [B, C, T]
    ↓
5. Loss Calculation
   loss = MSE(reconstructed, augmented_data)
    ↓
6. Backward Pass
   - Chỉ update AGF-TCN parameters
   - Augmentation module FROZEN (không update)
```

### Inference Flow

```
1. Input: test_data [B, C, T]
    ↓
2. Inference Masking
   - Window 0: Random masking
   - Window i>0: Mask last time-step only
    ↓
3. Augmentation (FROZEN, eval mode)
   augmented_data = augmentation(masked_data)
    ↓
4. AGF-TCN Forward Pass (eval mode)
   reconstructed = agf_tcn(augmented_data)
    ↓
5. Anomaly Score Calculation
   score = MSE(reconstructed, augmented_data) per time-step
   # Shape: [B, T] - 1 score per time-step
    ↓
6. Map to Time Series
   - Window 0: Map all scores
   - Window i>0: Map only last score
    ↓
7. Threshold & Prediction
   predictions = (scores >= threshold).astype(int)
```

---

## 📊 Chi Tiết Forward Pass

### Ví Dụ Cụ Thể

**Input**: `augmented_data` shape `[64, 2, 16]`
- Batch size: 64
- Channels: 2
- Time steps: 16

**Config**: 
```python
num_channels = [256]  # 1 TCN layer
dilation = 1          # Layer 0
```

#### Step 1: Encoder Network

**Layer 0 (AttentionFusionBlock)**:
- Input: `[64, 2, 16]`
- Dilation: 1

**Branch 1 (Kernel=3)**:
```
Conv1d(2 → 256, kernel=3, dilation=1, padding=2)
  → [64, 256, 18]
Chomp1d(2) → [64, 256, 16]
Dropout(0.1)
GELU()
SEBlock(256) → [64, 256, 16]
Conv1d(256 → 256, kernel=3, dilation=1, padding=2)
  → [64, 256, 18]
Chomp1d(2) → [64, 256, 16]
Dropout(0.1)
GELU()
SEBlock(256) → [64, 256, 16]
```

**Branch 2 (Kernel=5)**: Tương tự, output `[64, 256, 16]`
**Branch 3 (Kernel=7)**: Tương tự, output `[64, 256, 16]`

**Fusion (AvgFusion)**:
```
output = (branch1 + branch2 + branch3) / 3
→ [64, 256, 16]
```

#### Step 2: Decoder (CustomLinear)

```
Input: [64, 256, 16]
Weights1: [2, 256]
Bias1: [2, 16]

Output = Weights1 @ Input + Bias1
→ [64, 2, 16]
```

**Final Output**: `[64, 2, 16]` (giống input shape)

---

## 🎯 Tại Sao AGF-TCN Hiệu Quả?

### 1. **Multi-Scale Feature Extraction**
- 3 branches với kernel sizes khác nhau capture patterns ở nhiều scales
- Giúp model hiểu cả local và global patterns

### 2. **Dilated Convolutions**
- Tăng receptive field mà không tăng parameters
- Capture long-range dependencies hiệu quả

### 3. **Squeeze-and-Excitation**
- Tự động học channel nào quan trọng
- Recalibrate features để tập trung vào thông tin quan trọng

### 4. **Causal Convolution**
- Đảm bảo không leak thông tin từ tương lai
- Phù hợp cho time series

### 5. **Weight Normalization**
- Stabilize training
- Improve convergence

---

## ⚙️ Các Tham Số Quan Trọng

### `num_channels`
- **Mô tả**: Số channels ở mỗi TCN layer
- **Ví dụ**: `[256]` → 1 layer với 256 channels
- **Ví dụ**: `[128, 256]` → 2 layers: 128 → 256 channels
- **Ảnh hưởng**: 
  - Nhiều channels → nhiều features, nhưng nhiều parameters hơn
  - Nhiều layers → deeper network, capture complex patterns

### `dropout`
- **Mô tả**: Dropout rate trong AttentionFusionBlock
- **Default**: 0.1
- **Ảnh hưởng**: 
  - Cao → regularization mạnh, tránh overfitting
  - Thấp → model học tốt hơn nhưng dễ overfit

### `activation`
- **Options**: "relu", "gelu", "silu", "elu", "leak_relu", "swish"
- **Default**: "gelu"
- **Ảnh hưởng**: 
  - GELU: Smooth, tốt cho deep networks
  - ReLU: Đơn giản, nhanh
  - SiLU/Swish: Tốt cho một số tasks

### `fuse_type`
- **1**: AddFusion (cộng)
- **2**: AvgFusion (trung bình)
- **3**: MulFusion (nhân)
- **4**: ConcatFusion (nối + project)
- **5**: TripConFusion (project + nhân)
- **Ảnh hưởng**: Cách kết hợp 3 branches ảnh hưởng đến representation

### `window_size`
- **Mô tả**: Độ dài time series
- **Phải khớp**: Với window_size từ Phase 1
- **Ảnh hưởng**: 
  - Ngắn → ít context, nhanh hơn
  - Dài → nhiều context, chậm hơn

---

## 📝 Ví Dụ Sử Dụng

### Khởi Tạo Model

```python
from phase2.agf_tcn import Agf_TCN

# Config từ Phase 2 training
agf_tcn = Agf_TCN(
    num_inputs=2,              # Số channels
    num_channels=[256],        # TCN layers
    dropout=0.1,
    activation="gelu",
    fuse_type=2,              # AvgFusion
    window_size=16
)

# Forward pass
augmented_data = torch.randn(64, 2, 16)  # [B, C, T]
reconstructed = agf_tcn(augmented_data)  # [B, C, T]
print(reconstructed.shape)  # torch.Size([64, 2, 16])
```

### Trong Training Loop

```python
# Training mode
agf_tcn.train()

# Forward
reconstructed = agf_tcn(augmented_data)

# Loss
loss = nn.MSELoss()(reconstructed, augmented_data)

# Backward
loss.backward()
optimizer.step()
```

### Trong Inference

```python
# Eval mode
agf_tcn.eval()

with torch.no_grad():
    reconstructed = agf_tcn(augmented_data)
    
    # Anomaly score per time-step
    anomaly_scores = torch.mean(
        (reconstructed - augmented_data) ** 2, 
        dim=1  # Average over channels
    )
    # Shape: [B, T]
```

---

## 🔍 So Sánh Với Standard TCN

| Feature | Standard TCN | AGF-TCN |
|---------|--------------|---------|
| **Kernel Size** | Single (thường 3) | Multi-scale (3, 5, 7) |
| **Branches** | 1 | 3 parallel branches |
| **Fusion** | None | 5 fusion types |
| **Attention** | None | SEBlock (channel attention) |
| **Receptive Field** | Limited | Multi-scale + dilation |
| **Parameters** | Fewer | More (nhưng hiệu quả hơn) |

---

## 🐛 Debug Tips

### 1. Kiểm Tra Shape
```python
print(f"Input: {x.shape}")
x = agf_tcn.network(x)  # Encoder
print(f"After encoder: {x.shape}")
x = agf_tcn.decoder(x)  # Decoder
print(f"After decoder: {x.shape}")
```

### 2. Kiểm Tra Gradient
```python
for name, param in agf_tcn.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm()}")
    else:
        print(f"{name}: No gradient!")
```

### 3. Kiểm Tra Output Range
```python
reconstructed = agf_tcn(augmented_data)
print(f"Reconstructed range: [{reconstructed.min():.4f}, {reconstructed.max():.4f}]")
print(f"Augmented range: [{augmented_data.min():.4f}, {augmented_data.max():.4f}]")
```

---

## 📚 Tài Liệu Tham Khảo

- **Temporal Convolutional Networks (TCN)**: Bai et al., 2018
- **Squeeze-and-Excitation Networks**: Hu et al., 2018
- **Dilated Convolutions**: Yu & Koltun, 2016
- **Weight Normalization**: Salimans & Kingma, 2016

---

## ⚠️ Lưu Ý Quan Trọng

1. **Augmentation Module FROZEN**: Trong Phase 2, augmentation module không được update, chỉ AGF-TCN được train
2. **Input Phải Là Augmented Data**: AGF-TCN nhận input là output từ augmentation module, không phải raw data
3. **Sequence Length**: Window size phải khớp với Phase 1
4. **Normalization**: Dữ liệu đã được normalize về [0, 1] trước khi vào model
5. **Eval Mode**: Trong inference, phải set `agf_tcn.eval()` để tắt dropout và batch norm updates

---

## 🎓 Kết Luận

AGF-TCN là một kiến trúc mạnh mẽ kết hợp:
- **Multi-scale feature extraction** (3 branches)
- **Channel attention** (SEBlock)
- **Dilated convolutions** (temporal dependencies)
- **Flexible fusion** (5 fusion types)

Module này được thiết kế đặc biệt để học reconstruction từ augmented data, tạo ra anomaly scores dựa trên reconstruction error trong Phase 2 của hệ thống Anomaly Detection.


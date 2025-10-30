import numpy as np
from typing import Tuple


def create_sliding_windows(
    data: np.ndarray,
    window_size: int,
    stride: int = 1
) -> np.ndarray:
    """
    Tạo sliding windows từ time series data.
    
    Args:
        data: Input time series data với shape (n_channels, time_steps)
        window_size: Kích thước của mỗi window
        stride: Bước nhảy giữa các windows (default: 1)
    
    Returns:
        windows: Array với shape (n_windows, n_channels, window_size)
    
    Raises:
        ValueError: Nếu time_steps < window_size
        ValueError: Nếu stride không chia hết (time_steps - window_size)
    """
    n_channels, time_steps = data.shape
    
    # Kiểm tra điều kiện cơ bản
    if time_steps < window_size:
        raise ValueError(
            f"time_steps ({time_steps}) phải >= window_size ({window_size})"
        )
    
    # Kiểm tra stride có chia hết không
    remainder = (time_steps - window_size) % stride
    if remainder != 0:
        raise ValueError(
            f"Stride không chia hết dữ liệu!\n"
            f"  - time_steps: {time_steps}\n"
            f"  - window_size: {window_size}\n"
            f"  - stride: {stride}\n"
            f"  - Phần dư: {remainder} timesteps\n"
            f"  - Để phủ hết dữ liệu, (time_steps - window_size) phải chia hết cho stride\n"
            f"  - Gợi ý stride hợp lệ: các ước của {time_steps - window_size}"
        )
    
    # Tính số lượng windows
    n_windows = (time_steps - window_size) // stride + 1
    
    # Khởi tạo array cho windows
    windows = np.zeros((n_windows, n_channels, window_size), dtype=data.dtype)
    
    # Tạo sliding windows
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        windows[i] = data[:, start_idx:end_idx]
    
    return windows


def create_train_test_windows(
    train_data: np.ndarray,
    test_data: np.ndarray,
    window_size: int,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tạo sliding windows cho cả train và test data.
    
    Args:
        train_data: Training data với shape (n_channels, time_steps)
        test_data: Test data với shape (n_channels, time_steps)
        window_size: Kích thước của mỗi window
        stride: Bước nhảy giữa các windows
    
    Returns:
        train_windows: Array với shape (n_train_windows, n_channels, window_size)
        test_windows: Array với shape (n_test_windows, n_channels, window_size)
    """
    # Tạo windows cho train data
    train_windows = create_sliding_windows(
        train_data, 
        window_size, 
        stride
    )
    
    # Tạo windows cho test data
    test_windows = create_sliding_windows(
        test_data, 
        window_size, 
        stride
    )
    
    return train_windows, test_windows


def find_valid_strides(time_steps: int, window_size: int) -> list:
    """
    Tìm tất cả các giá trị stride hợp lệ cho dữ liệu.
    
    Args:
        time_steps: Độ dài time series
        window_size: Kích thước window
    
    Returns:
        List các giá trị stride hợp lệ (các ước của time_steps - window_size)
    """
    if time_steps < window_size:
        return []
    
    diff = time_steps - window_size
    if diff == 0:
        return [1]  # Chỉ có 1 window
    
    # Tìm tất cả ước của diff
    valid_strides = []
    for i in range(1, diff + 1):
        if diff % i == 0:
            valid_strides.append(i)
    
    return valid_strides


def batch_windows(windows: np.ndarray, batch_size: int) -> list:
    """
    Chia windows thành các batches.
    
    Args:
        windows: Array với shape (n_windows, n_channels, window_size)
        batch_size: Kích thước của mỗi batch
    
    Returns:
        List các batches
    """
    n_windows = windows.shape[0]
    batches = []
    
    for i in range(0, n_windows, batch_size):
        batch = windows[i:min(i + batch_size, n_windows)]
        batches.append(batch)
    
    return batches


# Example usage
if __name__ == "__main__":
    print("=== Demo Sliding Window với Stride Validation ===\n")
    
    # Tạo dữ liệu giả
    n_channels = 3
    time_steps = 1000
    window_size = 50
    
    train_data = np.random.rand(n_channels, time_steps)
    test_data = np.random.rand(n_channels, 500)
    
    print(f"Train data: shape {train_data.shape}")
    print(f"Test data: shape {test_data.shape}")
    print(f"Window size: {window_size}\n")
    
    # Tìm các stride hợp lệ
    print("--- Các stride hợp lệ cho train data ---")
    valid_strides_train = find_valid_strides(time_steps, window_size)
    print(f"Có {len(valid_strides_train)} giá trị stride hợp lệ")
    print(f"Một số ví dụ: {valid_strides_train[:10]}\n")
    
    # Demo với stride hợp lệ
    print("--- Test với stride HỢP LỆ (stride=50) ---")
    try:
        train_windows, test_windows = create_train_test_windows(
            train_data=train_data,
            test_data=test_data,
            window_size=window_size,
            stride=50  # 950 % 50 = 0 ✓
        )
        print(f"✓ Train windows: {train_windows.shape}")
        print(f"✓ Test windows: {test_windows.shape}")
        
        # Verify coverage
        last_end = (train_windows.shape[0] - 1) * 50 + window_size
        print(f"✓ Train data coverage: {last_end}/{time_steps} (đủ 100%)\n")
    except ValueError as e:
        print(f"✗ Lỗi: {e}\n")
    
    # Demo với stride KHÔNG hợp lệ
    print("--- Test với stride KHÔNG HỢP LỆ (stride=30) ---")
    try:
        train_windows, test_windows = create_train_test_windows(
            train_data=train_data,
            test_data=test_data,
            window_size=window_size,
            stride=30  # 950 % 30 = 20 ✗
        )
        print(f"Train windows: {train_windows.shape}")
        print(f"Test windows: {test_windows.shape}")
    except ValueError as e:
        print(f"✗ Lỗi được phát hiện (như mong đợi):")
        print(f"{e}\n")
    
    # Gợi ý stride cho user
    print("--- Gợi ý stride hợp lý ---")
    suggestions = [s for s in valid_strides_train if 1 < s <= 100][:5]
    print(f"Các stride từ 2-100: {suggestions}")


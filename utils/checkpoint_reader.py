"""
Checkpoint Reader - Đọc và hiển thị thông tin chi tiết từ checkpoint
Hỗ trợ checkpoint từ Phase 1 (.pth) và Phase 2 (.pt)
"""

import torch
import os
import argparse
from pathlib import Path
import pickle


def format_size(num_params):
    """Format số parameters thành dạng dễ đọc"""
    if num_params >= 1e6:
        return f"{num_params/1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params/1e3:.2f}K"
    else:
        return str(num_params)


def print_separator(char="=", length=80):
    """In dòng phân cách"""
    print(char * length)


def print_dict_recursive(d, indent=0, max_depth=3):
    """In dictionary đệ quy với giới hạn độ sâu"""
    if indent > max_depth:
        return

    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + f"├─ {key}:")
            print_dict_recursive(value, indent + 1, max_depth)
        elif isinstance(value, (list, tuple)) and len(value) > 0:
            print(
                "  " * indent
                + f"├─ {key}: {type(value).__name__} (length: {len(value)})"
            )
            if len(value) <= 5:
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        print("  " * (indent + 1) + f"[{i}]:")
                        print_dict_recursive(item, indent + 2, max_depth)
                    else:
                        print("  " * (indent + 1) + f"[{i}]: {item}")
        else:
            print("  " * indent + f"├─ {key}: {value}")


def read_checkpoint_info(checkpoint_path, verbose=False):
    """
    Đọc và hiển thị thông tin từ checkpoint

    Args:
        checkpoint_path: Đường dẫn đến file checkpoint
        verbose: Nếu True, hiển thị thông tin chi tiết về từng layer
    """
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: File không tồn tại: {checkpoint_path}")
        return

    print_separator("=")
    print(f"📂 CHECKPOINT READER")
    print_separator("=")
    print(f"File: {checkpoint_path}")
    print(f"Size: {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB")
    print_separator("-")

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Kiểm tra xem có phải là file .pkl không
        if checkpoint_path.endswith(".pkl"):
            print("⚠️  Warning: File .pkl có thể không phải checkpoint PyTorch chuẩn")
            if isinstance(checkpoint, dict):
                print(f"✓ Checkpoint chứa {len(checkpoint)} keys")
            return

        print(f"\n{'='*80}")
        print("📊 THÔNG TIN TỔNG QUAN")
        print(f"{'='*80}")

        # 1. Thông tin epoch và metrics
        if "epoch" in checkpoint:
            print(f"├─ Epoch: {checkpoint['epoch']}")

        if "loss" in checkpoint:
            print(f"├─ Loss: {checkpoint['loss']:.6f}")

        if "metrics" in checkpoint:
            print(f"├─ Metrics:")
            metrics = checkpoint["metrics"]
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"│  ├─ {key}: {value:.6f}")
                    else:
                        print(f"│  ├─ {key}: {value}")

        # 2. Thông tin config
        print(f"\n{'='*80}")
        print("⚙️  CONFIGURATION")
        print(f"{'='*80}")

        if "config" in checkpoint:
            config = checkpoint["config"]
            if isinstance(config, dict):
                print_dict_recursive(config, indent=0, max_depth=3)
            else:
                print(f"Config type: {type(config)}")
                print(config)
        else:
            print("❌ Không tìm thấy config trong checkpoint")

        # 3. Thông tin các state_dict
        print(f"\n{'='*80}")
        print("🧠 MODEL STATE DICTIONARIES")
        print(f"{'='*80}")

        state_dict_keys = [k for k in checkpoint.keys() if "state_dict" in k]

        for state_key in state_dict_keys:
            state_dict = checkpoint[state_key]
            model_name = state_key.replace("_state_dict", "").upper()

            print(f"\n┌─ {model_name}")
            print(f"│  ├─ Số layers: {len(state_dict)}")

            # Tính tổng số parameters
            total_params = sum(
                p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor)
            )
            print(
                f"│  ├─ Tổng parameters: {format_size(total_params)} ({total_params:,})"
            )

            if verbose:
                print(f"│  └─ Layer details:")
                for i, (name, param) in enumerate(state_dict.items()):
                    if isinstance(param, torch.Tensor):
                        num_params = param.numel()
                        print(f"│     ├─ [{i+1}] {name}")
                        print(f"│     │   ├─ Shape: {tuple(param.shape)}")
                        print(f"│     │   ├─ Dtype: {param.dtype}")
                        print(
                            f"│     │   └─ Params: {format_size(num_params)} ({num_params:,})"
                        )
                    else:
                        print(f"│     ├─ [{i+1}] {name}: {type(param)}")
            else:
                # Chỉ hiển thị 5 layers đầu và cuối
                items = list(state_dict.items())
                print(f"│  └─ Layers (showing first 5 and last 5):")

                for i, (name, param) in enumerate(items[:5]):
                    if isinstance(param, torch.Tensor):
                        print(
                            f"│     ├─ [{i+1}] {name}: {tuple(param.shape)} - {format_size(param.numel())}"
                        )
                    else:
                        print(f"│     ├─ [{i+1}] {name}: {type(param)}")

                if len(items) > 10:
                    print(f"│     ├─ ... ({len(items) - 10} layers ẩn)")

                for i, (name, param) in enumerate(items[-5:], len(items) - 5):
                    if isinstance(param, torch.Tensor):
                        print(
                            f"│     ├─ [{i+1}] {name}: {tuple(param.shape)} - {format_size(param.numel())}"
                        )
                    else:
                        print(f"│     ├─ [{i+1}] {name}: {type(param)}")

        # 4. Thông tin optimizer
        if "optimizer_state_dict" in checkpoint:
            print(f"\n{'='*80}")
            print("🎯 OPTIMIZER STATE")
            print(f"{'='*80}")
            opt_state = checkpoint["optimizer_state_dict"]
            print(f"├─ Optimizer keys: {list(opt_state.keys())}")

            if "param_groups" in opt_state:
                for i, pg in enumerate(opt_state["param_groups"]):
                    print(f"├─ Param Group {i}:")
                    for key, value in pg.items():
                        if key != "params":
                            print(f"│  ├─ {key}: {value}")

        # 5. Thông tin scheduler (nếu có)
        if "scheduler_state_dict" in checkpoint:
            print(f"\n{'='*80}")
            print("📈 SCHEDULER STATE")
            print(f"{'='*80}")
            sched_state = checkpoint["scheduler_state_dict"]
            for key, value in sched_state.items():
                if not key.startswith("_"):
                    print(f"├─ {key}: {value}")

        # 6. Tổng kết
        print(f"\n{'='*80}")
        print("📋 TỔNG KẾT")
        print(f"{'='*80}")
        print(f"├─ Checkpoint keys: {list(checkpoint.keys())}")
        print(
            f"├─ Total checkpoint size: {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB"
        )
        print(f"└─ Status: ✓ Đọc thành công")
        print_separator("=")

    except Exception as e:
        print(f"\n❌ Error khi đọc checkpoint: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        import traceback

        traceback.print_exc()


def list_checkpoints(checkpoint_dir):
    """Liệt kê tất cả các checkpoint trong thư mục"""
    print_separator("=")
    print(f"📁 DANH SÁCH CHECKPOINTS")
    print_separator("=")
    print(f"Directory: {checkpoint_dir}\n")

    if not os.path.exists(checkpoint_dir):
        print(f"❌ Error: Thư mục không tồn tại: {checkpoint_dir}")
        return []

    checkpoints = []
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            if file.endswith((".pth", ".pt", ".pkl")):
                full_path = os.path.join(root, file)
                size_mb = os.path.getsize(full_path) / (1024 * 1024)
                rel_path = os.path.relpath(full_path, checkpoint_dir)
                checkpoints.append((rel_path, size_mb, full_path))

    if not checkpoints:
        print("❌ Không tìm thấy checkpoint nào (.pth, .pt, .pkl)")
        return []

    # Sắp xếp theo tên
    checkpoints.sort()

    print(f"Tìm thấy {len(checkpoints)} checkpoints:\n")
    for i, (rel_path, size_mb, full_path) in enumerate(checkpoints, 1):
        print(f"{i:3d}. {rel_path:<60s} ({size_mb:>8.2f} MB)")

    print_separator("=")
    return checkpoints


def compare_checkpoints(checkpoint_path1, checkpoint_path2):
    """So sánh 2 checkpoint"""
    print_separator("=")
    print(f"⚖️  SO SÁNH 2 CHECKPOINTS")
    print_separator("=")

    try:
        cp1 = torch.load(checkpoint_path1, map_location="cpu")
        cp2 = torch.load(checkpoint_path2, map_location="cpu")

        print(f"Checkpoint 1: {os.path.basename(checkpoint_path1)}")
        print(f"Checkpoint 2: {os.path.basename(checkpoint_path2)}")
        print()

        # So sánh epoch
        if "epoch" in cp1 and "epoch" in cp2:
            print(
                f"Epoch: {cp1['epoch']} vs {cp2['epoch']} (Δ={cp2['epoch']-cp1['epoch']})"
            )

        # So sánh loss
        if "loss" in cp1 and "loss" in cp2:
            print(
                f"Loss: {cp1['loss']:.6f} vs {cp2['loss']:.6f} (Δ={cp2['loss']-cp1['loss']:.6f})"
            )

        # So sánh metrics
        if "metrics" in cp1 and "metrics" in cp2:
            print("\nMetrics comparison:")
            metrics1 = cp1["metrics"]
            metrics2 = cp2["metrics"]
            if isinstance(metrics1, dict) and isinstance(metrics2, dict):
                all_keys = set(metrics1.keys()) | set(metrics2.keys())
                for key in sorted(all_keys):
                    v1 = metrics1.get(key, "N/A")
                    v2 = metrics2.get(key, "N/A")
                    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                        print(f"  {key}: {v1:.6f} vs {v2:.6f} (Δ={v2-v1:.6f})")
                    else:
                        print(f"  {key}: {v1} vs {v2}")

        # So sánh số lượng parameters
        print("\nModel parameters:")
        for state_key in [k for k in cp1.keys() if "state_dict" in k]:
            if state_key in cp2:
                params1 = sum(
                    p.numel()
                    for p in cp1[state_key].values()
                    if isinstance(p, torch.Tensor)
                )
                params2 = sum(
                    p.numel()
                    for p in cp2[state_key].values()
                    if isinstance(p, torch.Tensor)
                )
                model_name = state_key.replace("_state_dict", "")
                print(
                    f"  {model_name}: {format_size(params1)} vs {format_size(params2)}"
                )

        print_separator("=")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Checkpoint Reader - Đọc thông tin chi tiết từ PyTorch checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Đọc thông tin từ 1 checkpoint
  python checkpoint_reader.py -f phase1/checkpoints/ecg_best_model.pth
  
  # Đọc với thông tin chi tiết
  python checkpoint_reader.py -f phase1/checkpoints/ecg_best_model.pth -v
  
  # Liệt kê tất cả checkpoints trong thư mục
  python checkpoint_reader.py -l phase1/checkpoints
  
  # So sánh 2 checkpoints
  python checkpoint_reader.py -c checkpoint1.pth checkpoint2.pth
        """,
    )

    parser.add_argument(
        "-f", "--file", type=str, help="Đường dẫn đến file checkpoint cần đọc"
    )
    parser.add_argument(
        "-l", "--list", type=str, help="Liệt kê tất cả checkpoints trong thư mục"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Hiển thị thông tin chi tiết về tất cả layers",
    )
    parser.add_argument(
        "-c",
        "--compare",
        nargs=2,
        metavar=("CP1", "CP2"),
        help="So sánh 2 checkpoint files",
    )

    args = parser.parse_args()

    # Nếu không có argument nào, hiển thị help
    if not any([args.file, args.list, args.compare]):
        parser.print_help()
        return

    # Xử lý list checkpoints
    if args.list:
        list_checkpoints(args.list)
        return

    # Xử lý compare
    if args.compare:
        compare_checkpoints(args.compare[0], args.compare[1])
        return

    # Xử lý đọc single checkpoint
    if args.file:
        read_checkpoint_info(args.file, verbose=args.verbose)


if __name__ == "__main__":
    main()

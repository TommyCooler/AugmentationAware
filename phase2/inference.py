"""
Phase 2 Inference: Anomaly Detection với Best Threshold Search

Usage:
    python phase2/inference.py --checkpoint checkpoints/phase2_ucr_135_best.pt
    
hoặc tùy chỉnh:
    python phase2/inference.py --checkpoint <path> --dataset ucr --subset 135
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.phase2_dataloader import prepare_phase2_data, Phase2Dataset
from data.dataloader import (
    ucr_sub_ds_processing,
    smd_sub_ds_processing,
    smap_msl_sub_ds_processing,
    psm_sub_ds_processing
)
from modules.augmentation import Augmentation
from phase2.agf_tcn import Agf_TCN
from utils.point_adjustment import evaluate_with_pa


class Phase2Inference:
    """
    Inference cho Phase 2 Anomaly Detection
    """
    
    def __init__(self, checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        print(f"\n📦 Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract config
        if 'config' not in checkpoint:
            raise ValueError("Checkpoint thiếu 'config'! Không thể tái tạo model.")
        
        self.config = checkpoint['config']
        print(f"  ✓ Config loaded")
        print(f"    Dataset: {self.config['dataset_name']}_{self.config['subset']}")
        print(f"    Window size: {self.config['window_size']}")
        print(f"    Channels: {self.config['n_channels']}")
        
        # Recreate Augmentation
        self.augmentation = Augmentation(
            in_channels=self.config['n_channels'],
            seq_len=self.config['window_size'],
            out_channels=self.config['n_channels'],
            kernel_size=3,
            num_layers=2,
            dropout=0.1,
            temperature=1.0,
            hard=False,
            transformer_d_model=self.config.get('transformer_d_model', 128),
            transformer_nhead=self.config.get('transformer_nhead', 2)
        )
        self.augmentation.load_state_dict(checkpoint['augmentation_state_dict'])
        self.augmentation.to(device)
        self.augmentation.eval()
        print(f"  ✓ Augmentation loaded and frozen")
        
        # Recreate AGF-TCN
        self.agf_tcn = Agf_TCN(
            num_inputs=self.config['n_channels'],
            num_channels=self.config['agf_tcn_channels'],
            dropout=self.config['dropout'],
            activation=self.config['activation'],
            fuse_type=self.config['fuse_type'],
            window_size=self.config['window_size']
        )
        self.agf_tcn.load_state_dict(checkpoint['agf_tcn_state_dict'])
        self.agf_tcn.to(device)
        self.agf_tcn.eval()
        print(f"  ✓ AGF-TCN loaded")
        
        # Print checkpoint info
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            print(f"\n📊 Checkpoint metrics (from training):")
            print(f"  Test Loss: {metrics.get('test_loss', 'N/A'):.6f}")
            print(f"  F1 (training threshold): {metrics.get('f1', 'N/A'):.4f}")
            print(f"  Precision: {metrics.get('precision', 'N/A'):.4f}")
            print(f"  Recall: {metrics.get('recall', 'N/A'):.4f}")
    
    def predict(self, test_loader, threshold=None, search_best_threshold=True, use_point_adjustment=True):
        
        # Ensure models are in eval mode (disable dropout, batch norm, etc.)
        self.augmentation.eval()
        self.agf_tcn.eval()
        
        print("\n🔮 Running inference...")
        all_losses = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data, batch_labels in tqdm(test_loader, desc="Computing anomaly scores"):
                batch_data = batch_data.to(self.device)
                
                # Get augmented data
                augmented_data = self.augmentation(batch_data)
                
                # Reconstruct
                reconstructed = self.agf_tcn(augmented_data)
                
                # Compute reconstruction error per sample
                batch_losses = torch.mean((reconstructed - augmented_data) ** 2, dim=(1, 2))
                
                all_losses.append(batch_losses.cpu().numpy())
                all_labels.append(batch_labels.cpu().numpy())
        
        # Concatenate all results
        anomaly_scores = np.concatenate(all_losses)
        true_labels = np.concatenate(all_labels).astype(int)
        
        print(f"  ✓ Computed {len(anomaly_scores)} anomaly scores")
        
        # Evaluate with Point Adjustment
        if search_best_threshold:
            print("\n🔍 Searching for best threshold...")
        
        metrics = evaluate_with_pa(
            anomaly_scores=anomaly_scores,
            labels=true_labels,
            threshold=threshold,
            apply_pa=use_point_adjustment,
            search_best_threshold=search_best_threshold
        )
        
        # Add average reconstruction error
        metrics['test_loss'] = np.mean(anomaly_scores)
        
        return metrics, anomaly_scores, true_labels
    
    def print_results(self, metrics):
        """In kết quả inference"""
        print("\n" + "="*60)
        print("INFERENCE RESULTS")
        print("="*60)
        
        print(f"\n📊 Reconstruction Loss:")
        print(f"  Test Loss (avg): {metrics['test_loss']:.6f}")
        
        if 'best_threshold' in metrics:
            print(f"\n🎯 Best Threshold Search:")
            print(f"  Best Threshold: {metrics['best_threshold']:.6f}")
            print(f"  Best F1-Score: {metrics['best_f1']:.4f}")
            print(f"  Best Precision: {metrics.get('best_precision', 0):.4f}")
            print(f"  Best Recall: {metrics.get('best_recall', 0):.4f}")
            print(f"  Best Accuracy: {metrics.get('best_accuracy', 0):.4f}")
        else:
            print(f"\n🎯 Results (threshold={metrics.get('threshold', 'N/A'):.6f}):")
            print(f"  F1-Score: {metrics['f1']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        if 'total_segments' in metrics and metrics['total_segments'] > 0:
            print(f"\n📈 Anomaly Segments:")
            print(f"  Total segments: {metrics['total_segments']}")
            print(f"  Detected segments: {metrics['detected_segments']}")
            print(f"  Detection rate: {metrics['segment_detection_rate']:.2%}")
        
        if 'TP' in metrics:
            print(f"\n🔢 Confusion Matrix:")
            print(f"  TP: {metrics['TP']}, FP: {metrics['FP']}")
            print(f"  FN: {metrics['FN']}, TN: {metrics['TN']}")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Phase 2 Inference')
    parser.add_argument('--checkpoint', type=str, default='phase2/checkpoints/phase2_ucr_135_best.pt',
                      help='Path to checkpoint file')
    parser.add_argument('--dataset', type=str, default=None,
                      help='Dataset name (optional, will use from checkpoint if not provided)')
    parser.add_argument('--subset', type=str, default=None,
                      help='Dataset subset (optional, will use from checkpoint if not provided)')
    parser.add_argument('--batch_size', type=int, default=None,
                      help='Batch size for inference (optional, will use from checkpoint if not provided)')
    parser.add_argument('--threshold', type=float, default=None,
                      help='Fixed threshold (if not set, will search for best)')
    parser.add_argument('--no_pa', action='store_true',
                      help='Disable Point Adjustment')
    parser.add_argument('--no_search', action='store_true',
                      help='Disable threshold search (use 95th percentile)')
    parser.add_argument('--device', type=str, default=None,
                      help='Device (cuda/cpu), auto-detect if not set')
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("Phase 2: Inference Mode")
    print("="*60)
    
    # Load model
    inferencer = Phase2Inference(args.checkpoint, device=args.device)
    
    # Get config from checkpoint
    config = inferencer.config
    
    # Override dataset if provided
    dataset_name = args.dataset if args.dataset else config['dataset_name']
    subset = args.subset if args.subset else config['subset']
    batch_size = args.batch_size if args.batch_size else config.get('batch_size', 32)
    
    print(f"\n[1/3] Loading test data: {dataset_name}_{subset}")
    print(f"  Batch size: {batch_size}")
    
    # Load data
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path_base = os.path.join(project_root, 'data', 'datasets')
    
    dataloader_func = {
        'ucr': ucr_sub_ds_processing,
        'smd': smd_sub_ds_processing,
        'smap_msl': smap_msl_sub_ds_processing,
        'psm': psm_sub_ds_processing,
    }
    
    loader_func = dataloader_func[dataset_name]
    
    # Prepare data (CHỈ load test data cho inference)
    _, _, test_windows, test_labels = prepare_phase2_data(
        dataset_name=dataset_name,
        subset=subset,
        loader_func=loader_func,
        window_size=config['window_size'],
        stride=config.get('stride', 1),
        data_path_base=data_path_base
    )
    
    print(f"  ✓ Test set: {len(test_windows)} windows")
    print(f"  ✓ Anomaly rate: {test_labels.sum() / len(test_labels) * 100:.2f}%")
    
    # Create test dataloader only
    print("\n[2/3] Creating test dataloader...")
    test_dataset = Phase2Dataset(test_windows, test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    print(f"  ✓ Test batches: {len(test_loader)}")
    
    # Run inference
    print("\n[3/3] Running inference...")
    metrics, anomaly_scores, true_labels = inferencer.predict(
        test_loader,
        threshold=args.threshold,
        search_best_threshold=not args.no_search,
        use_point_adjustment=not args.no_pa
    )
    
    # Print results
    inferencer.print_results(metrics)
    
    # Save results (optional)
    output_dir = os.path.join(project_root, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(
        output_dir,
        f"inference_{dataset_name}_{subset}_results.npz"
    )
    
    np.savez(
        output_file,
        anomaly_scores=anomaly_scores,
        true_labels=true_labels,
        metrics=metrics
    )
    
    print(f"\n💾 Results saved to: {output_file}")


if __name__ == '__main__':
    main()

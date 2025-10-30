import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.phase1_dataloader import prepare_phase1_data, create_phase1_dataloader

from data.dataloader import (
    ucr_sub_ds_processing,
    smd_sub_ds_processing,
    smap_msl_sub_ds_processing,
    psm_sub_ds_processing
)
from modules.augmentation import Augmentation
from phase1.encoder import MLPEncoder, CNNEncoder
from pytorch_metric_learning import losses

import setproctitle
setproctitle.setproctitle("Tran Chi Tam - Phase 1 Training")



class ContrastiveDataset(torch.utils.data.Dataset):
    """
    Dataset that applies augmentation on-the-fly to create positive pairs.
    """
    
    def __init__(self, windows, augmentation_module):
        """
        Args:
            windows: numpy array of shape (n_windows, n_channels, window_size)
            augmentation_module: nn.Module for data augmentation
        """
        self.windows = torch.from_numpy(windows).float()
        self.augmentation = augmentation_module
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        """
        Returns:
            original: original window
            augmented: augmented version of the same window
        """
        original = self.windows[idx]
        
        # Apply augmentation
        with torch.no_grad():
            augmented = self.augmentation(original.unsqueeze(0)).squeeze(0)
        
        return original, augmented


class Phase1Trainer:
    """
    Trainer for Phase 1: Contrastive learning with NTXent loss.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        augmentation: nn.Module,
        temperature: float = 0.07,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_scheduler: bool = True,
        use_grad_clip: bool = True,
        max_grad_norm: float = 1.0,
        config: dict = None  # Store config for inference
    ):
        self.encoder = encoder.to(device)
        self.augmentation = augmentation.to(device)
        self.device = device
        self.use_scheduler = use_scheduler
        self.use_grad_clip = use_grad_clip
        self.max_grad_norm = max_grad_norm
        self.config = config or {}  # Store for saving to checkpoint
        
        # NTXent Loss
        self.criterion = losses.NTXentLoss(temperature=temperature)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            list(self.encoder.parameters()) + list(self.augmentation.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler (optional)
        self.scheduler = None
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=100,
                eta_min=1e-6
            )
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.encoder.train()
        self.augmentation.train()
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch_windows in enumerate(pbar):
            batch_windows = batch_windows.to(self.device)
            batch_size = batch_windows.shape[0]
            
            # Apply augmentation to create positive pairs
            augmented_windows = self.augmentation(batch_windows)
            
            # Encode both original and augmented
            z_original = self.encoder(batch_windows)      # (batch, proj_dim)
            z_augmented = self.encoder(augmented_windows) # (batch, proj_dim)
            
            # Create embeddings and labels for NTXent loss
            # Concatenate: [z1_1, z1_2, ..., z2_1, z2_2, ...]
            embeddings = torch.cat([z_original, z_augmented], dim=0)
            
            # Labels: each pair has same label
            # [0, 1, 2, ..., 0, 1, 2, ...] so that (i, i+batch_size) forms positive pair
            labels = torch.arange(batch_size).repeat(2).to(self.device)
            
            # Compute NTXent loss
            loss = self.criterion(embeddings, labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (optional)
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.augmentation.parameters()),
                    max_norm=self.max_grad_norm
                )
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, dataloader, num_epochs, save_dir='checkpoints'):
        """Full training loop"""
        os.makedirs(save_dir, exist_ok=True)
        
        best_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            avg_loss = self.train_epoch(dataloader, epoch)
            
            print(f"Epoch {epoch}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
            
            # Learning rate scheduling (optional)
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Save checkpoint only when loss improves
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(
                    os.path.join(save_dir, 'best_model.pth'),
                    epoch,
                    avg_loss
                )
                print(f"✓ Saved best model with loss: {best_loss:.4f}")
    
    def save_checkpoint(self, path, epoch, loss):
        """Save model checkpoint with full config for inference"""
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'augmentation_state_dict': self.augmentation.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config,  # Save full config for inference
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"  ✓ Checkpoint saved with config: {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.augmentation.load_state_dict(checkpoint['augmentation_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['loss']


def main():
    """
    Main training script for Phase 1.
    Example: Train on UCR datasets 135-138
    """
    
    # Configuration
    config = {
        'window_size': 16,
        'stride': 1,
        'batch_size': 64,
        'num_epochs': 200,
        'learning_rate': 1e-4,
        'temperature': 0.07,
        'projection_dim': 256,
        'encoder_type': 'mlp',  # 'mlp' or 'cnn'
        
        # Augmentation Transformer hyperparameters
        'transformer_d_model': 128,  # Transformer hidden dimension
        'transformer_nhead': 2,       # Number of attention heads (must be even)
        
        # Training options
        'use_scheduler': False,  # Use learning rate scheduler
        'use_grad_clip': False,  # Use gradient clipping
        'max_grad_norm': 1.0,   # Max gradient norm for clipping
        
    }
    
    print("=" * 60)
    print("Phase 1: Contrastive Learning Training")
    print("=" * 60)
    
    # Step 1: Prepare datasets configuration
    # Example: UCR datasets 135-138
    datasets_info = [
        {'name': 'ucr', 'subset': '135', 'loader': 'ucr'},
        {'name': 'ucr', 'subset': '136', 'loader': 'ucr'},
        {'name': 'ucr', 'subset': '137', 'loader': 'ucr'},
        {'name': 'ucr', 'subset': '138', 'loader': 'ucr'},
    ]
    
    # Dataloader function mapping
    dataloader_func = {
        'ucr': ucr_sub_ds_processing,
        'smd': smd_sub_ds_processing,
        'smap_msl': smap_msl_sub_ds_processing,
        'psm': psm_sub_ds_processing,
    }
    
    # Step 2: Load and prepare data
    print("\n[1/5] Loading and preparing data...")
    # Đảm bảo đường dẫn tính từ project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path_base = os.path.join(project_root, 'data', 'datasets')
    
    all_train_windows, stats = prepare_phase1_data(
        datasets_info=datasets_info,
        window_size=config['window_size'],
        stride=config['stride'],
        dataloader_func=dataloader_func,
        data_path_base=data_path_base
    )
    
    print("\nDataset Statistics:")
    print(f"  Total windows: {stats['total_windows']}")
    print(f"  Window shape: {stats['shape']}")
    for ds_name, n_windows in zip(stats['datasets'], stats['n_windows_per_dataset']):
        print(f"  - {ds_name}: {n_windows} windows")
    
    # Get data dimensions
    n_channels = all_train_windows.shape[1]
    window_size = all_train_windows.shape[2]
    
    # Update config with actual dimensions (important for inference!)
    config['n_channels'] = n_channels
    config['window_size'] = window_size
    
    # Step 3: Initialize augmentation module
    print("\n[2/5] Initializing augmentation module...")
    augmentation = Augmentation(
        in_channels=n_channels,
        seq_len=window_size,
        out_channels=n_channels,
        kernel_size=3,
        num_layers=2,
        dropout=0.1,
        temperature=1.0,
        hard=False,
        transformer_d_model=config['transformer_d_model'],
        transformer_nhead=config['transformer_nhead']
    )
    
    # Step 4: Initialize encoder
    print("\n[3/5] Initializing encoder...")
    if config['encoder_type'] == 'mlp':
        encoder = MLPEncoder(
            input_channels=n_channels,
            window_size=window_size,
            hidden_dims=[256, 256],
            projection_dim=config['projection_dim'],
            dropout=0.1
        )
    else:  # 'cnn'
        encoder = CNNEncoder(
            input_channels=n_channels,
            hidden_channels=[64, 128, 256],
            kernel_size=3,
            projection_dim=config['projection_dim'],
            dropout=0.1
        )
    
    print(f"  Encoder: {config['encoder_type'].upper()}")
    print(f"  Input: ({n_channels}, {window_size})")
    print(f"  Output: ({config['projection_dim']},)")
    
    # Step 5: Create dataloader
    print("\n[4/5] Creating dataloader...")
    dataloader = create_phase1_dataloader(
        all_train_windows=all_train_windows,
        batch_size=config['batch_size'],
        num_workers=4,
        shuffle=True
    )
    print(f"  Batches per epoch: {len(dataloader)}")
    
    # Step 6: Initialize trainer and train
    print("\n[5/5] Starting training...")
    trainer = Phase1Trainer(
        encoder=encoder,
        augmentation=augmentation,
        temperature=config['temperature'],
        learning_rate=config['learning_rate'],
        weight_decay=1e-6,
        use_scheduler=config['use_scheduler'],
        use_grad_clip=config['use_grad_clip'],
        max_grad_norm=config['max_grad_norm'],
        config=config  # Pass full config for checkpoint
    )
    
    # Train
    trainer.train(
        dataloader=dataloader,
        num_epochs=config['num_epochs'],
        save_dir='checkpoints'
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()



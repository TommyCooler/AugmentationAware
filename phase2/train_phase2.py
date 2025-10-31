"""
Phase 2 Training: Supervised Anomaly Detection with AGF-TCN
- Load pre-trained Augmentation module (frozen)
- Train AGF-TCN for reconstruction
- MSE Loss between augmented data and reconstructed data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
from datetime import datetime

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

import setproctitle
setproctitle.setproctitle("Tran Chi Tam - Phase 2 Training")


class Phase2Trainer:
    """
    Trainer for Phase 2: Supervised anomaly detection with reconstruction
    """
    
    def __init__(
        self,
        augmentation: nn.Module,
        agf_tcn: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_scheduler: bool = True,
        use_grad_clip: bool = True,
        max_grad_norm: float = 1.0,
        config: dict = None  # Store config for inference
    ):
        self.augmentation = augmentation.to(device)
        self.agf_tcn = agf_tcn.to(device)
        self.device = device
        self.use_scheduler = use_scheduler
        self.use_grad_clip = use_grad_clip
        self.max_grad_norm = max_grad_norm
        self.config = config or {}  # Store for saving to checkpoint
        
        # Freeze augmentation module
        self._freeze_augmentation()
        
        # Only optimize AGF-TCN parameters
        self.optimizer = optim.Adam(
            self.agf_tcn.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler (optional)
        self.scheduler = None
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        
        # Reconstruction loss
        self.criterion = nn.MSELoss()
        
        print(f"\nPhase 2 Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Augmentation: FROZEN")
        print(f"  AGF-TCN parameters: {sum(p.numel() for p in self.agf_tcn.parameters()):,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in self.agf_tcn.parameters() if p.requires_grad):,}")
    
    def _freeze_augmentation(self):
        """Freeze all parameters in augmentation module"""
        for param in self.augmentation.parameters():
            param.requires_grad = False
        self.augmentation.eval()  # Set to eval mode
        print("  ✓ Augmentation module frozen")
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.agf_tcn.train()
        self.augmentation.eval()  # Keep augmentation in eval mode
        
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (batch_data, _) in enumerate(pbar):  # Labels not used in reconstruction training
            batch_data = batch_data.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                # Get augmented data (frozen augmentation)
                augmented_data = self.augmentation(batch_data)
            
            # Reconstruct from augmented data
            reconstructed = self.agf_tcn(augmented_data)
            
            # Compute reconstruction loss
            loss = self.criterion(reconstructed, augmented_data)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (optional)
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.agf_tcn.parameters(),
                    max_norm=self.max_grad_norm
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.6f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        
        return {
            'train_loss': avg_loss
        }
    
    def save_checkpoint(self, path, epoch, metrics):
        """Save model checkpoint with full config for inference"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'agf_tcn_state_dict': self.agf_tcn.state_dict(),
            'augmentation_state_dict': self.augmentation.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,  # Save full config for inference
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"  ✓ Checkpoint saved with config: {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.agf_tcn.load_state_dict(checkpoint['agf_tcn_state_dict'])
        self.augmentation.load_state_dict(checkpoint['augmentation_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"  Checkpoint loaded from {path}")
        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


def main():
    # Configuration
    config = {
        # Data config
        'dataset_name': 'ucr',
        'subset': '135',  # Specific dataset: 135, 136, 137, or 138
        'window_size': 16,
        'stride': 1,
        
        # Model config
        'agf_tcn_channels': [64, 64],  # TCN hidden channels
        'dropout': 0.1,
        'activation': 'gelu',
        'fuse_type': 5,  # TripConFusion
        
        # Augmentation Transformer hyperparameters (must match Phase 1)
        'transformer_d_model': 128,  # Transformer hidden dimension
        'transformer_nhead': 2,       # Number of attention heads (must be even)
        
        # Training config
        'batch_size': 64,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-6,
        'use_scheduler': False,  # Use learning rate scheduler
        'use_grad_clip': False,  # Use gradient clipping
        'max_grad_norm': 1.0,   # Max gradient norm for clipping
        
        # Phase 1 checkpoint (pre-trained augmentation)
        'phase1_checkpoint': 'phase1/checkpoints/best_model.pth',
        
        # Misc
        'num_workers': 0,
        'save_dir': 'phase2/checkpoints',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("=" * 60)
    print("Phase 2: Supervised Anomaly Detection Training")
    print("=" * 60)
    
    # Step 1: Prepare data
    print(f"\n[1/6] Loading dataset: {config['dataset_name']}_{config['subset']}")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path_base = os.path.join(project_root, 'data', 'datasets')
    
    # Dataloader function mapping
    dataloader_func = {
        'ucr': ucr_sub_ds_processing,
        'smd': smd_sub_ds_processing,
        'smap_msl': smap_msl_sub_ds_processing,
        'psm': psm_sub_ds_processing,
    }
    
    loader_func = dataloader_func[config['dataset_name']]
    
    # Load data (chỉ dùng train, test data cho inference)
    train_windows, train_labels, _, _ = prepare_phase2_data(
        dataset_name=config['dataset_name'],
        subset=config['subset'],
        loader_func=loader_func,
        window_size=config['window_size'],
        stride=config['stride'],
        data_path_base=data_path_base
    )
    
    # Step 2: Create train dataloader only
    print("\n[2/6] Creating train dataloader...")
    train_dataset = Phase2Dataset(train_windows, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True
    )
    print(f"  Train batches: {len(train_loader)}")
    
    # Get data dimensions
    n_channels = train_windows.shape[1]
    window_size = train_windows.shape[2]
    
    # Update config with actual dimensions (important for inference!)
    config['n_channels'] = n_channels
    config['window_size'] = window_size
    
    # Step 3: Initialize Augmentation (load from Phase 1)
    print("\n[3/6] Loading pre-trained Augmentation from Phase 1...")
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
    
    # Load Phase 1 checkpoint
    phase1_checkpoint_path = os.path.join(project_root, config['phase1_checkpoint'])
    if os.path.exists(phase1_checkpoint_path):
        checkpoint = torch.load(phase1_checkpoint_path, map_location=config['device'])
        augmentation.load_state_dict(checkpoint['augmentation_state_dict'])
        print(f"  ✓ Loaded augmentation from: {phase1_checkpoint_path}")
    else:
        print(f"  ⚠ WARNING: Phase 1 checkpoint not found at {phase1_checkpoint_path}")
        print(f"  ⚠ Using randomly initialized augmentation (not recommended!)")
    
    # Step 4: Initialize AGF-TCN
    print("\n[4/6] Initializing AGF-TCN...")
    agf_tcn = Agf_TCN(
        num_inputs=n_channels,
        num_channels=config['agf_tcn_channels'],
        dropout=config['dropout'],
        activation=config['activation'],
        fuse_type=config['fuse_type'],
        window_size=window_size
    )
    print(f"  AGF-TCN channels: {config['agf_tcn_channels']}")
    print(f"  Input shape: ({n_channels}, {window_size})")
    print(f"  Output shape: ({n_channels}, {window_size})")
    
    # Step 5: Initialize trainer
    print("\n[5/6] Initializing trainer...")
    trainer = Phase2Trainer(
        augmentation=augmentation,
        agf_tcn=agf_tcn,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        device=config['device'],
        use_scheduler=config['use_scheduler'],
        use_grad_clip=config['use_grad_clip'],
        max_grad_norm=config['max_grad_norm'],
        config=config  # Pass full config for checkpoint
    )
    
    # Step 6: Training loop
    print("\n[6/6] Starting training...")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Device: {config['device']}")
    
    best_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        # Print metrics
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_metrics['train_loss']:.6f}")
        
        # Learning rate scheduling (optional)
        if trainer.scheduler is not None:
            trainer.scheduler.step(train_metrics['train_loss'])
        
        # Save checkpoint when train_loss improves
        if train_metrics['train_loss'] < best_loss:
            best_loss = train_metrics['train_loss']
            best_epoch = epoch
            
            save_path = os.path.join(
                project_root,
                config['save_dir'],
                f"phase2_{config['dataset_name']}_{config['subset']}_best.pt"
            )
            trainer.save_checkpoint(save_path, epoch, train_metrics)
            print(f"  ✓ New best model! Train Loss: {best_loss:.6f}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best Train Loss: {best_loss:.6f} (Epoch {best_epoch})")
    print(f"Saved checkpoint: phase2/checkpoints/phase2_{config['dataset_name']}_{config['subset']}_best.pt")
    print("="*60)


if __name__ == '__main__':
    main()


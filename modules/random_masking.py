import torch
import torch.nn as nn


class RandomTimeMasking(nn.Module):
    """
    Random time masking module that masks a percentage of time steps.
    
    Args:
        mask_ratio (float): Percentage of time steps to mask (0.0 to 1.0). Default: 0.15
        mask_value (str): How to mask values. Options:
            - 'zero': Mask with zeros
            - 'mean': Mask with mean value across time dimension
            - 'noise': Mask with random noise
            Default: 'zero'
        mask_mode (str): How to apply masking. Options:
            - 'channel_wise': Each channel masked independently
            - 'temporal': All channels masked at same time steps
            Default: 'temporal'
    """
    
    def __init__(self, mask_ratio=0.15, mask_value='zero', mask_mode='temporal'):
        super(RandomTimeMasking, self).__init__()
        
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio must be between 0.0 and 1.0, got {mask_ratio}")
        
        if mask_value not in ['zero', 'mean', 'noise']:
            raise ValueError(f"mask_value must be one of ['zero', 'mean', 'noise'], got {mask_value}")
        
        if mask_mode not in ['channel_wise', 'temporal']:
            raise ValueError(f"mask_mode must be one of ['channel_wise', 'temporal'], got {mask_mode}")
        
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.mask_mode = mask_mode
    
    def forward(self, x):
        """
        Apply random time masking to input.
        
        Args:
            x: Input tensor with shape (batch, channels, seq_len)
        
        Returns:
            Masked tensor with same shape as input
        """
        if not self.training or self.mask_ratio == 0.0:
            # No masking during inference or if mask_ratio is 0
            return x
        
        batch_size, n_channels, seq_len = x.shape
        
        # Create mask
        if self.mask_mode == 'temporal':
            # All channels masked at same time steps
            n_mask = int(seq_len * self.mask_ratio)
            if n_mask > 0:
                # Randomly select time steps to mask
                mask_indices = torch.randperm(seq_len, device=x.device)[:n_mask]
                mask = torch.ones_like(x, dtype=torch.bool)
                mask[:, :, mask_indices] = False
            else:
                mask = torch.ones_like(x, dtype=torch.bool)
        else:  # channel_wise
            # Each channel masked independently
            n_mask = int(seq_len * self.mask_ratio)
            mask = torch.ones_like(x, dtype=torch.bool)
            if n_mask > 0:
                for ch in range(n_channels):
                    mask_indices = torch.randperm(seq_len, device=x.device)[:n_mask]
                    mask[:, ch, mask_indices] = False
        
        # Apply masking
        if self.mask_value == 'zero':
            masked_x = x * mask
        elif self.mask_value == 'mean':
            # Compute mean across time dimension for each channel
            mean_values = x.mean(dim=2, keepdim=True)  # (batch, channels, 1)
            masked_x = x * mask + mean_values * (~mask)
        else:  # noise
            # Mask with random noise (same distribution as input)
            noise = torch.randn_like(x) * x.std(dim=2, keepdim=True) + x.mean(dim=2, keepdim=True)
            masked_x = x * mask + noise * (~mask)
        
        return masked_x


class InferenceMasking(nn.Module):
    """
    Inference masking module with special logic:
    - Window 0 (first window): Random time masking (like training)
    - Window 1 to last: Only mask the last time-step
    
    Args:
        mask_ratio (float): Percentage of time steps to mask for window 0 (0.0 to 1.0). Default: 0.15
        mask_value (str): How to mask values. Options:
            - 'zero': Mask with zeros
            - 'mean': Mask with mean value across time dimension
            - 'noise': Mask with random noise
            Default: 'zero'
        mask_mode (str): How to apply masking for window 0. Options:
            - 'channel_wise': Each channel masked independently
            - 'temporal': All channels masked at same time steps
            Default: 'temporal'
    """
    
    def __init__(self, mask_ratio=0.15, mask_value='zero', mask_mode='temporal'):
        super(InferenceMasking, self).__init__()
        
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio must be between 0.0 and 1.0, got {mask_ratio}")
        
        if mask_value not in ['zero', 'mean', 'noise']:
            raise ValueError(f"mask_value must be one of ['zero', 'mean', 'noise'], got {mask_value}")
        
        if mask_mode not in ['channel_wise', 'temporal']:
            raise ValueError(f"mask_mode must be one of ['channel_wise', 'temporal'], got {mask_mode}")
        
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.mask_mode = mask_mode
        self.window_index = 0  # Track current window index
    
    def reset(self):
        """Reset window index to 0"""
        self.window_index = 0
    
    def forward(self, x, window_idx=None):
        """
        Apply inference masking to input.
        
        Args:
            x: Input tensor with shape (batch, channels, seq_len)
            window_idx: Optional window index. If None, uses internal counter.
        
        Returns:
            Masked tensor with same shape as input
        """
        if self.mask_ratio == 0.0:
            return x
        
        batch_size, n_channels, seq_len = x.shape
        
        # Determine window index
        if window_idx is not None:
            current_window_idx = window_idx
        else:
            current_window_idx = self.window_index
            self.window_index += batch_size
        
        # Create mask based on window index
        if current_window_idx == 0:
            # Window 0: Random time masking (like training)
            n_mask = int(seq_len * self.mask_ratio)
            if n_mask > 0:
                if self.mask_mode == 'temporal':
                    # All channels masked at same time steps
                    mask_indices = torch.randperm(seq_len, device=x.device)[:n_mask]
                    mask = torch.ones_like(x, dtype=torch.bool)
                    mask[:, :, mask_indices] = False
                else:  # channel_wise
                    # Each channel masked independently
                    mask = torch.ones_like(x, dtype=torch.bool)
                    for ch in range(n_channels):
                        mask_indices = torch.randperm(seq_len, device=x.device)[:n_mask]
                        mask[:, ch, mask_indices] = False
            else:
                mask = torch.ones_like(x, dtype=torch.bool)
        else:
            # Window 1 to last: Only mask the last time-step
            mask = torch.ones_like(x, dtype=torch.bool)
            mask[:, :, -1] = False  # Mask last time-step for all channels
        
        # Apply masking
        if self.mask_value == 'zero':
            masked_x = x * mask
        elif self.mask_value == 'mean':
            # Compute mean across time dimension for each channel
            mean_values = x.mean(dim=2, keepdim=True)  # (batch, channels, 1)
            masked_x = x * mask + mean_values * (~mask)
        else:  # noise
            # Mask with random noise (same distribution as input)
            noise = torch.randn_like(x) * x.std(dim=2, keepdim=True) + x.mean(dim=2, keepdim=True)
            masked_x = x * mask + noise * (~mask)
        
        return masked_x


# Test code
if __name__ == "__main__":
    # Test với input shape: (batch=2, channels=3, seq_len=100)
    batch_size = 2
    n_channels = 3
    seq_len = 100
    
    x = torch.randn(batch_size, n_channels, seq_len)
    
    print("Testing RandomTimeMasking...")
    print(f"Input shape: {x.shape}")
    
    # Test với mask_ratio = 0.15, mask_value = 'zero', mask_mode = 'temporal'
    masking = RandomTimeMasking(
        mask_ratio=0.15,
        mask_value='zero',
        mask_mode='temporal'
    )
    masking.train()  # Set to training mode
    
    masked_x = masking(x)
    print(f"Output shape: {masked_x.shape}")
    print(f"Mask ratio: {masking.mask_ratio}")
    print(f"Mask value: {masking.mask_value}")
    print(f"Mask mode: {masking.mask_mode}")
    
    # Check số lượng time steps bị mask
    n_masked = (masked_x == 0).sum(dim=1).sum(dim=1)  # Count zeros per batch
    print(f"Number of masked time steps per batch: {n_masked}")
    
    # Test với mask_value = 'mean'
    masking_mean = RandomTimeMasking(
        mask_ratio=0.2,
        mask_value='mean',
        mask_mode='temporal'
    )
    masking_mean.train()
    masked_x_mean = masking_mean(x)
    print(f"\nTest with mask_value='mean':")
    print(f"Output shape: {masked_x_mean.shape}")
    
    # Test với mask_mode = 'channel_wise'
    masking_channel = RandomTimeMasking(
        mask_ratio=0.15,
        mask_value='zero',
        mask_mode='channel_wise'
    )
    masking_channel.train()
    masked_x_channel = masking_channel(x)
    print(f"\nTest with mask_mode='channel_wise':")
    print(f"Output shape: {masked_x_channel.shape}")
    
    # Test inference mode (no masking)
    masking.eval()
    masked_x_eval = masking(x)
    print(f"\nTest in eval mode (no masking):")
    print(f"Are input and output equal? {torch.equal(x, masked_x_eval)}")


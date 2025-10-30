import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """
    MLP encoder for learning representations in Phase 1 contrastive learning.
    
    Input: (batch, n_channels, window_size)
    Output: (batch, projection_dim)
    """
    
    def __init__(
        self,
        input_channels: int,
        window_size: int,
        hidden_dims: list = [512, 512, 256],
        projection_dim: int = 128,
        dropout: float = 0.1
    ):
        super(MLPEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.window_size = window_size
        self.input_dim = input_channels * window_size
        
        # Build MLP layers
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Projection head
        layers.append(nn.Linear(prev_dim, projection_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, n_channels, window_size)
        
        Returns:
            Encoded representation of shape (batch, projection_dim)
        """
        # Flatten the input
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # (batch, n_channels * window_size)
        
        # Pass through encoder
        z = self.encoder(x)  # (batch, projection_dim)
        
        # L2 normalize for better contrastive learning
        z = nn.functional.normalize(z, dim=1)
        
        return z


class CNNEncoder(nn.Module):
    """
    CNN-based encoder for learning representations.
    Alternative to MLP encoder with better temporal feature extraction.
    
    Input: (batch, n_channels, window_size)
    Output: (batch, projection_dim)
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: list = [64, 128, 256],
        kernel_size: int = 3,
        projection_dim: int = 128,
        dropout: float = 0.1
    ):
        super(CNNEncoder, self).__init__()
        
        # Build CNN layers
        cnn_layers = []
        in_ch = input_channels
        
        for out_ch in hidden_channels:
            cnn_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            in_ch = out_ch
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels[-1], projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, n_channels, window_size)
        
        Returns:
            Encoded representation of shape (batch, projection_dim)
        """
        # Pass through CNN
        x = self.cnn(x)  # (batch, hidden_channels[-1], reduced_length)
        
        # Global average pooling
        x = self.gap(x).squeeze(-1)  # (batch, hidden_channels[-1])
        
        # Projection
        z = self.projection(x)  # (batch, projection_dim)
        
        # L2 normalize
        z = nn.functional.normalize(z, dim=1)
        
        return z



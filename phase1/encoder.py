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
        dropout: float = 0.1,
    ):
        super(MLPEncoder, self).__init__()

        self.input_channels = input_channels
        self.window_size = window_size
        self.input_dim = input_channels * window_size

        # Build MLP layers
        layers = []
        prev_dim = self.input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
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

        # Flatten the input
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # (batch, n_channels * window_size)
        # Pass through encoder
        z = self.encoder(x)  # (batch, projection_dim)
        # L2 normalize for better contrastive learning
        z = nn.functional.normalize(z, dim=1)

        return z

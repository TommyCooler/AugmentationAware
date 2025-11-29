import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from pytorch_tcn import TCN


class LinearAugmentation(nn.Module):

    def __init__(self, input_dim):
        super(LinearAugmentation, self).__init__()

        self.weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(input_dim))
        nn.init.normal_(self.weights, mean=0.0, std=0.001)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = torch.matmul(self.weights, x)
        x = x + self.bias.unsqueeze(0).unsqueeze(2)
        return x


class MLPaugmentation(nn.Module):
    def __init__(self, in_channels, seq_len, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len

        in_feats = in_channels * seq_len

        self.net = nn.Sequential(
            nn.Linear(in_feats, in_feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        B, C, _ = x.shape

        x = x.view(B, -1)  # [B, C*S]
        x = self.net(x)  # [B, C*S]
        x = x.view(B, C, self.seq_len)  # [B, C, S]
        return x


class CNN1DAugmentation(nn.Module):
    def __init__(self, in_channels, kernel_size, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding=padding)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, T]
        x = self.conv(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class CNN1DCausalAugmentation(nn.Module):
    def __init__(self, in_channels, kernel_size, dropout=0.1):
        super().__init__()
        padding = 0
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding=padding)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, T]
        x = F.pad(x, (self.left_padding, 0))
        x = self.conv(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


# class TCNAugmentation(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         kernel_size: int = 3,
#         dropout: float = 0.1,
#         causal: bool = True,
#     ):

#         super().__init__()
#         self.in_channels = in_channels

#         # 1 residual block, out_channels = in_channels → giữ nguyên số kênh
#         num_channels = [in_channels]

#         self.tcn = TCN(
#             num_inputs=in_channels,
#             num_channels=num_channels,
#             kernel_size=kernel_size,
#             dropout=dropout,
#             causal=causal,
#             use_norm="None",
#             activation="relu",
#             use_skip_connections=False,
#             input_shape="NCL",
#         )

#     def forward(self, x):
#         if x.dim() == 2:
#             x = x.unsqueeze(1)
#         return self.tcn(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, mean=0.0, std=0.02)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerEncoderAugmentation(nn.Module):

    def __init__(
        self,
        in_channels,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=5000,
    ):

        super(TransformerEncoderAugmentation, self).__init__()

        self.in_channels = in_channels
        self.d_model = d_model

        self.input_projection = nn.Linear(in_channels, d_model)

        self.pos_encoder = LearnablePositionalEncoding(
            d_model=d_model, max_len=max_seq_len, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.output_projection = nn.Linear(d_model, in_channels)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.constant_(self.input_projection.bias, 0.0)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.constant_(self.output_projection.bias, 0.0)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(
            x, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        x = self.output_projection(x)
        x = x.transpose(1, 2)
        return x


class Augmentation(nn.Module):
    """
    Multi-strategy augmentation module with Gumbel-Softmax selection.

    Args:
        in_channels (int): Number of input channels
        seq_len (int): Sequence length (time steps)
        out_channels (int, optional): Number of output channels. Defaults to in_channels.
        kernel_size (int): Kernel size for CNN/TCN. Default: 3
        num_layers (int): Number of layers for MLP/TCN/Transformer. Default: 2
        dropout (float): Dropout rate. Default: 0.1
        temperature (float): Gumbel-Softmax temperature. Default: 1.0
        hard (bool): Use hard Gumbel-Softmax. Default: False
        transformer_d_model (int): Transformer hidden dimension. Default: 512
        transformer_nhead (int): Number of attention heads (must be even). Default: 8

    Note:
        - transformer_nhead must be even to avoid nested tensor warnings
        - transformer_d_model must be divisible by transformer_nhead
    """

    def __init__(
        self,
        in_channels,
        seq_len,
        kernel_size=3,
        num_layers=2,
        dropout=0.1,
        temperature=1.0,
        hard=False,
        transformer_d_model=512,
        transformer_nhead=8,
    ):
        super(Augmentation, self).__init__()

        # Validate transformer hyperparameters
        if transformer_nhead % 2 != 0:
            raise ValueError(f"transformer_nhead must be even, got {transformer_nhead}")
        if transformer_d_model % transformer_nhead != 0:
            raise ValueError(
                f"transformer_d_model ({transformer_d_model}) must be divisible by transformer_nhead ({transformer_nhead})"
            )

        self.temperature = temperature
        self.hard = hard

        # Initialize augmentation modules
        self.linear = LinearAugmentation(input_dim=in_channels)

        self.mlp = MLPaugmentation(
            in_channels=in_channels,
            seq_len=seq_len,
            dropout=dropout,
        )

        self.cnn = CNN1DAugmentation(
            in_channels=in_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.cnn_causal = CNN1DCausalAugmentation(
            in_channels=in_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # self.tcn = TCNAugmentation(
        #     in_channels=in_channels,
        #     kernel_size=kernel_size,
        #     dropout=dropout,
        # )

        # Transformer with configurable hyperparameters
        self.transformer = TransformerEncoderAugmentation(
            in_channels=in_channels,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=num_layers,
            dim_feedforward=transformer_d_model * 4,
            dropout=dropout,
            max_seq_len=seq_len,
        )

        self.alpha = nn.Parameter(torch.ones(5) / 5.0)

    def forward(self, x):
        linear_out = self.linear(x)
        mlp_out = self.mlp(x)
        cnn_out = self.cnn(x)
        cnn_causal_out = self.cnn_causal(x)
        transformer_out = self.transformer(x)

        outputs = torch.stack(
            [linear_out, mlp_out, cnn_out, cnn_causal_out, transformer_out], dim=0
        )

        if self.training:
            probs = F.gumbel_softmax(
                self.alpha, tau=self.temperature, hard=self.hard, dim=0
            )
        else:
            probs = F.softmax(self.alpha / self.temperature, dim=0)

        weighted = outputs * probs.view(-1, 1, 1, 1)
        combined_output = weighted.sum(dim=0)

        return combined_output


# Test code
if __name__ == "__main__":
    # Test với input shape: (batch=2, channels=64, seq_len=100)
    batch_size = 2
    in_channels = 64
    seq_len = 100

    x = torch.randn(batch_size, in_channels, seq_len)

    model = Augmentation(
        in_channels=in_channels,
        seq_len=seq_len,
        out_channels=64,
        kernel_size=3,
        num_layers=2,
        dropout=0.1,
    )

    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Alpha weights: {F.softmax(model.alpha, dim=0)}")

import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x2, x2, x2)[0])
        x2 = self.norm2(x)
        x2 = self.linear2(self.dropout(F.gelu((self.linear1(x2)))))
        x = x + self.dropout2(x2)
        return x


class ResNetBlock(nn.Module):
    # Define a basic ResNet block with two convolutional layers and a residual connection
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual
        return self.relu(out)

class UpsampleTransformerDecoderV3(nn.Module):
    def __init__(self, input_channels, num_upsamples, num_blocks, num_heads, ff_dim, output_channels):
        super().__init__()
        self.up_blocks = nn.ModuleList()
        self.transformer_blocks = nn.ModuleList()

        for i in range(num_upsamples):
            out_channels = input_channels // 2

            # Adjust the out_channels to be divisible by num_heads
            if out_channels % num_heads != 0:
                out_channels += num_heads - (out_channels % num_heads)

            # Ensure that out_channels aligns with the next layer's expectations
            # Adjust the following layers accordingly

            self.up_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1),
                    ResNetBlock(out_channels),
                    GELU()
                )
            )

            input_channels = out_channels  # Update input_channels for the next loop iteration

            # Make sure the transformer block aligns with the updated input_channels
            transformer_blocks = nn.ModuleList()
            for _ in range(num_blocks):
                transformer_blocks.append(TransformerDecoderBlock(input_channels, num_heads, ff_dim))
            self.transformer_blocks.append(transformer_blocks)

        # The final_conv layer must align with the final output of the last transformer block
        self.final_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        for up_block, transformer_block_list in zip(self.up_blocks, self.transformer_blocks):
            x = up_block(x)  # Upsample
            B, C, H, W = x.shape
            x = x.view(B, C, H * W).permute(0, 2, 1)  # Reshape for transformer block
            for transformer_block in transformer_block_list:
                x = transformer_block(x)
            x = x.permute(0, 2, 1).view(B, C, H, W)  # Reshape back to image

        return self.final_conv(x)
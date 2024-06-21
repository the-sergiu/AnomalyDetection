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

class UpsampleTransformerDecoder(nn.Module):
    def __init__(self, input_channels, num_upsamples, num_blocks, num_heads, ff_dim, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.num_upsamples = num_upsamples
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.output_channels = output_channels
        self.up_blocks = nn.ModuleList()
        self.transformer_blocks = nn.ModuleList()

        # Ensure that the initial input_channels is divisible by 2^num_upsamples
        assert input_channels % (2 ** num_upsamples) == 0, "input_channels must be divisible by 2^num_upsamples"

        for _ in range(num_upsamples):
            out_channels = input_channels // 2

            # Ensure the resulting channels are divisible by num_heads for the transformer
            if out_channels % num_heads != 0:
                out_channels += num_heads - (out_channels % num_heads)
            
            self.up_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # as putea pune Nearest neigh in loc de bilinear
                    nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1),
                    # Putem adauga aici resnet block
                    GELU() # Punem leaky relu in loc de GELU
                )
            )
            input_channels = out_channels

            transformer_blocks = nn.ModuleList()
            for _ in range(num_blocks):
                transformer_blocks.append(TransformerDecoderBlock(input_channels, num_heads, ff_dim))
            self.transformer_blocks.append(transformer_blocks)

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
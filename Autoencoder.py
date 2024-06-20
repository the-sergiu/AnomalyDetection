import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model

from models.modules.mobileone import reparameterize_model
from UpsampleTransformerDecoder import UpsampleTransformerDecoder

sys.path.append("./ml-fastvit")



class TransformerAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(TransformerAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x[-1])
        return x


class CNNEncoder(nn.Module):
    """
    Example Usage:
    batch_size = 8  # Example batch size
    example_input = torch.randn(batch_size, 3, 64, 64)
    encoder_teacher = CNNEncoder()
    latent_representation = encoder_teacher(example_input)
    """

    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            # Input: (batch_size, 3, 64, 64)
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1
            ),  # (batch_size, 32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1
            ),  # (batch_size, 64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1
            ),  # (batch_size, 128, 8, 8)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1
            ),  # (batch_size, 256, 4, 4)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1
            ),  # (batch_size, 512, 2, 2)
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return x


class CNNDecoder(nn.Module):
    """
    Example Usage:
    batch_size = 8  # Example batch size
    example_input = torch.randn(batch_size, 3, 64, 64)
    latent_representation = CNNEncoder(example_input) # Should output torch.Size([8, 512, 2, 2])
    decoder_teacher = CNNDecoder()
    reconstructed_images = decoder_teacher(latent_representation)
    """

    def __init__(self):
        super(CNNDecoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(
            512, 256, kernel_size=4, stride=2, padding=1
        )  # output shape: (batch_size, 256, 4, 4)
        self.deconv2 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1
        )  # output shape: (batch_size, 128, 8, 8)
        self.deconv3 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )  # output shape: (batch_size, 64, 16, 16)
        self.deconv4 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )  # output shape: (batch_size, 32, 32, 32)
        self.deconv5 = nn.ConvTranspose2d(
            32, 3, kernel_size=4, stride=2, padding=1
        )  # output shape: (batch_size, 3, 64, 64)

        # Batch normalization layers
        self.batchnorm1 = nn.BatchNorm2d(256)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.batchnorm4 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.deconv1(x)))
        x = F.relu(self.batchnorm2(self.deconv2(x)))
        x = F.relu(self.batchnorm3(self.deconv3(x)))
        x = F.relu(self.batchnorm4(self.deconv4(x)))
        x = torch.sigmoid(self.deconv5(x))
        return x


class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()
        self.encoder = CNNEncoder()
        self.decoder = CNNDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AvenueAutoencoder(nn.Module):
    def __init__(
        self,
        path=None,
        input_channels=512,
        num_upsamples=5,
        num_blocks=2,
        num_heads=8,
        ff_dim=1024,
        output_channels=3,
    ):
        super(AvenueAutoencoder, self).__init__()
        self.encoder = create_model(
            "fastvit_t12", fork_feat=True
        )  # can turn fork_feat to False
        self.encoder = reparameterize_model(self.encoder)
        self.decoder = UpsampleTransformerDecoder(
            input_channels=input_channels,
            num_upsamples=num_upsamples,  # Adjusted to 5 upsampling steps
            num_blocks=num_blocks,
            num_heads=num_heads,
            ff_dim=ff_dim,
            output_channels=output_channels,
            # use_positional_encoding=True
        )

        if path:
            self.encoder.load_state_dict(torch.load(path)["encoder_state_dict"])
            self.decoder.load_state_dict(torch.load(path)["decoder_state_dict"])

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x[-1])
        return x

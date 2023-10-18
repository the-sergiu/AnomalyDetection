import matplotlib.pyplot as plt
import numpy as np
import torch

from mae import (
    mae_vit_base_patch16,
    mae_vit_large_patch16,
    mae_vit_huge_patch14,
    MaskedAutoencoderViT,
)

from typing import Any

imagenet_mean = torch.Tensor([0.485, 0.456, 0.406])
imagenet_std = torch.Tensor([0.229, 0.224, 0.225])


def load_mae(
    chkpt_dir: str, arch: MaskedAutoencoderViT = mae_vit_large_patch16, device="cpu"
) -> None:
    """Load the MaskedAutoEncoder for Vision Transformer from Meta pre-trained model"""
    # Load Architecture
    model = arch().to(device)

    # Load model into memory
    checkpoint = torch.load(chkpt_dir, map_location=device)
    msg = model.load_state_dict(checkpoint["model"], strict=True)
    print(msg)

    return model


def to_nchw(img: torch.Tensor) -> torch.Tensor:
    """Convert image tensor from NHWC to NCHW format."""
    return torch.einsum("nhwc->nchw", img)


def to_nhwc(img: torch.Tensor) -> torch.Tensor:
    """Convert image tensor from NCHW to NHWC format."""
    return torch.einsum("nchw->nhwc", img)


def apply_mask(
    image: torch.Tensor, mask: torch.Tensor, recon: torch.Tensor
) -> torch.Tensor:
    """Apply the mask to the image and combine with the reconstruction."""
    im_masked = image * (1 - mask)
    im_paste = im_masked + recon * mask
    return im_masked, im_paste


def process_image(img: np.array, model: Any, device: str):
    """
    Process the input image using the Masked Autoencoder (MAE) architecture on the GPU.

    Returns:
        tuple: Original image, masked image, reconstructed image, and combined original and reconstructed image.
    """
    # Convert numpy array to tensor and move to device.
    x = torch.tensor(img).to(device).float().unsqueeze(dim=0)

    # Convert the image from NHWC format to NCHW format which is commonly used with convolutional neural networks.
    x = to_nchw(x)

    # Forward pass through the MAE model.
    loss, y, mask = model(x, mask_ratio=0.75)

    # Convert patches back to the image format and convert layout to NHWC.
    y = model.unpatchify(y)
    y = to_nhwc(y).detach()

    # Reshape the mask tensor and unpatchify it to match image dimensions.
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)
    mask = to_nhwc(model.unpatchify(mask)).detach()

    # Convert the original image tensor to NHWC layout.
    x = to_nhwc(x)

    # Apply the mask to the original image and combine it with the MAE reconstruction.
    im_masked, im_paste = apply_mask(x, mask, y)

    return x[0], im_masked[0], y[0], im_paste[0]


def show_image(image, title=""):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis("off")
    return


def visualize_images(
    original: torch.Tensor,
    masked: torch.Tensor,
    reconstruction: torch.Tensor,
    reconstructed_visible: torch.Tensor,
) -> None:
    # make the plt figure larger
    plt.rcParams["figure.figsize"] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(original.cpu(), "original")

    plt.subplot(1, 4, 2)
    show_image(masked.cpu(), "masked")

    plt.subplot(1, 4, 3)
    show_image(reconstruction.cpu(), "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(reconstructed_visible.cpu(), "reconstruction + visible")

    plt.show()

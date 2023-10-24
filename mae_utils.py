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


def process_image(img: np.array, model: Any, device: str):
    x = torch.tensor(img).to(device).float()

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum("nhwc->nchw", x)

    # run MAE
    loss, y, mask = model(x, mask_ratio=0.75)
    y = model.unpatchify(y)
    y = torch.einsum("nchw->nhwc", y).detach()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(
        1, 1, model.patch_embed.patch_size[0] ** 2 * 3
    )  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum("nchw->nhwc", mask).detach()

    x = torch.einsum("nchw->nhwc", x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

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

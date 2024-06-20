import gc
import sys
from urllib.request import urlopen

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torchvision.transforms as T
import torchvision.utils as vutils
import wandb
from sklearn.cluster import KMeans
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T

from Autoencoder import AvenueAutoencoder

sys.path.append("./ml-fastvit")

# Before starting the training, make sure to clear any residual memory
gc.collect()
torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE USED: {device}")

# This should be adjusted accordingly based on dataset used
# The main datasets used were: Avenue, ShanghaiTech & UBNormal
DATASET_NAME = "AVENUE"
DATASET_TRAIN_PATH = "./datasets/Avenue Dataset/objects/train/"
DATASET_TEST_PATH = "./datasets/Avenue Dataset/objects/test/"

# DATASET_NAME = "SHANGHAITECH"
# DATASET_TRAIN_PATH = './datasets/shanghaitech/objects/train'
# DATASET_TEST_PATH = './datasets/shanghaitech/objects/test'

# DATASET_NAME = "UBNORMAL"
# DATASET_TRAIN_PATH = './datasets/UBNormal/train_normal_objects'


TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
NUM_EPOCHS = 1

# Declare Train & Test Dataloader
transform = T.Compose([T.Resize((64, 64)), T.ToTensor(), T.Normalize(0.5, 0.5)])
train_dataset = datasets.ImageFolder(root=DATASET_TRAIN_PATH, transform=transform)
train_loader = DataLoader(
    train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2
)
print(train_dataset)

# Declare Model, Loss & Optimizer
autoencoder = AvenueAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(autoencoder.parameters(), lr=0.001)

# Training loop
losses = []
smallest_loss = 1


def train_loop(wandb_run: bool = False) -> None:
    if wandb_run:
        # start a new wandb run to track this script
        run = wandb.init(
            # set the wandb project where this run will be logged
            project="Anomaly Detection",
            # track hyperparameters and run metadata
            config={
                "encoder": "T12",
                "decoder": f"{autoencoder.decoder.__class__}",
                "dataset": DATASET_NAME,
                "predictions": "YoloV8_2",
                "epochs": f"{NUM_EPOCHS}",
                "input_channels": f"{autoencoder.decoder.input_channels}",
                "num_upsamples": f"{autoencoder.decoder.num_upsamples}",
                "num_blocks": f"{autoencoder.decoder.num_blocks}",
                "num_heads": f"{autoencoder.decoder.num_heads}",
                "ff_dim": f"{autoencoder.decoder.ff_dim}",
                "output_channels": f"{autoencoder.decoder.output_channels}",
            },
        )

    print("Traning started....")
    smallest_loss = 1

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for inputs, _ in train_loader:  # Note the unpacking here to ignore the labels
            inputs = inputs.to(device)  # Move inputs to the device

            # Zero the parameter gradients
            optimizer.zero_grad()
            reconstructed_imgs = autoencoder(inputs)

            # Calculate loss
            loss = criterion(reconstructed_imgs, inputs)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            losses.append(loss.item())

        if running_loss / len(train_loader) < smallest_loss:
            smallest_loss = running_loss / len(train_loader)
            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "encoder_state_dict": autoencoder.encoder.state_dict(),
                "decoder_state_dict": autoencoder.decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": running_loss / len(train_loader),
            }
            torch.save(
                checkpoint,
                f"autoencoder_t12_decoder_yolov8_2_adamw_numheads_{autoencoder.decoder.num_heads}_ffdim_{autoencoder.decoder.ff_dim}_numblocks_{autoencoder.decoder.num_blocks}_epoch_{epoch+1}_mseloss_{running_loss / len(train_loader)}.pth",
            )

        if wandb_run:
            run.log({"Loss": loss, "epoch": epoch + 1})
        
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")
        running_loss = 0.0

    print("Finished Training")

    if wandb_run:
        run.finish()


def main():
    train_loop()


if __name__ == "__main__":
    main()

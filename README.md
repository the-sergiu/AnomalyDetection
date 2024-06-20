# Anomaly Detection
TODO: Contents


# Project Setup
## 1. Locally installing dependencies

### 1.1. Create Virtual Environment within Linux (or WSL) using these commands (more or less, depending on what you already have installed)
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.10
sudo apt install python3-pip python3.10-venv

python3 -m venv /path/to/new/virtual/environment
python3.10 -m venv myenv
source myenv/bin/activate
```

### 1.2. Also make sure to install other potentially relevant packages after activating current environment.
```py
pip3 install wheel jupyter
pip3 install --upgrade pip setuptools wheel
```

### 1.3. Now's the shaky part - we want to install a very specific Cuda Toolkit version, with a very specific PyTorch version. This can cause headaches - and a lot of frustration. We want to install cuda toolkit 11.6 with PyTorch version 1.13. As such, assuming we are using Ubuntu 20.04, we'll want to run something along the lines of:
```bash
# CUDA installation on Ubuntu - https://gist.github.com/ksopyla/bf74e8ce2683460d8de6e0dc389fc7f5
#############
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

# new key, added 2022-04-25 22:52
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
############

# The more important commands
sudo apt update
sudo apt install cuda-toolkit-11-6

# PyTorch installation with CUDA 11.6
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

New versions of cuda toolkit should work, but there might be issues when installing specific PyTorch versions.

## 2. "Installing" FastViT (Apple Research)
Clone the `ml-fastvit` repo into the root of this repo:
```
git clone git@github.com:apple/ml-fastvit.git
```
We will leverage the code from here later by appending this folder to our Python packages path.




# TODO: Datasets
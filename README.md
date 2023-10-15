# Anomaly Detection


## Setup

### Recognize Anything Model ( + Grounded Segment Anything)

- Create Virtual Environment within Linux (or WSL) using these commands (more or less, depending on what you already have installed)
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

- Also make sure to install other potentially relevant packages after activating current environment.
```py
pip3 install wheel jupyter
pip3 install --upgrade pip setuptools wheel
```

- Now's the shaky part - we want to install a very specific Cuda Toolkit version, with a very specific PyTorch version. This can cause headaches - and a lot of frustration. According to the creators of the original repo, they run cuda 11.6 with PyTorch version 1.13. As such, we'll want to run something along the lines of:
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
- Download Weights and Models in relevant folders
```
download RAM and Tag2Text checkpoints to ./pretrained/ from https://github.com/majinyu666/recognize-anything/tree/main#toolbox-checkpoints

pretrained\ram_swin_large_14m.pth
pretrained\tag2text_swin_14m.pth

# download GroundingDINO and SAM checkpoints to ./Grounded-Segment-Anything/ from step 1 of https://github.com/IDEA-Research/Grounded-Segment

Grounded-Segment-Anything\sam_vit_h_4b8939.pth
Grounded-Segment-Anything\groundingdino_swint_ogc.pth

```

- Install actual modules using `requirements.txt` and `setup.py`
```
# Recognize Anything Reqs
pip install -r recognize-anything-requirements.txt
pip install -e .
pip install git+https://github.com/xinyu1205/recognize-anything.git

# Go into Grounded-SAM repo
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git

cd ./Grounded-Segment-Anything

pip install -r ./requirements.txt
pip install ./segment_anything
pip install ./GroundingDINO

cd ..

pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel
```

- We can now launch the notebook server
```
jupyter notebook --no-browser
```

- When developing, these commands will come in handy
```
wsl
cd ../../ # To go to /c/mnt in WSL
cd Users/Sergiu/Desktop/AnomalyDetection
source venv/bin/activate
jupyter notebook --no-browser
```
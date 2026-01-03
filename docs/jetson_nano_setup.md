# SmallVLA Setup on Jetson Nano

This guide covers setting up the SmallVLA model on your Jetson Nano for JetBot navigation.

## Prerequisites

- Jetson Nano with JetPack 4.6+ installed
- JetBot hardware assembled
- SSH access or monitor/keyboard connected
- Internet connection

## Step 1: System Preparation

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    libopenblas-base \
    libopenmpi-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpython3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev

# Increase swap (recommended for 4GB Nano)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
```

## Step 1b: Mount USB for Extra Storage (Recommended)

If you run low on disk space during installation, use a USB drive for build files:

```bash
# Plug in a USB drive (8GB+ recommended)
# Find the device name
lsblk

# Mount with proper permissions (usually /dev/sda1)
sudo mkdir -p /mnt/usb
sudo mount -o uid=$(id -u),gid=$(id -g) /dev/sda1 /mnt/usb

# Create directories for build cache
mkdir -p /mnt/usb/.cargo /mnt/usb/tmp

# Set environment variables (add to ~/.bashrc for persistence)
export CARGO_HOME=/mnt/usb/.cargo
export TMPDIR=/mnt/usb/tmp

# Optional: Move HuggingFace cache to USB for model downloads
export HF_HOME=/mnt/usb/.cache/huggingface
mkdir -p $HF_HOME
```

To make the USB mount persistent across reboots, add to `/etc/fstab`:
```bash
# Find USB UUID
sudo blkid /dev/sda1

# Add to fstab (replace UUID with your value)
echo "UUID=YOUR-UUID-HERE /mnt/usb vfat uid=1000,gid=1000,umask=022 0 0" | sudo tee -a /etc/fstab
```

## Step 2: Install PyTorch for Jetson

NVIDIA provides pre-built PyTorch wheels for Jetson. Do NOT use pip install torch directly.

```bash
# For JetPack 4.6 (L4T R32.6.1) - PyTorch 1.10
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# For JetPack 5.x (L4T R35.x) - PyTorch 2.0
# Check https://forums.developer.nvidia.com/t/pytorch-for-jetson/
wget https://developer.download.nvidia.cn/compute/redist/jp/v51/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
pip3 install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl

# Install torchvision (must match PyTorch version)
# For PyTorch 1.10:
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
python3 setup.py install --user
cd ..

# Verify installation
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Step 3: Clone JetBot Repository

```bash
# Clone the repository
cd ~
git clone https://github.com/YOUR_REPO/jetbot.git
cd jetbot

# Checkout the smallVLA branch
git checkout smallVLA
```

## Step 4: Install Python Dependencies

First, upgrade pip and install Rust (needed for tokenizers):

```bash
# Upgrade pip
pip3 install --user --upgrade pip

# Install Rust compiler (required for tokenizers)
sudo apt-get install -y rustc cargo libzmq3-dev python3-zmq

# If low on disk space, ensure USB is mounted (see Step 1b)
# export CARGO_HOME=/mnt/usb/.cargo
# export TMPDIR=/mnt/usb/tmp
```

Install Python packages (note: Python 3.6 on JetPack 4.x requires older versions):

```bash
# Install basic dependencies first
python3 -m pip install --user numpy Pillow pyyaml tqdm einops

# For JetPack 4.x (Python 3.6) - use older transformers
# Building tokenizers takes 30-60 minutes on Nano
python3 -m pip install --user transformers==4.12.0 accelerate==0.15.0

# For JetPack 5.x (Python 3.8+) - use newer versions
# python3 -m pip install --user transformers accelerate

# Install timm (specific version for compatibility)
python3 -m pip install --user "timm>=0.9.10,<1.0.0"

# Optional: Install opencv (may already be installed with JetPack)
python3 -m pip install --user opencv-python
```

## Step 5: Configure Memory Optimization

Create a script to optimize Jetson for inference:

```bash
# Create optimization script
cat > ~/jetbot_optimize.sh << 'EOF'
#!/bin/bash
# Maximize Jetson Nano performance for VLA inference

# Set to maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Disable GUI to free memory (optional, run headless)
# sudo systemctl stop gdm3

# Set GPU memory allocation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "Jetson optimized for SmallVLA inference"
EOF

chmod +x ~/jetbot_optimize.sh
```

## Step 6: Test SmallVLA

```bash
# Run optimization
~/jetbot_optimize.sh

# Navigate to jetbot directory
cd ~/jetbot

# Test SmolVLA loading (this will download the model first time)
python3 -c "
from server.vla_server.smolvla_wrapper import SmolVLAWrapper
print('Loading SmolVLA...')
vla = SmolVLAWrapper(device='cuda')
print('SmolVLA loaded successfully!')
print(vla.get_memory_usage())
"
```

## Step 7: Run VLA Server

```bash
# Start the VLA server
cd ~/jetbot
python3 -m server.vla_server.server \
    --model-type smolvla \
    --device cuda \
    --port 5555
```

The server will:
1. Load SmolVLA model (~1-2 minutes first time)
2. Run warmup inference
3. Listen on port 5555 for JetBot client connections

## Step 8: Run JetBot Client

In a separate terminal or on the JetBot:

```bash
cd ~/jetbot

# Test with VLA navigation app
python3 -m jetbot.apps.vla_navigation \
    --instruction "go forward" \
    --server-host localhost \
    --max-speed 0.3
```

## Running as a Service

To run the VLA server automatically on boot:

```bash
# Create systemd service
sudo tee /etc/systemd/system/vla-server.service << EOF
[Unit]
Description=SmallVLA Server for JetBot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/jetbot
Environment="PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512"
ExecStartPre=/usr/bin/nvpmodel -m 0
ExecStart=/usr/bin/python3 -m server.vla_server.server --model-type smolvla --device cuda
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable vla-server
sudo systemctl start vla-server

# Check status
sudo systemctl status vla-server
```

## Memory Tips

If you encounter out-of-memory errors:

1. **Close other applications** - Free up GPU memory
2. **Use CPU mode** - `--device cpu` (slower but uses system RAM)
3. **Reduce swap usage** - Increase swap file size
4. **Disable desktop** - Run headless to free ~500MB

```bash
# Check memory usage
free -h
nvidia-smi  # or tegrastats for Jetson
```

## Fine-tuning on Jetson

You can fine-tune the action head directly on the Jetson (slowly):

```bash
cd ~/jetbot

# Use small batch size and FP16
python3 -m server.vla_server.fine_tuning.train_smolvla \
    --data-dir dataset_vla_synthetic_large \
    --output-dir models/smolvla_jetbot \
    --epochs 10 \
    --batch-size 1 \
    --device cuda
```

For faster training, train on a desktop GPU and copy the model:

```bash
# On desktop: train and save
python -m server.vla_server.fine_tuning.train_smolvla ...

# Copy to Jetson
scp -r models/smolvla_jetbot jetson@<jetson-ip>:~/jetbot/models/
```

## Troubleshooting

### Model download fails
```bash
# Set HuggingFace cache directory with more space
export HF_HOME=/path/to/larger/drive/.cache/huggingface
```

### CUDA out of memory
```bash
# Use CPU fallback
python3 -m server.vla_server.server --model-type smolvla --device cpu
```

### Slow inference
```bash
# Ensure performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Check GPU utilization
tegrastats
```

### Import errors
```bash
# Reinstall transformers with no cache
pip3 install --user --no-cache-dir transformers
```

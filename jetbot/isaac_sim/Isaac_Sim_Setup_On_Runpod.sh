#!/bin/bash
# ============================================
# Isaac Sim Setup on RunPod - Complete Guide
# ============================================
#
# This script sets up Isaac Sim on RunPod for JetBot VLA data collection.
#
# RUNPOD CONFIGURATION:
# ---------------------
# 1. Create a new GPU Pod (RTX 4090 or A100 recommended)
# 2. In "Edit Pod" -> "Container Image", enter: nvcr.io/nvidia/isaac-sim:4.2.0
# 3. In "Environment Variables", add:
#    - ACCEPT_EULA=Y
#    - PRIVACY_CONSENT=Y
# 4. Start the pod and connect via SSH or Web Terminal
#
# NOTE: Isaac Sim is pre-installed in the container image.
# No need to pip install - just use /isaac-sim/python.sh

set -e  # Exit on error

echo "============================================"
echo "Isaac Sim Setup on RunPod"
echo "============================================"

# ============================================
# STEP 0: System packages (git, etc.)
# ============================================
echo ""
echo "Step 0: Installing system packages..."
apt-get update
apt-get install -y git curl wget

# ============================================
# STEP 1: Verify Isaac Sim installation
# ============================================
echo ""
echo "Step 1: Verifying Isaac Sim installation..."
echo "Isaac Sim location: /isaac-sim/"
echo "Python interpreter: /isaac-sim/python.sh"

# Check Isaac Sim version
/isaac-sim/python.sh -c "import isaacsim; print(f'Isaac Sim version: {isaacsim.__version__}')" 2>/dev/null || echo "Version check skipped"

# Quick verification that Isaac Sim can start
cd /workspace
/isaac-sim/python.sh -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
print('Isaac Sim initialized successfully!')
app.close()
"

# ============================================
# STEP 2: Clone JetBot repository
# ============================================
echo ""
echo "Step 2: Cloning JetBot repository..."
cd /workspace

# Remove existing clone if present
if [ -d "jetbot" ]; then
    echo "Removing existing jetbot directory..."
    rm -rf jetbot
fi

git clone https://github.com/shraavb/jetbot.git
cd jetbot
git checkout isaac-sim

echo "Repository cloned and checked out to isaac-sim branch"

# ============================================
# STEP 3: Set environment variables
# ============================================
echo ""
echo "Step 3: Setting environment variables..."
export PYTHONPATH=/workspace/jetbot:$PYTHONPATH

# Add to bashrc for persistence
echo 'export PYTHONPATH=/workspace/jetbot:$PYTHONPATH' >> ~/.bashrc

echo "PYTHONPATH set to: $PYTHONPATH"

# ============================================
# STEP 4: Install additional Python packages (if needed)
# ============================================
echo ""
echo "Step 4: Installing additional Python packages..."

# Isaac Sim has its own Python environment at /isaac-sim/python.sh
# Only install packages that are NOT already included

# Check if zmq is available (needed for VLA server communication)
/isaac-sim/python.sh -c "import zmq; print('zmq already installed')" 2>/dev/null || \
    /isaac-sim/python.sh -m pip install pyzmq

# Check if PIL is available (should be, but verify)
/isaac-sim/python.sh -c "from PIL import Image; print('PIL already installed')" 2>/dev/null || \
    /isaac-sim/python.sh -m pip install Pillow

echo "Python packages verified"

# ============================================
# STEP 5: Download JetBot USD asset
# ============================================
echo ""
echo "Step 5: Downloading JetBot USD asset..."

# NOTE: Run script directly, NOT as module (avoids jetbot hardware import errors)
# The jetbot package has hardware dependencies (traitlets, etc.) that are not
# available in Isaac Sim environment

/isaac-sim/python.sh /workspace/jetbot/jetbot/isaac_sim/runpod_setup.py --download-assets

echo "JetBot asset downloaded"

# ============================================
# STEP 6: Test simulation (optional quick test)
# ============================================
echo ""
echo "Step 6: Testing simulation..."
/isaac-sim/python.sh /workspace/jetbot/jetbot/isaac_sim/runpod_setup.py --test-sim

# ============================================
# STEP 7: Collect training data
# ============================================
echo ""
echo "Step 7: Collecting training data..."
echo ""
echo "NOTE: Use runpod_setup.py directly for data collection"
echo "Do NOT use collect_language_data.py as a module - it triggers jetbot hardware imports"
echo ""

# Small test run first (10 episodes = 500 samples)
echo "Running small test collection (10 episodes)..."
/isaac-sim/python.sh /workspace/jetbot/jetbot/isaac_sim/runpod_setup.py \
  --collect-data \
  --episodes 10 \
  --steps 50 \
  --output /workspace/dataset_vla_test \
  2>&1 | tee /workspace/collect_data_test.log

# Uncomment for larger dataset (200 episodes = 10,000 samples)
# echo "Running full data collection (200 episodes)..."
# /isaac-sim/python.sh /workspace/jetbot/jetbot/isaac_sim/runpod_setup.py \
#   --collect-data \
#   --episodes 200 \
#   --steps 50 \
#   --output /workspace/dataset_vla_real \
#   2>&1 | tee /workspace/collect_data.log

# ============================================
# STEP 8: Verify collected data
# ============================================
echo ""
echo "Step 8: Verifying collected data..."

OUTPUT_DIR="/workspace/dataset_vla_test"

echo "Checking output directory: $OUTPUT_DIR"
ls -la $OUTPUT_DIR/ 2>/dev/null | head -10 || echo "Directory not found"

echo ""
echo "File counts:"
JPG_COUNT=$(ls $OUTPUT_DIR/*.jpg 2>/dev/null | wc -l)
JSON_COUNT=$(ls $OUTPUT_DIR/*.json 2>/dev/null | wc -l)
echo "  JPG files: $JPG_COUNT"
echo "  JSON files: $JSON_COUNT"

echo ""
echo "Total size:"
du -sh $OUTPUT_DIR/ 2>/dev/null || echo "Cannot determine size"

echo ""
echo "Sample JSON:"
ls $OUTPUT_DIR/*.json 2>/dev/null | head -1 | xargs cat 2>/dev/null || echo "No JSON files found"

echo ""
echo "Sample image size (should be 5-20KB for real rendered images, not 150KB for noise):"
ls -la $OUTPUT_DIR/*.jpg 2>/dev/null | head -3

# ============================================
# COMPLETE
# ============================================
echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "QUICK REFERENCE:"
echo "----------------"
echo ""
echo "1. Run Python scripts:"
echo "   /isaac-sim/python.sh /path/to/script.py"
echo ""
echo "2. Collect more data:"
echo "   /isaac-sim/python.sh /workspace/jetbot/jetbot/isaac_sim/runpod_setup.py \\"
echo "     --collect-data --episodes 200 --steps 50 --output /workspace/dataset_vla_real"
echo ""
echo "3. Test VLA model (requires VLA server running elsewhere):"
echo "   /isaac-sim/python.sh /workspace/jetbot/jetbot/isaac_sim/runpod_setup.py \\"
echo "     --test-vla --vla-host <SERVER_IP> --vla-port 5555"
echo ""
echo "IMPORTANT NOTES:"
echo "----------------"
echo "1. Always run scripts DIRECTLY with /isaac-sim/python.sh /path/to/script.py"
echo "   Do NOT use -m module syntax (triggers jetbot hardware imports)"
echo ""
echo "2. Isaac Sim arguments like --/log/level=info are NOT supported"
echo "   These scripts use standard argparse"
echo ""
echo "3. Data format: Each sample has {uuid}.jpg (image) and {uuid}.json (metadata)"
echo "   JSON contains: instruction, action (left_speed, right_speed), episode, step, etc."
echo ""
echo "4. For training, transfer the dataset to your training machine and run:"
echo "   python -m server.vla_server.fine_tuning.train_smolvla \\"
echo "     --data-dir /path/to/dataset --epochs 20 --output-dir ./models/smolvla_jetbot"

# Isaac Sim Integration for JetBot VLA

This module provides NVIDIA Isaac Sim integration for VLA training and testing.

## Quick Start on RunPod

### 1. Create RunPod Instance
- Select RTX 4090 GPU ($0.29/hr Spot)
- Configure ports: 8888 (HTTP), 22 (TCP for SSH)
- Add 100GB storage

### 2. Install Isaac Sim
```bash
# Install via pip
pip install isaacsim[all] --extra-index-url https://pypi.nvidia.com

# Install system libraries
apt-get update && apt-get install -y \
    libvulkan1 vulkan-tools libxt6 libxmu6 \
    libxi6 libglu1-mesa libgl1-mesa-glx libegl1-mesa
```

### 3. Download JetBot Asset
```bash
mkdir -p /workspace/assets
wget -O /workspace/assets/jetbot.usd \
    "https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Isaac/4.0/Isaac/Robots/Jetbot/jetbot.usd"
```

### 4. Clone and Install JetBot
```bash
cd /workspace
git clone <your-repo> jetbot
cd jetbot && pip install -e .
```

### 5. Test Simulation
```bash
python jetbot/isaac_sim/runpod_setup.py --test-sim
```

## Running VLA with Simulation

### Option A: VLA Server on Same Machine
```bash
# Terminal 1: Start VLA server
python -m server.vla_server.server --port 5555

# Terminal 2: Run simulation
python jetbot/isaac_sim/runpod_setup.py --run-vla --instruction "go forward"
```

### Option B: VLA Server on Different Machine
```bash
# On VLA server (e.g., local Mac with MPS)
python -m server.vla_server.server --port 5555 --device mps

# On RunPod
python jetbot/isaac_sim/runpod_setup.py --run-vla \
    --vla-host <VLA_SERVER_IP> --vla-port 5555
```

## Requirements

1. **NVIDIA Isaac Sim** (version 4.5+)
   - Pip installation: `pip install isaacsim[all] --extra-index-url https://pypi.nvidia.com`
   - Or install from [NVIDIA Omniverse](https://developer.nvidia.com/isaac-sim)

2. **GPU with CUDA support**
   - RTX 3090/4090 or better recommended
   - 24GB+ VRAM for full VLA + simulation

## Usage

### Basic Simulation

```python
from jetbot.isaac_sim import JetBotSim

# Create and start simulation
sim = JetBotSim(headless=False)
sim.start(scene='simple_room')

# Control robot
sim.set_velocity(left=0.5, right=0.5)  # Move forward

# Get camera image
image = sim.get_camera_image()

# Step simulation
sim.step()

# Cleanup
sim.stop()
```

### VLA Testing in Simulation

```python
from jetbot.isaac_sim import VLASimInterface

# Create interface (requires VLA server running)
interface = VLASimInterface(
    vla_server_host='localhost',
    vla_server_port=5555,
    headless=True
)

interface.start()

# Run VLA-guided navigation
result = interface.run_vla_navigation(
    instruction="go forward and turn left"
)
print(f"Completed in {result['num_steps']} steps")

interface.stop()
```

### Collecting Synthetic Training Data

```python
from jetbot.isaac_sim import VLASimInterface

interface = VLASimInterface(headless=True)
interface.start()

# Collect training data
path = interface.collect_data(
    num_episodes=100,
    steps_per_episode=50,
    save_dir='./sim_data',
    policy='scripted'  # or 'random'
)

print(f"Data saved to: {path}")
interface.stop()
```

### Command Line Usage

```bash
# Collect data
python -m jetbot.isaac_sim.vla_sim_interface \
    --mode collect \
    --episodes 100 \
    --output ./sim_data \
    --headless

# Test VLA model
python -m jetbot.isaac_sim.vla_sim_interface \
    --mode test \
    --vla-host localhost \
    --vla-port 5555

# Run navigation
python -m jetbot.isaac_sim.vla_sim_interface \
    --mode navigate \
    --instruction "go to the red box" \
    --vla-host localhost
```

## API Reference

### JetBotSim

| Method | Description |
|--------|-------------|
| `start(scene)` | Start simulation with specified scene |
| `stop()` | Stop simulation |
| `step()` | Advance simulation by one timestep |
| `set_velocity(left, right)` | Set wheel speeds [-1, 1] |
| `get_camera_image()` | Get RGB camera image |
| `get_position()` | Get robot world position |
| `reset()` | Reset robot to initial state |

### VLASimInterface

| Method | Description |
|--------|-------------|
| `start()` | Start simulation and connect to VLA |
| `stop()` | Stop and cleanup |
| `run_vla_navigation(instruction)` | Run VLA-guided navigation |
| `collect_data(...)` | Collect synthetic training data |
| `test_vla_model(...)` | Test VLA model performance |

## Scenes

Available built-in scenes:
- `simple_room` - Basic room environment
- `warehouse` - Warehouse with shelves
- `office` - Office environment
- `grid` - Simple grid floor

Custom USD scenes can be loaded by path.

## Data Format

Collected data follows the JetBot VLA dataset format:

```
sim_data/
├── {uuid}.jpg      # 224x224 RGB image
├── {uuid}.json     # Metadata
└── ...
```

Metadata format:
```json
{
  "instruction": "go forward",
  "action": {
    "left_speed": 0.5,
    "right_speed": 0.5
  },
  "episode": 0,
  "step": 10,
  "source": "isaac_sim"
}
```

## Resources

- [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/)
- [Isaac Lab (RL Framework)](https://isaac-sim.github.io/IsaacLab/)
- [Wheeled Robots API](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/py/source/extensions/isaacsim.robot.wheeled_robots/docs/index.html)

#!/usr/bin/env python3
"""
RunPod Setup Script for JetBot VLA Simulation

Self-contained script that runs Isaac Sim without JetBot hardware dependencies.

Usage:
    python runpod_setup.py --download-assets
    python runpod_setup.py --test-sim
    python runpod_setup.py --collect-data --episodes 10 --output /workspace/sim_data
"""

import os
import sys
import json
import time
import uuid
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple


# Default navigation instructions for data collection
DEFAULT_INSTRUCTIONS = [
    "go forward",
    "move forward",
    "drive straight ahead",
    "turn left",
    "turn right",
    "rotate left",
    "rotate right",
    "go backward",
    "move back",
    "stop",
    "halt",
    "avoid the obstacle",
    "go around the obstacle",
    "approach the object",
    "move toward the target",
    "navigate to the goal",
    "follow the path",
    "explore the area",
    "turn around",
    "make a u-turn"
]


def download_jetbot_asset(output_dir: str = "/workspace/assets") -> str:
    """Download JetBot USD asset from NVIDIA."""
    import urllib.request

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    jetbot_url = "https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Isaac/4.0/Isaac/Robots/Jetbot/jetbot.usd"
    jetbot_path = output_path / "jetbot.usd"

    if jetbot_path.exists():
        print(f"JetBot asset already exists at {jetbot_path}")
        return str(jetbot_path)

    print(f"Downloading JetBot asset to {jetbot_path}...")
    urllib.request.urlretrieve(jetbot_url, jetbot_path)
    print("Download complete!")

    return str(jetbot_path)


def get_scripted_action(instruction: str) -> Tuple[float, float]:
    """Get action based on instruction keywords with noise."""
    instruction = instruction.lower()
    noise = np.random.normal(0, 0.1, 2)

    if 'forward' in instruction or 'straight' in instruction:
        action = (0.5, 0.5)
    elif 'left' in instruction:
        action = (0.2, 0.5)
    elif 'right' in instruction:
        action = (0.5, 0.2)
    elif 'backward' in instruction or 'back' in instruction:
        action = (-0.3, -0.3)
    elif 'stop' in instruction or 'halt' in instruction:
        action = (0.0, 0.0)
    elif 'avoid' in instruction or 'around' in instruction:
        if np.random.random() > 0.5:
            action = (0.3, 0.5)
        else:
            action = (0.5, 0.3)
    else:
        action = (0.3, 0.3)

    return (
        float(np.clip(action[0] + noise[0], -1.0, 1.0)),
        float(np.clip(action[1] + noise[1], -1.0, 1.0))
    )


def test_simulation() -> bool:
    """Test that Isaac Sim and JetBot work correctly."""
    print("Testing Isaac Sim with JetBot...")

    try:
        from isaacsim import SimulationApp
        simulation_app = SimulationApp({"headless": True})

        from omni.isaac.core import World
        from omni.isaac.wheeled_robots.robots import WheeledRobot
        from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController

        world = World()
        world.scene.add_default_ground_plane()

        jetbot_path = "/workspace/assets/jetbot.usd"
        if not os.path.exists(jetbot_path):
            jetbot_path = download_jetbot_asset()

        jetbot = world.scene.add(
            WheeledRobot(
                prim_path="/World/JetBot",
                name="jetbot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=jetbot_path,
            )
        )

        controller = DifferentialController(
            name="diff_controller",
            wheel_radius=0.0325,
            wheel_base=0.1
        )

        world.reset()

        # Run a few steps
        for i in range(20):
            wheel_velocities = controller.forward([0.3, 0.1])
            jetbot.apply_wheel_actions(wheel_velocities)
            world.step(render=False)
            if i % 5 == 0:
                pos, _ = jetbot.get_world_pose()
                print(f"Step {i}: Pos=[{pos[0]:.3f}, {pos[1]:.3f}]")

        print("Test PASSED! JetBot simulation working.")
        simulation_app.close()
        return True

    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def collect_synthetic_data(
    output_dir: str = "/workspace/sim_data",
    num_episodes: int = 10,
    steps_per_episode: int = 50,
    instructions: List[str] = None
):
    """Collect synthetic training data - self-contained without jetbot imports."""
    print(f"Collecting synthetic data...")
    print(f"  Episodes: {num_episodes}")
    print(f"  Steps per episode: {steps_per_episode}")
    print(f"  Output: {output_dir}")

    if instructions is None:
        instructions = DEFAULT_INSTRUCTIONS

    # Create output directory
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    try:
        from isaacsim import SimulationApp
        simulation_app = SimulationApp({"headless": True})

        from omni.isaac.core import World
        from omni.isaac.wheeled_robots.robots import WheeledRobot
        from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
        from omni.isaac.sensor import Camera
        from PIL import Image

        # Create world
        world = World()
        world.scene.add_default_ground_plane()

        # Load JetBot
        jetbot_path = "/workspace/assets/jetbot.usd"
        if not os.path.exists(jetbot_path):
            jetbot_path = download_jetbot_asset()

        jetbot = world.scene.add(
            WheeledRobot(
                prim_path="/World/JetBot",
                name="jetbot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=jetbot_path,
            )
        )

        controller = DifferentialController(
            name="diff_controller",
            wheel_radius=0.0325,
            wheel_base=0.1
        )

        # Find or create camera
        import omni.usd
        from pxr import UsdGeom, Gf
        from omni.isaac.core.utils.prims import create_prim

        stage = omni.usd.get_context().get_stage()

        # First, search for existing camera prim in the JetBot USD
        camera_prim_path = None
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                camera_prim_path = str(prim.GetPath())
                print(f"Found existing camera at: {camera_prim_path}")
                break

        # If no camera found, create one at world level (not attached to robot)
        if camera_prim_path is None:
            camera_prim_path = "/World/Camera"
            print(f"No camera found in USD, creating at: {camera_prim_path}")
            create_prim(camera_prim_path, "Camera")

            # Position camera to look at the scene from above/front
            camera_prim = stage.GetPrimAtPath(camera_prim_path)
            xform = UsdGeom.Xformable(camera_prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.3))  # In front, slightly elevated
            xform.AddRotateXYZOp().Set(Gf.Vec3d(0, 30, 180))  # Looking back at robot

        # IMPORTANT: Reset world BEFORE camera initialization
        world.reset()

        # Create Camera wrapper
        camera = Camera(
            prim_path=camera_prim_path,
            resolution=(224, 224),
            frequency=30
        )
        camera.initialize()

        # Warm-up frames for camera to initialize properly
        print("Warming up camera...")
        for _ in range(20):
            world.step(render=True)

        total_samples = 0

        for episode in range(num_episodes):
            # Reset robot position
            world.reset()

            # Let simulation settle
            for _ in range(5):
                world.step(render=True)

            # Select random instruction for this episode
            instruction = np.random.choice(instructions)

            for step in range(steps_per_episode):
                # Get camera image
                world.step(render=True)
                rgba = camera.get_rgba()

                if rgba is None:
                    print(f"Warning: No camera image at episode {episode}, step {step}")
                    continue

                rgb = rgba[:, :, :3]

                # Get action from scripted policy
                left_speed, right_speed = get_scripted_action(instruction)

                # Save sample
                sample_id = str(uuid.uuid4())

                # Save image
                img_pil = Image.fromarray(rgb.astype(np.uint8))
                img_pil.save(save_path / f"{sample_id}.jpg", quality=95)

                # Save metadata
                metadata = {
                    'instruction': instruction,
                    'action': {
                        'left_speed': left_speed,
                        'right_speed': right_speed
                    },
                    'episode': episode,
                    'step': step,
                    'timestamp': time.time(),
                    'source': 'isaac_sim'
                }
                with open(save_path / f"{sample_id}.json", 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Apply action
                linear = (left_speed + right_speed) / 2.0 * 0.3
                angular = (right_speed - left_speed) / 0.1 * 0.3
                wheel_velocities = controller.forward([linear, angular])
                jetbot.apply_wheel_actions(wheel_velocities)

                total_samples += 1

            print(f"Episode {episode+1}/{num_episodes} complete ({total_samples} samples)")

        print(f"\nData collection complete: {total_samples} samples saved to {save_path}")
        simulation_app.close()

    except Exception as e:
        print(f"Data collection failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='RunPod Setup for JetBot VLA Simulation (Self-contained)'
    )
    parser.add_argument(
        '--download-assets',
        action='store_true',
        help='Download JetBot USD asset'
    )
    parser.add_argument(
        '--test-sim',
        action='store_true',
        help='Test Isaac Sim with JetBot'
    )
    parser.add_argument(
        '--collect-data',
        action='store_true',
        help='Collect synthetic training data'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of episodes for data collection'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=50,
        help='Steps per episode'
    )
    parser.add_argument(
        '--output',
        default='/workspace/sim_data',
        help='Output directory for data'
    )

    args = parser.parse_args()

    if args.download_assets:
        download_jetbot_asset()

    if args.test_sim:
        success = test_simulation()
        sys.exit(0 if success else 1)

    if args.collect_data:
        collect_synthetic_data(
            output_dir=args.output,
            num_episodes=args.episodes,
            steps_per_episode=args.steps
        )

    if not any([args.download_assets, args.test_sim, args.collect_data]):
        parser.print_help()


if __name__ == '__main__':
    main()

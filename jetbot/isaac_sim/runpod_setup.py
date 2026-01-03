#!/usr/bin/env python3
"""
RunPod Setup Script for JetBot VLA Simulation

This script sets up the JetBot VLA simulation environment on RunPod.
Run this after connecting to your RunPod instance.

Usage:
    python runpod_setup.py --download-assets
    python runpod_setup.py --test-sim
    python runpod_setup.py --run-vla --vla-host <host> --vla-port <port>
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


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
        for i in range(10):
            wheel_velocities = controller.forward([0.3, 0.1])
            jetbot.apply_wheel_actions(wheel_velocities)
            world.step(render=False)

        pos, _ = jetbot.get_world_pose()
        print(f"Test passed! JetBot position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

        simulation_app.close()
        return True

    except Exception as e:
        print(f"Test failed: {e}")
        return False


def run_vla_simulation(
    vla_host: str = "localhost",
    vla_port: int = 5555,
    instruction: str = "go forward",
    max_steps: int = 200
):
    """Run VLA-guided simulation."""
    print(f"Starting VLA simulation...")
    print(f"  VLA Server: {vla_host}:{vla_port}")
    print(f"  Instruction: {instruction}")

    # Add jetbot module to path
    jetbot_path = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(jetbot_path))

    from jetbot.isaac_sim import VLASimInterface

    interface = VLASimInterface(
        vla_server_host=vla_host,
        vla_server_port=vla_port,
        headless=True,
        scene='grid'
    )

    try:
        interface.start()

        result = interface.run_vla_navigation(
            instruction=instruction,
            max_steps=max_steps,
            on_step=lambda step, img, action: print(
                f"Step {step}: action=({action[0]:.2f}, {action[1]:.2f})"
            ) if step % 20 == 0 else None
        )

        print(f"\nNavigation complete!")
        print(f"  Steps: {result['num_steps']}")
        print(f"  Final position: {result['final_position']}")

    finally:
        interface.stop()


def collect_synthetic_data(
    output_dir: str = "/workspace/sim_data",
    num_episodes: int = 10,
    steps_per_episode: int = 50
):
    """Collect synthetic training data."""
    print(f"Collecting synthetic data...")
    print(f"  Episodes: {num_episodes}")
    print(f"  Steps per episode: {steps_per_episode}")
    print(f"  Output: {output_dir}")

    jetbot_path = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(jetbot_path))

    from jetbot.isaac_sim import VLASimInterface

    interface = VLASimInterface(
        headless=True,
        scene='grid'
    )

    try:
        interface.start()

        path = interface.collect_data(
            num_episodes=num_episodes,
            steps_per_episode=steps_per_episode,
            save_dir=output_dir,
            policy='scripted'
        )

        print(f"\nData collection complete: {path}")

    finally:
        interface.stop()


def main():
    parser = argparse.ArgumentParser(
        description='RunPod Setup for JetBot VLA Simulation'
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
        '--run-vla',
        action='store_true',
        help='Run VLA-guided simulation'
    )
    parser.add_argument(
        '--collect-data',
        action='store_true',
        help='Collect synthetic training data'
    )
    parser.add_argument(
        '--vla-host',
        default='localhost',
        help='VLA server hostname'
    )
    parser.add_argument(
        '--vla-port',
        type=int,
        default=5555,
        help='VLA server port'
    )
    parser.add_argument(
        '--instruction',
        default='go forward',
        help='Navigation instruction for VLA'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of episodes for data collection'
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

    if args.run_vla:
        run_vla_simulation(
            vla_host=args.vla_host,
            vla_port=args.vla_port,
            instruction=args.instruction
        )

    if args.collect_data:
        collect_synthetic_data(
            output_dir=args.output,
            num_episodes=args.episodes
        )

    if not any([args.download_assets, args.test_sim, args.run_vla, args.collect_data]):
        parser.print_help()


if __name__ == '__main__':
    main()

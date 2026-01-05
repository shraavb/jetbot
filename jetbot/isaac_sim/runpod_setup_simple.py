#!/usr/bin/env python3
"""
RunPod Setup Script for JetBot VLA Simulation (SIMPLE VERSION)

Simplified script that uses ONLY native Isaac Sim primitives for obstacles.
Uses omni.isaac.core.objects which handles all USD/xform operations correctly.

Usage:
    python runpod_setup_simple.py --download-assets
    python runpod_setup_simple.py --test-sim
    python runpod_setup_simple.py --collect-data --episodes 100 --output /workspace/sim_data
"""

import os
import sys
import json
import time
import uuid
import argparse
import random
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
    "approach the red cube",
    "approach the blue cube",
    "approach the green cube",
    "move toward the target",
    "navigate to the goal",
    "go to the red object",
    "go to the blue object",
    "turn around",
    "make a u-turn"
]

# Colors for domain randomization (RGB normalized 0-1)
OBSTACLE_COLORS = {
    'red': (0.9, 0.1, 0.1),
    'green': (0.1, 0.8, 0.1),
    'blue': (0.1, 0.1, 0.9),
    'yellow': (0.9, 0.9, 0.1),
    'orange': (0.9, 0.5, 0.1),
    'purple': (0.6, 0.1, 0.8),
    'cyan': (0.1, 0.8, 0.8),
    'white': (0.9, 0.9, 0.9),
}

GROUND_COLORS = [
    (0.3, 0.3, 0.3),   # Gray
    (0.4, 0.3, 0.2),   # Brown
    (0.2, 0.2, 0.25),  # Dark gray
    (0.5, 0.5, 0.45),  # Light gray
    (0.3, 0.25, 0.2),  # Dark brown
    (0.6, 0.55, 0.5),  # Beige
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


def create_native_obstacles(world, num_obstacles: int, robot_pos: tuple = (0, 0), robot_yaw: float = 0) -> List:
    """
    Create obstacles using native Isaac Sim objects.
    Uses omni.isaac.core.objects which handles USD properly.
    """
    from omni.isaac.core.objects import DynamicCuboid, DynamicSphere, DynamicCylinder, VisualCuboid, VisualSphere, VisualCylinder

    obstacles = []
    shape_types = ['cube', 'sphere', 'cylinder']

    for i in range(num_obstacles):
        # Random position in robot's local frame (in front of robot)
        local_x = np.random.uniform(0.3, 1.5)
        local_y = np.random.uniform(-0.8, 0.8)

        # Transform to world coordinates
        cos_yaw = np.cos(robot_yaw)
        sin_yaw = np.sin(robot_yaw)
        x = robot_pos[0] + local_x * cos_yaw - local_y * sin_yaw
        y = robot_pos[1] + local_x * sin_yaw + local_y * cos_yaw

        # Random color
        color_name = random.choice(list(OBSTACLE_COLORS.keys()))
        color = np.array(OBSTACLE_COLORS[color_name])

        # Random shape
        shape_type = random.choice(shape_types)

        prim_path = f"/World/Obstacle_{i}"

        if shape_type == 'cube':
            size = np.random.uniform(0.05, 0.12)
            obstacle = VisualCuboid(
                prim_path=prim_path,
                name=f"obstacle_{i}",
                position=np.array([x, y, size]),
                scale=np.array([size * 2, size * 2, size * 2]),
                color=color
            )
        elif shape_type == 'sphere':
            radius = np.random.uniform(0.04, 0.10)
            obstacle = VisualSphere(
                prim_path=prim_path,
                name=f"obstacle_{i}",
                position=np.array([x, y, radius]),
                radius=radius,
                color=color
            )
        elif shape_type == 'cylinder':
            radius = np.random.uniform(0.03, 0.08)
            height = np.random.uniform(0.08, 0.20)
            obstacle = VisualCylinder(
                prim_path=prim_path,
                name=f"obstacle_{i}",
                position=np.array([x, y, height / 2]),
                radius=radius,
                height=height,
                color=color
            )

        world.scene.add(obstacle)
        obstacles.append(obstacle)

    return obstacles


def remove_native_obstacles(world, obstacles: List):
    """Remove obstacles from the world."""
    for obstacle in obstacles:
        try:
            world.scene.remove_object(obstacle.name)
        except:
            pass


def randomize_lighting(stage):
    """Randomize scene lighting for visual diversity."""
    from pxr import UsdLux, Gf, UsdGeom

    light_path = "/World/DistantLight"
    light_prim = stage.GetPrimAtPath(light_path)

    if not light_prim.IsValid():
        light_prim = stage.DefinePrim(light_path, "DistantLight")

    light = UsdLux.DistantLight(light_prim)

    # Randomize intensity
    intensity = np.random.uniform(500, 2000)
    light.GetIntensityAttr().Set(intensity)

    # Randomize color temperature
    color_temp = np.random.uniform(4000, 8000)
    if color_temp < 6500:
        r, g, b = 1.0, 0.9 + 0.1 * (color_temp - 4000) / 2500, 0.8 + 0.2 * (color_temp - 4000) / 2500
    else:
        r, g, b = 0.9 - 0.1 * (color_temp - 6500) / 1500, 0.95, 1.0
    light.GetColorAttr().Set(Gf.Vec3f(r, g, b))

    # Randomize angle
    xform = UsdGeom.Xformable(light_prim)
    xform.ClearXformOpOrder()
    angle_x = np.random.uniform(30, 70)
    angle_y = np.random.uniform(-45, 45)
    xform.AddRotateXYZOp().Set(Gf.Vec3d(angle_x, angle_y, 0))


def randomize_ground_color(stage):
    """Randomize ground plane color."""
    from pxr import UsdGeom, Gf

    ground_path = "/World/defaultGroundPlane/GroundPlane/CollisionMesh"
    ground_prim = stage.GetPrimAtPath(ground_path)

    if ground_prim.IsValid():
        color = GROUND_COLORS[np.random.randint(len(GROUND_COLORS))]
        gprim = UsdGeom.Gprim(ground_prim)
        gprim.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])


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
    instructions: List[str] = None,
    num_obstacles: int = 15
):
    """
    Collect synthetic training data using native Isaac Sim objects.
    """
    print(f"Collecting synthetic data (SIMPLE version - native Isaac Sim objects)...")
    print(f"  Episodes: {num_episodes}")
    print(f"  Steps per episode: {steps_per_episode}")
    print(f"  Obstacles per scene: 5-{num_obstacles}")
    print(f"  Output: {output_dir}")

    if instructions is None:
        instructions = DEFAULT_INSTRUCTIONS

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

        # Create camera attached to robot using native Isaac Sim Camera
        import omni.usd
        from pxr import UsdGeom, Gf
        from omni.isaac.core.utils.prims import create_prim

        stage = omni.usd.get_context().get_stage()

        camera_prim_path = "/World/JetBot/chassis/front_camera"
        print(f"Creating first-person camera at: {camera_prim_path}")
        create_prim(camera_prim_path, "Camera")

        camera_prim = stage.GetPrimAtPath(camera_prim_path)
        xform = UsdGeom.Xformable(camera_prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(0.1, 0.0, 0.06))
        xform.AddRotateXYZOp().Set(Gf.Vec3d(90, 0, -90))

        camera_geom = UsdGeom.Camera(camera_prim)
        camera_geom.GetFocalLengthAttr().Set(18.0)
        camera_geom.GetHorizontalApertureAttr().Set(20.955)
        camera_geom.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10.0))

        world.reset()

        camera = Camera(
            prim_path=camera_prim_path,
            resolution=(224, 224),
            frequency=30
        )
        camera.initialize()

        print("Warming up camera...")
        for _ in range(20):
            world.step(render=True)

        total_samples = 0
        obstacles = []

        for episode in range(num_episodes):
            # Remove previous obstacles
            if obstacles:
                remove_native_obstacles(world, obstacles)
                obstacles = []

            # Reset world clears physics state
            world.reset()

            # Random robot pose
            rand_x = np.random.uniform(-0.3, 0.3)
            rand_y = np.random.uniform(-0.3, 0.3)
            rand_yaw = np.random.uniform(-np.pi, np.pi)

            quat_w = np.cos(rand_yaw / 2)
            quat_z = np.sin(rand_yaw / 2)
            orientation = np.array([quat_w, 0.0, 0.0, quat_z])

            jetbot.set_world_pose(
                position=np.array([rand_x, rand_y, 0.05]),
                orientation=orientation
            )

            # Add obstacles using native Isaac Sim objects
            actual_num_obstacles = np.random.randint(5, num_obstacles + 1)
            obstacles = create_native_obstacles(
                world,
                actual_num_obstacles,
                robot_pos=(rand_x, rand_y),
                robot_yaw=rand_yaw
            )

            # Randomize lighting and ground
            try:
                randomize_lighting(stage)
            except:
                pass

            try:
                randomize_ground_color(stage)
            except:
                pass

            # Let simulation settle
            for _ in range(10):
                world.step(render=True)

            instruction = np.random.choice(instructions)

            scene_info = {
                'num_obstacles': len(obstacles),
                'has_obstacles': len(obstacles) > 0
            }

            for step in range(steps_per_episode):
                world.step(render=True)
                rgba = camera.get_rgba()

                if rgba is None:
                    print(f"Warning: No camera image at episode {episode}, step {step}")
                    continue

                rgb = rgba[:, :, :3]

                left_speed, right_speed = get_scripted_action(instruction)

                sample_id = str(uuid.uuid4())

                img_pil = Image.fromarray(rgb.astype(np.uint8))
                img_pil.save(save_path / f"{sample_id}.jpg", quality=95)

                metadata = {
                    'instruction': instruction,
                    'action': {
                        'left_speed': left_speed,
                        'right_speed': right_speed
                    },
                    'episode': episode,
                    'step': step,
                    'timestamp': time.time(),
                    'source': 'isaac_sim_simple',
                    'scene': scene_info
                }
                with open(save_path / f"{sample_id}.json", 'w') as f:
                    json.dump(metadata, f, indent=2)

                linear = (left_speed + right_speed) / 2.0 * 0.3
                angular = (right_speed - left_speed) / 0.1 * 0.3
                wheel_velocities = controller.forward([linear, angular])
                jetbot.apply_wheel_actions(wheel_velocities)

                total_samples += 1

            print(f"Episode {episode+1}/{num_episodes} complete "
                  f"({total_samples} samples, {len(obstacles)} obstacles)")

        if obstacles:
            remove_native_obstacles(world, obstacles)

        print(f"\nData collection complete: {total_samples} samples saved to {save_path}")
        simulation_app.close()

    except Exception as e:
        print(f"Data collection failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='RunPod Setup for JetBot VLA Simulation (SIMPLE - Native Isaac Sim Objects)'
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
    parser.add_argument(
        '--obstacles',
        type=int,
        default=15,
        help='Max number of obstacles per scene'
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
            steps_per_episode=args.steps,
            num_obstacles=args.obstacles
        )

    if not any([args.download_assets, args.test_sim, args.collect_data]):
        parser.print_help()


if __name__ == '__main__':
    main()

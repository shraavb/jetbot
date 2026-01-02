"""
JetBot Simulation in Isaac Sim

This module provides a JetBot robot simulation using NVIDIA Isaac Sim.
It enables VLA training data collection and model testing in simulation.

Requirements:
    - NVIDIA Isaac Sim installed
    - isaacsim Python packages available

Usage:
    from jetbot.isaac_sim import JetBotSim

    sim = JetBotSim()
    sim.start()

    # Control robot
    sim.set_velocity(left=0.5, right=0.5)

    # Get camera image
    image = sim.get_camera_image()

    sim.stop()
"""

import numpy as np
from typing import Tuple, Optional
from PIL import Image


class JetBotSim:
    """
    JetBot simulation wrapper for Isaac Sim.

    Provides a consistent interface for controlling a simulated JetBot
    that matches the real robot's API.

    Attributes:
        wheel_radius: Wheel radius in meters (default: 0.03 for JetBot)
        wheel_base: Distance between wheels in meters (default: 0.1125)
        max_linear_speed: Maximum linear speed in m/s
        max_angular_speed: Maximum angular speed in rad/s
    """

    # JetBot physical parameters
    WHEEL_RADIUS = 0.03  # 3 cm
    WHEEL_BASE = 0.1125  # 11.25 cm

    def __init__(
        self,
        headless: bool = False,
        physics_dt: float = 1/60.0,
        rendering_dt: float = 1/60.0,
        max_linear_speed: float = 0.3,
        max_angular_speed: float = 1.0,
        camera_resolution: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize JetBot simulation.

        Args:
            headless: Run simulation without GUI
            physics_dt: Physics timestep in seconds
            rendering_dt: Rendering timestep in seconds
            max_linear_speed: Maximum linear velocity in m/s
            max_angular_speed: Maximum angular velocity in rad/s
            camera_resolution: Camera image resolution (width, height)
        """
        self.headless = headless
        self.physics_dt = physics_dt
        self.rendering_dt = rendering_dt
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.camera_resolution = camera_resolution

        self._simulation_app = None
        self._world = None
        self._jetbot = None
        self._controller = None
        self._camera = None
        self._is_running = False

    def start(self, scene: str = "simple_room") -> None:
        """
        Start the Isaac Sim simulation.

        Args:
            scene: Scene to load ('simple_room', 'warehouse', 'office', or custom USD path)
        """
        try:
            from isaacsim import SimulationApp

            # Initialize simulation app
            self._simulation_app = SimulationApp({
                "headless": self.headless,
                "width": 1280,
                "height": 720
            })

            # Import after SimulationApp is created
            from isaacsim.core.api import World
            from isaacsim.robot.wheeled_robots import WheeledRobot
            from isaacsim.robot.wheeled_robots.controllers import DifferentialController
            from omni.isaac.sensor import Camera

            # Create world
            self._world = World(
                physics_dt=self.physics_dt,
                rendering_dt=self.rendering_dt
            )

            # Load scene
            self._load_scene(scene)

            # Add JetBot robot
            self._jetbot = self._world.scene.add(
                WheeledRobot(
                    prim_path="/World/JetBot",
                    name="jetbot",
                    wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                    create_robot=True,
                    usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Robots/Jetbot/jetbot.usd",
                    position=np.array([0.0, 0.0, 0.0])
                )
            )

            # Create differential controller
            self._controller = DifferentialController(
                name="jetbot_controller",
                wheel_radius=self.WHEEL_RADIUS,
                wheel_base=self.WHEEL_BASE
            )

            # Add camera
            self._camera = Camera(
                prim_path="/World/JetBot/chassis/camera",
                resolution=self.camera_resolution
            )
            self._camera.initialize()

            # Reset world
            self._world.reset()
            self._is_running = True

            print(f"JetBot simulation started (headless={self.headless})")

        except ImportError as e:
            raise ImportError(
                "Isaac Sim not found. Please install NVIDIA Isaac Sim and ensure "
                "the Python environment is properly configured.\n"
                f"Original error: {e}"
            )

    def _load_scene(self, scene: str) -> None:
        """Load a simulation scene."""
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        import omni.usd

        assets_root = get_assets_root_path()

        scene_paths = {
            "simple_room": f"{assets_root}/Isaac/Environments/Simple_Room/simple_room.usd",
            "warehouse": f"{assets_root}/Isaac/Environments/Simple_Warehouse/warehouse.usd",
            "office": f"{assets_root}/Isaac/Environments/Office/office.usd",
            "grid": f"{assets_root}/Isaac/Environments/Grid/default_environment.usd"
        }

        if scene in scene_paths:
            scene_path = scene_paths[scene]
        else:
            # Assume custom path
            scene_path = scene

        # Load USD
        omni.usd.get_context().open_stage(scene_path)

    def stop(self) -> None:
        """Stop the simulation and cleanup resources."""
        if self._simulation_app is not None:
            self._simulation_app.close()
            self._simulation_app = None

        self._world = None
        self._jetbot = None
        self._controller = None
        self._camera = None
        self._is_running = False

        print("JetBot simulation stopped")

    def step(self) -> None:
        """Advance the simulation by one timestep."""
        if not self._is_running:
            raise RuntimeError("Simulation not running. Call start() first.")

        self._world.step(render=not self.headless)

    def set_velocity(self, left: float, right: float) -> None:
        """
        Set wheel velocities.

        Args:
            left: Left wheel speed [-1.0, 1.0]
            right: Right wheel speed [-1.0, 1.0]
        """
        if not self._is_running:
            raise RuntimeError("Simulation not running. Call start() first.")

        # Clamp values
        left = max(-1.0, min(1.0, left))
        right = max(-1.0, min(1.0, right))

        # Convert to linear/angular velocity
        linear = (left + right) / 2.0 * self.max_linear_speed
        angular = (right - left) / self.WHEEL_BASE * self.max_linear_speed

        # Get wheel commands from controller
        wheel_actions = self._controller.forward(
            command=[linear, angular]
        )

        # Apply to robot
        self._jetbot.apply_wheel_actions(wheel_actions)

    def set_differential_velocity(self, linear: float, angular: float) -> None:
        """
        Set velocity using differential drive (linear, angular).

        Args:
            linear: Linear velocity in m/s
            angular: Angular velocity in rad/s
        """
        if not self._is_running:
            raise RuntimeError("Simulation not running. Call start() first.")

        wheel_actions = self._controller.forward(
            command=[linear, angular]
        )
        self._jetbot.apply_wheel_actions(wheel_actions)

    def get_camera_image(self) -> np.ndarray:
        """
        Get current camera image.

        Returns:
            RGB image as numpy array (H, W, 3)
        """
        if not self._is_running:
            raise RuntimeError("Simulation not running. Call start() first.")

        # Get RGBA image
        rgba = self._camera.get_rgba()

        # Convert to RGB
        rgb = rgba[:, :, :3]

        return rgb

    def get_camera_image_pil(self) -> Image.Image:
        """
        Get current camera image as PIL Image.

        Returns:
            RGB PIL Image
        """
        rgb = self.get_camera_image()
        return Image.fromarray(rgb)

    def get_position(self) -> np.ndarray:
        """
        Get robot position in world coordinates.

        Returns:
            Position array [x, y, z]
        """
        if not self._is_running:
            raise RuntimeError("Simulation not running. Call start() first.")

        position, _ = self._jetbot.get_world_pose()
        return position

    def get_orientation(self) -> np.ndarray:
        """
        Get robot orientation as quaternion.

        Returns:
            Quaternion array [w, x, y, z]
        """
        if not self._is_running:
            raise RuntimeError("Simulation not running. Call start() first.")

        _, orientation = self._jetbot.get_world_pose()
        return orientation

    def reset(self, position: Optional[np.ndarray] = None) -> None:
        """
        Reset robot to initial position.

        Args:
            position: Optional new position [x, y, z]
        """
        if not self._is_running:
            raise RuntimeError("Simulation not running. Call start() first.")

        self._world.reset()

        if position is not None:
            self._jetbot.set_world_pose(position=position)

    @property
    def is_running(self) -> bool:
        """Check if simulation is running."""
        return self._is_running


def create_simple_scene():
    """Create a simple test scene with obstacles."""
    try:
        from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
        from omni.isaac.core.prims import XFormPrim

        # Add ground plane
        ground = VisualCuboid(
            prim_path="/World/ground",
            name="ground",
            position=np.array([0.0, 0.0, -0.01]),
            scale=np.array([10.0, 10.0, 0.02]),
            color=np.array([0.5, 0.5, 0.5])
        )

        # Add some obstacles
        obstacles = []
        obstacle_positions = [
            [1.0, 0.0, 0.1],
            [-0.5, 0.8, 0.1],
            [0.5, -0.7, 0.1]
        ]

        for i, pos in enumerate(obstacle_positions):
            obs = DynamicCuboid(
                prim_path=f"/World/obstacle_{i}",
                name=f"obstacle_{i}",
                position=np.array(pos),
                scale=np.array([0.2, 0.2, 0.2]),
                color=np.array([0.8, 0.2, 0.2])
            )
            obstacles.append(obs)

        return ground, obstacles

    except ImportError:
        print("Warning: Isaac Sim not available, cannot create scene")
        return None, []

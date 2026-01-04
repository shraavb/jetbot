"""
Language-Grounded Data Collection for VLA Training

Collects training data with colored objects and natural language commands.
Uses a scripted or heuristic policy to generate labeled (image, instruction, action) tuples.

The data format matches the existing JetBot VLA dataset format:
- {uuid}.jpg: Camera image
- {uuid}.json: Metadata with instruction and action

Usage:
    python -m jetbot.isaac_sim.collect_language_data \
        --config multi_target \
        --episodes 100 \
        --output ./dataset_language

    # Or with mock mode (no Isaac Sim required)
    python -m jetbot.isaac_sim.collect_language_data \
        --mock \
        --episodes 50 \
        --output ./dataset_language_mock
"""

import time
import json
import uuid
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from PIL import Image


@dataclass
class LanguageCommand:
    """A language command with associated target and action policy."""
    instruction: str
    target_type: Optional[str]  # 'ball', 'chair', 'door', 'person', etc.
    target_color: Optional[str]  # 'red', 'blue', 'green', etc.
    command_type: str  # 'approach', 'avoid', 'follow', 'directional'


# Language templates for generating diverse commands
LANGUAGE_TEMPLATES = {
    'approach_color_object': [
        "go towards the {color} {object}",
        "go to the {color} {object}",
        "navigate to the {color} {object}",
        "move towards the {color} {object}",
        "approach the {color} {object}",
        "drive to the {color} {object}",
        "head towards the {color} {object}",
        "go in the direction of the {color} {object}",
    ],
    'approach_object': [
        "go towards the {object}",
        "go to the {object}",
        "navigate to the {object}",
        "move towards the {object}",
        "approach the {object}",
        "drive to the {object}",
        "head towards the {object}",
    ],
    'avoid_object': [
        "go around the {object}",
        "avoid the {object}",
        "navigate around the {object}",
        "circumvent the {object}",
        "steer clear of the {object}",
        "go past the {object}",
    ],
    'avoid_robot': [
        "avoid colliding into the jetbot",
        "swerve around the jetbot in front of you",
        "go around the other robot",
        "avoid the jetbot ahead",
        "navigate around the robot",
        "steer clear of the jetbot",
        "don't hit the other robot",
        "avoid the robot in your path",
        "go around the jetbot",
        "swerve to avoid the robot",
        "avoid collision with the jetbot",
        "navigate past the other robot",
    ],
    'speed_modulation': [
        "speed up as you go towards the {color} {object}",
        "slow down as you approach the {object}",
        "go fast towards the {color} {object}",
        "slowly approach the {object}",
        "quickly navigate to the {color} {object}",
        "carefully approach the {object}",
        "rush towards the {color} {object}",
        "cautiously move towards the {object}",
        "accelerate as you head to the {object}",
        "decelerate near the {object}",
        "go full speed to the {color} {object}",
        "move slowly towards the {object}",
    ],
    'conditional_actions': [
        "turn left near the {object}",
        "turn right when you reach the {object}",
        "slow down as you go around the {color} {object}",
        "speed up after passing the {object}",
        "stop when you reach the {object}",
        "turn around at the {color} {object}",
        "go left after the {object}",
        "turn right before the {object}",
        "slow down near the {object}",
        "speed up once past the {object}",
        "stop near the {color} {object}",
        "turn left at the {object}",
    ],
    'complex_navigation': [
        "go quickly towards the {color} {object} then slow down",
        "speed up as you go around the {object}",
        "slowly approach the {object} while avoiding obstacles",
        "fast approach to the {color} {object} then stop",
        "carefully navigate around the {object} then speed up",
        "rush to the {color} {object} but slow near the obstacle",
        "accelerate towards the {object} avoiding obstacles",
    ],
    'follow_person': [
        "follow the person",
        "follow the person in front of you",
        "go after the person",
        "track the person",
        "follow the human",
        "go towards the person",
        "move towards the person",
    ],
    'directional': [
        "go forward",
        "move forward",
        "drive straight ahead",
        "go straight",
        "turn left",
        "turn right",
        "rotate left",
        "rotate right",
        "spin left",
        "spin right",
        "go backward",
        "move back",
        "reverse",
        "stop",
        "halt",
    ],
}

COLORS = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
APPROACHABLE_OBJECTS = ['ball', 'box', 'cone', 'door']
AVOIDABLE_OBJECTS = ['chair', 'box', 'obstacle', 'wall', 'jetbot', 'robot']


class LanguageDataCollector:
    """
    Collects training data for language-grounded VLA control.

    Uses a heuristic policy that:
    1. For approach commands: Moves towards the target object
    2. For avoid commands: Moves away from / around the object
    3. For directional commands: Executes simple motor commands
    """

    def __init__(
        self,
        output_dir: str = './dataset_language',
        headless: bool = True,
        scene_config: str = 'multi_target'
    ):
        """
        Initialize data collector.

        Args:
            output_dir: Directory to save collected data
            headless: Run simulation without GUI
            scene_config: Scene configuration to use
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        self.scene_config = scene_config

        self._world = None
        self._scene = None
        self._jetbot = None
        self._camera = None
        self._controller = None
        self._simulation_app = None

        self._sample_count = 0

    def setup(self) -> None:
        """Setup simulation environment."""
        try:
            from isaacsim import SimulationApp

            self._simulation_app = SimulationApp({
                "headless": self.headless,
                "width": 1280,
                "height": 720
            })

            from omni.isaac.core import World
            from omni.isaac.wheeled_robots.robots import WheeledRobot
            from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
            from omni.isaac.sensor import Camera

            from .experiment_scene import ExperimentScene, SCENE_CONFIGS, load_scene_from_config

            # Create world
            self._world = World(physics_dt=1/60.0, rendering_dt=1/60.0)

            # Setup scene
            self._scene = ExperimentScene()
            self._scene.setup(self._world)

            if self.scene_config in SCENE_CONFIGS:
                load_scene_from_config(self._scene, SCENE_CONFIGS[self.scene_config])

            # Add JetBot
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

            self._controller = DifferentialController(
                name="jetbot_controller",
                wheel_radius=0.03,
                wheel_base=0.1125
            )

            self._camera = Camera(
                prim_path="/World/JetBot/chassis/camera",
                resolution=(224, 224)
            )
            self._camera.initialize()

            self._world.reset()
            print("Simulation setup complete")
            print(f"Scene: {self._scene.get_scene_description()}")

        except ImportError as e:
            raise ImportError(f"Isaac Sim not available: {e}")

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._simulation_app:
            self._simulation_app.close()

    def generate_language_command(self) -> LanguageCommand:
        """Generate a random language command based on scene objects."""
        # Get available objects
        all_objects = self._scene.get_all_objects()

        # Check if there are jetbots in the scene
        jetbots = self._scene.get_objects_by_type('jetbot')
        has_jetbots = len(jetbots) > 0

        if has_jetbots:
            command_type = np.random.choice([
                'approach_color_object',
                'approach_object',
                'avoid_object',
                'avoid_robot',
                'speed_modulation',
                'conditional_actions',
                'complex_navigation',
                'follow_person',
                'directional'
            ], p=[0.15, 0.10, 0.08, 0.12, 0.15, 0.12, 0.08, 0.08, 0.12])
        else:
            command_type = np.random.choice([
                'approach_color_object',
                'approach_object',
                'avoid_object',
                'speed_modulation',
                'conditional_actions',
                'complex_navigation',
                'follow_person',
                'directional'
            ], p=[0.18, 0.12, 0.10, 0.18, 0.15, 0.10, 0.07, 0.10])

        if command_type == 'approach_color_object':
            # Find colored objects
            colored_objects = [
                obj for obj in all_objects.values()
                if obj.color is not None and obj.object_type in APPROACHABLE_OBJECTS
            ]
            if colored_objects:
                obj = np.random.choice(colored_objects)
                template = np.random.choice(LANGUAGE_TEMPLATES['approach_color_object'])
                instruction = template.format(
                    color=obj.color.name.lower(),
                    object=obj.object_type
                )
                return LanguageCommand(
                    instruction=instruction,
                    target_type=obj.object_type,
                    target_color=obj.color.name.lower(),
                    command_type='approach'
                )

        elif command_type == 'approach_object':
            approachable = [
                obj for obj in all_objects.values()
                if obj.object_type in APPROACHABLE_OBJECTS + ['door', 'person']
            ]
            if approachable:
                obj = np.random.choice(approachable)
                template = np.random.choice(LANGUAGE_TEMPLATES['approach_object'])
                instruction = template.format(object=obj.object_type)
                return LanguageCommand(
                    instruction=instruction,
                    target_type=obj.object_type,
                    target_color=None,
                    command_type='approach'
                )

        elif command_type == 'avoid_object':
            avoidable = [
                obj for obj in all_objects.values()
                if obj.object_type in AVOIDABLE_OBJECTS
            ]
            if avoidable:
                obj = np.random.choice(avoidable)
                template = np.random.choice(LANGUAGE_TEMPLATES['avoid_object'])
                instruction = template.format(object=obj.object_type)
                return LanguageCommand(
                    instruction=instruction,
                    target_type=obj.object_type,
                    target_color=None,
                    command_type='avoid'
                )

        elif command_type == 'avoid_robot':
            jetbots = self._scene.get_objects_by_type('jetbot')
            if jetbots:
                instruction = np.random.choice(LANGUAGE_TEMPLATES['avoid_robot'])
                return LanguageCommand(
                    instruction=instruction,
                    target_type='jetbot',
                    target_color=None,
                    command_type='avoid'
                )

        elif command_type == 'speed_modulation':
            # Speed modulation commands - approach with speed modifier
            colored_objects = [
                obj for obj in all_objects.values()
                if obj.color is not None and obj.object_type in APPROACHABLE_OBJECTS
            ]
            all_approachable = [
                obj for obj in all_objects.values()
                if obj.object_type in APPROACHABLE_OBJECTS + ['door', 'person']
            ]

            template = np.random.choice(LANGUAGE_TEMPLATES['speed_modulation'])

            # Check if template needs color
            if '{color}' in template and colored_objects:
                obj = np.random.choice(colored_objects)
                instruction = template.format(
                    color=obj.color.name.lower(),
                    object=obj.object_type
                )
                return LanguageCommand(
                    instruction=instruction,
                    target_type=obj.object_type,
                    target_color=obj.color.name.lower(),
                    command_type='speed_approach'
                )
            elif all_approachable:
                obj = np.random.choice(all_approachable)
                instruction = template.format(object=obj.object_type)
                return LanguageCommand(
                    instruction=instruction,
                    target_type=obj.object_type,
                    target_color=None,
                    command_type='speed_approach'
                )

        elif command_type == 'conditional_actions':
            # Conditional actions - turn/stop/speed change near objects
            all_approachable = [
                obj for obj in all_objects.values()
                if obj.object_type in APPROACHABLE_OBJECTS + ['door', 'person', 'chair']
            ]
            colored_objects = [
                obj for obj in all_objects.values()
                if obj.color is not None
            ]

            template = np.random.choice(LANGUAGE_TEMPLATES['conditional_actions'])

            if '{color}' in template and colored_objects:
                obj = np.random.choice(colored_objects)
                instruction = template.format(
                    color=obj.color.name.lower(),
                    object=obj.object_type
                )
                return LanguageCommand(
                    instruction=instruction,
                    target_type=obj.object_type,
                    target_color=obj.color.name.lower(),
                    command_type='conditional'
                )
            elif all_approachable:
                obj = np.random.choice(all_approachable)
                instruction = template.format(object=obj.object_type)
                return LanguageCommand(
                    instruction=instruction,
                    target_type=obj.object_type,
                    target_color=None,
                    command_type='conditional'
                )

        elif command_type == 'complex_navigation':
            # Complex navigation - multi-step instructions
            colored_objects = [
                obj for obj in all_objects.values()
                if obj.color is not None and obj.object_type in APPROACHABLE_OBJECTS
            ]
            all_approachable = [
                obj for obj in all_objects.values()
                if obj.object_type in APPROACHABLE_OBJECTS + ['door', 'person', 'chair']
            ]

            template = np.random.choice(LANGUAGE_TEMPLATES['complex_navigation'])

            if '{color}' in template and colored_objects:
                obj = np.random.choice(colored_objects)
                instruction = template.format(
                    color=obj.color.name.lower(),
                    object=obj.object_type
                )
                return LanguageCommand(
                    instruction=instruction,
                    target_type=obj.object_type,
                    target_color=obj.color.name.lower(),
                    command_type='complex'
                )
            elif all_approachable:
                obj = np.random.choice(all_approachable)
                instruction = template.format(object=obj.object_type)
                return LanguageCommand(
                    instruction=instruction,
                    target_type=obj.object_type,
                    target_color=None,
                    command_type='complex'
                )

        elif command_type == 'follow_person':
            persons = self._scene.get_objects_by_type('person')
            if persons:
                instruction = np.random.choice(LANGUAGE_TEMPLATES['follow_person'])
                return LanguageCommand(
                    instruction=instruction,
                    target_type='person',
                    target_color=None,
                    command_type='follow'
                )

        # Default to directional command
        instruction = np.random.choice(LANGUAGE_TEMPLATES['directional'])
        return LanguageCommand(
            instruction=instruction,
            target_type=None,
            target_color=None,
            command_type='directional'
        )

    def compute_heuristic_action(
        self,
        command: LanguageCommand,
        robot_position: np.ndarray,
        robot_orientation: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute action based on heuristic policy.

        Args:
            command: Language command
            robot_position: Robot's current position [x, y, z]
            robot_orientation: Robot's quaternion orientation [w, x, y, z]

        Returns:
            (left_speed, right_speed) tuple
        """
        # Add noise for realistic training data
        noise = np.random.normal(0, 0.05, 2)

        if command.command_type == 'directional':
            return self._directional_action(command.instruction, noise)

        # Find target object
        target = self._find_target(command)
        if target is None:
            # Fallback to random exploration
            return (
                np.clip(0.3 + noise[0], -1, 1),
                np.clip(0.3 + noise[1], -1, 1)
            )

        # Compute relative position to target
        target_pos = target.position[:2]
        robot_pos = robot_position[:2]
        direction = target_pos - robot_pos
        distance = np.linalg.norm(direction)

        # Get robot heading from quaternion
        # Assuming quaternion is [w, x, y, z]
        w, x, y, z = robot_orientation
        # Yaw from quaternion
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

        # Angle to target
        target_angle = np.arctan2(direction[1], direction[0])
        angle_diff = target_angle - yaw

        # Normalize angle to [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        # Determine speed modifier from instruction
        speed_mult = self._get_speed_modifier(command.instruction)

        if command.command_type == 'approach' or command.command_type == 'follow':
            # Move towards target
            if abs(angle_diff) > 0.3:
                # Need to turn
                if angle_diff > 0:
                    # Turn left
                    left, right = 0.2, 0.5
                else:
                    # Turn right
                    left, right = 0.5, 0.2
            else:
                # Go forward
                speed = min(0.6, distance * 0.5 + 0.2)
                left, right = speed, speed

        elif command.command_type == 'speed_approach':
            # Approach with speed modulation
            if abs(angle_diff) > 0.3:
                # Turn towards target
                if angle_diff > 0:
                    left, right = 0.2 * speed_mult, 0.5 * speed_mult
                else:
                    left, right = 0.5 * speed_mult, 0.2 * speed_mult
            else:
                # Move with modified speed
                base_speed = min(0.6, distance * 0.5 + 0.2)
                speed = base_speed * speed_mult
                left, right = speed, speed

        elif command.command_type == 'conditional':
            # Conditional actions - behavior changes near target
            instruction_lower = command.instruction.lower()

            if distance < 0.8:  # Near the target
                if 'turn left' in instruction_lower:
                    left, right = 0.15, 0.45
                elif 'turn right' in instruction_lower:
                    left, right = 0.45, 0.15
                elif 'stop' in instruction_lower:
                    left, right = 0.0, 0.0
                elif 'slow' in instruction_lower:
                    left, right = 0.2, 0.2
                elif 'turn around' in instruction_lower:
                    left, right = 0.4, -0.4
                else:
                    left, right = 0.3, 0.3
            else:
                # Approach the target first
                if abs(angle_diff) > 0.3:
                    if angle_diff > 0:
                        left, right = 0.2, 0.5
                    else:
                        left, right = 0.5, 0.2
                else:
                    speed = min(0.5, distance * 0.4 + 0.2)
                    left, right = speed, speed

        elif command.command_type == 'complex':
            # Complex multi-step navigation
            instruction_lower = command.instruction.lower()

            # Determine phase based on distance
            if distance > 1.0:  # Far from target - initial phase
                if 'quickly' in instruction_lower or 'fast' in instruction_lower or 'rush' in instruction_lower:
                    speed_mult = 1.3
                else:
                    speed_mult = 1.0

                if abs(angle_diff) > 0.3:
                    if angle_diff > 0:
                        left, right = 0.25 * speed_mult, 0.55 * speed_mult
                    else:
                        left, right = 0.55 * speed_mult, 0.25 * speed_mult
                else:
                    speed = min(0.7, distance * 0.4 + 0.3) * speed_mult
                    left, right = speed, speed
            else:  # Near target - secondary phase
                if 'slow down' in instruction_lower or 'then slow' in instruction_lower:
                    left, right = 0.2, 0.2
                elif 'stop' in instruction_lower:
                    left, right = 0.05, 0.05
                elif 'speed up' in instruction_lower:
                    left, right = 0.6, 0.6
                else:
                    left, right = 0.3, 0.3

        elif command.command_type == 'avoid':
            # Move away from / around target
            if distance < 0.5:
                # Too close, turn away
                if angle_diff > 0:
                    left, right = 0.5, 0.2
                else:
                    left, right = 0.2, 0.5
            else:
                # Move perpendicular
                perp_angle = angle_diff + np.pi/2
                if abs(perp_angle) > 0.3:
                    if perp_angle > 0:
                        left, right = 0.3, 0.5
                    else:
                        left, right = 0.5, 0.3
                else:
                    left, right = 0.4, 0.4

        else:
            left, right = 0.3, 0.3

        return (
            np.clip(left + noise[0], -1, 1),
            np.clip(right + noise[1], -1, 1)
        )

    def _directional_action(
        self,
        instruction: str,
        noise: np.ndarray
    ) -> Tuple[float, float]:
        """Get action for directional commands."""
        instruction = instruction.lower()

        if 'forward' in instruction or 'straight' in instruction:
            left, right = 0.5, 0.5
        elif 'left' in instruction:
            left, right = 0.2, 0.5
        elif 'right' in instruction:
            left, right = 0.5, 0.2
        elif 'backward' in instruction or 'back' in instruction or 'reverse' in instruction:
            left, right = -0.3, -0.3
        elif 'stop' in instruction or 'halt' in instruction:
            left, right = 0.0, 0.0
        else:
            left, right = 0.3, 0.3

        return (
            np.clip(left + noise[0], -1, 1),
            np.clip(right + noise[1], -1, 1)
        )

    def _find_target(self, command: LanguageCommand):
        """Find target object for a command."""
        if command.target_color and command.target_type:
            # Look for specific color + type
            from .experiment_scene import ObjectColor
            try:
                color = ObjectColor[command.target_color.upper()]
                colored = self._scene.get_objects_by_color(color)
                for obj in colored:
                    if obj.object_type == command.target_type:
                        return obj
            except KeyError:
                pass

        if command.target_type:
            objects = self._scene.get_objects_by_type(command.target_type)
            if objects:
                return objects[0]

        return None

    def _get_speed_modifier(self, instruction: str) -> float:
        """
        Extract speed modifier from instruction text.

        Returns a multiplier for speed based on keywords:
        - "speed up", "fast", "quickly", "rush", "accelerate", "full speed" -> 1.3-1.5
        - "slow down", "slowly", "carefully", "cautiously", "decelerate" -> 0.5-0.7
        - Default -> 1.0
        """
        instruction_lower = instruction.lower()

        # Fast keywords
        fast_keywords = ['speed up', 'fast', 'quickly', 'rush', 'accelerate', 'full speed']
        for kw in fast_keywords:
            if kw in instruction_lower:
                return np.random.uniform(1.2, 1.5)

        # Slow keywords
        slow_keywords = ['slow down', 'slowly', 'carefully', 'cautiously', 'decelerate', 'move slowly']
        for kw in slow_keywords:
            if kw in instruction_lower:
                return np.random.uniform(0.4, 0.7)

        return 1.0

    def collect_episode(
        self,
        steps: int = 50,
        randomize_start: bool = True
    ) -> int:
        """
        Collect one episode of data.

        Args:
            steps: Number of steps per episode
            randomize_start: Randomize robot starting position

        Returns:
            Number of samples collected
        """
        # Reset
        self._world.reset()

        if randomize_start:
            # Random start position within bounds
            start_x = np.random.uniform(-1.5, 1.5)
            start_y = np.random.uniform(-1.5, 1.5)
            start_yaw = np.random.uniform(-np.pi, np.pi)

            # Set position (quaternion for yaw rotation around z)
            quat = np.array([np.cos(start_yaw/2), 0, 0, np.sin(start_yaw/2)])
            self._jetbot.set_world_pose(
                position=np.array([start_x, start_y, 0.0]),
                orientation=quat
            )

        # Generate command for this episode
        command = self.generate_language_command()
        samples_collected = 0

        for step in range(steps):
            # Get camera image
            rgba = self._camera.get_rgba()
            rgb = rgba[:, :, :3]

            # Get robot state
            position, orientation = self._jetbot.get_world_pose()

            # Compute action
            left, right = self.compute_heuristic_action(command, position, orientation)

            # Save sample
            sample_id = str(uuid.uuid4())

            # Save image
            img = Image.fromarray(rgb)
            img.save(self.output_dir / f"{sample_id}.jpg", quality=95)

            # Save metadata
            metadata = {
                'instruction': command.instruction,
                'action': {
                    'left_speed': float(left),
                    'right_speed': float(right)
                },
                'command_type': command.command_type,
                'target_type': command.target_type,
                'target_color': command.target_color,
                'robot_position': position.tolist(),
                'episode_step': step,
                'timestamp': time.time(),
                'source': 'isaac_sim_language'
            }

            with open(self.output_dir / f"{sample_id}.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            # Apply action
            linear = (left + right) / 2.0 * 0.3
            angular = (right - left) / 0.1125 * 0.3
            wheel_actions = self._controller.forward(command=[linear, angular])
            self._jetbot.apply_wheel_actions(wheel_actions)

            # Step simulation
            self._world.step(render=not self.headless)

            samples_collected += 1
            self._sample_count += 1

        return samples_collected

    def collect_data(
        self,
        num_episodes: int = 100,
        steps_per_episode: int = 50
    ) -> Dict[str, Any]:
        """
        Collect multiple episodes of training data.

        Args:
            num_episodes: Number of episodes to collect
            steps_per_episode: Steps per episode

        Returns:
            Collection statistics
        """
        print(f"\nCollecting {num_episodes} episodes ({steps_per_episode} steps each)")
        print(f"Output: {self.output_dir}")
        print(f"Scene: {self._scene.get_scene_description()}\n")

        start_time = time.time()
        total_samples = 0

        for episode in range(num_episodes):
            samples = self.collect_episode(steps=steps_per_episode)
            total_samples += samples

            if (episode + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = total_samples / elapsed
                print(f"Episode {episode + 1}/{num_episodes} - "
                      f"Total samples: {total_samples} - "
                      f"Rate: {rate:.1f} samples/sec")

        duration = time.time() - start_time

        stats = {
            'num_episodes': num_episodes,
            'steps_per_episode': steps_per_episode,
            'total_samples': total_samples,
            'duration_seconds': duration,
            'samples_per_second': total_samples / duration,
            'output_dir': str(self.output_dir)
        }

        # Save collection stats
        with open(self.output_dir / 'collection_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\nCollection complete!")
        print(f"Total samples: {total_samples}")
        print(f"Duration: {duration:.1f}s")
        print(f"Rate: {stats['samples_per_second']:.1f} samples/sec")

        return stats


class MockLanguageDataCollector:
    """
    Mock data collector for testing without Isaac Sim.
    Generates synthetic images and metadata.
    """

    def __init__(
        self,
        output_dir: str = './dataset_language_mock'
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sample_count = 0

    def setup(self) -> None:
        print("Mock collector initialized (no simulation)")

    def cleanup(self) -> None:
        pass

    def collect_data(
        self,
        num_episodes: int = 100,
        steps_per_episode: int = 50
    ) -> Dict[str, Any]:
        """Generate mock training data."""
        print(f"\nGenerating mock data: {num_episodes} episodes")

        all_instructions = []
        for templates in LANGUAGE_TEMPLATES.values():
            all_instructions.extend(templates)

        # Add color variations
        for template in LANGUAGE_TEMPLATES['approach_color_object']:
            for color in COLORS:
                for obj in APPROACHABLE_OBJECTS:
                    all_instructions.append(template.format(color=color, object=obj))

        for template in LANGUAGE_TEMPLATES['approach_object']:
            for obj in APPROACHABLE_OBJECTS + ['door', 'person']:
                all_instructions.append(template.format(object=obj))

        for template in LANGUAGE_TEMPLATES['avoid_object']:
            for obj in AVOIDABLE_OBJECTS:
                all_instructions.append(template.format(object=obj))

        # Add speed modulation variations
        for template in LANGUAGE_TEMPLATES['speed_modulation']:
            if '{color}' in template:
                for color in COLORS:
                    for obj in APPROACHABLE_OBJECTS:
                        all_instructions.append(template.format(color=color, object=obj))
            else:
                for obj in APPROACHABLE_OBJECTS + ['door', 'person']:
                    all_instructions.append(template.format(object=obj))

        # Add conditional action variations
        for template in LANGUAGE_TEMPLATES['conditional_actions']:
            if '{color}' in template:
                for color in COLORS:
                    for obj in APPROACHABLE_OBJECTS:
                        all_instructions.append(template.format(color=color, object=obj))
            else:
                for obj in APPROACHABLE_OBJECTS + ['door', 'person', 'chair']:
                    all_instructions.append(template.format(object=obj))

        # Add complex navigation variations
        for template in LANGUAGE_TEMPLATES['complex_navigation']:
            if '{color}' in template:
                for color in COLORS:
                    for obj in APPROACHABLE_OBJECTS:
                        all_instructions.append(template.format(color=color, object=obj))
            else:
                for obj in APPROACHABLE_OBJECTS + ['door', 'person', 'chair']:
                    all_instructions.append(template.format(object=obj))

        total_samples = 0
        start_time = time.time()

        for episode in range(num_episodes):
            instruction = np.random.choice(all_instructions)

            for step in range(steps_per_episode):
                sample_id = str(uuid.uuid4())

                # Create synthetic image (colored noise)
                img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(self.output_dir / f"{sample_id}.jpg", quality=85)

                # Generate action based on instruction
                left, right = self._instruction_to_action(instruction)

                metadata = {
                    'instruction': instruction,
                    'action': {
                        'left_speed': float(left),
                        'right_speed': float(right)
                    },
                    'episode_step': step,
                    'timestamp': time.time(),
                    'source': 'mock'
                }

                with open(self.output_dir / f"{sample_id}.json", 'w') as f:
                    json.dump(metadata, f, indent=2)

                total_samples += 1

            if (episode + 1) % 20 == 0:
                print(f"Episode {episode + 1}/{num_episodes}")

        duration = time.time() - start_time

        stats = {
            'num_episodes': num_episodes,
            'total_samples': total_samples,
            'duration_seconds': duration,
            'output_dir': str(self.output_dir)
        }

        with open(self.output_dir / 'collection_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\nMock data generation complete: {total_samples} samples")
        return stats

    def _instruction_to_action(self, instruction: str) -> Tuple[float, float]:
        """Map instruction to action."""
        instruction_lower = instruction.lower()
        noise = np.random.normal(0, 0.1, 2)

        # Determine speed modifier
        speed_mult = self._get_speed_modifier(instruction_lower)

        # Check for speed modulation commands first
        if any(kw in instruction_lower for kw in ['speed up', 'fast', 'quickly', 'rush', 'accelerate', 'full speed']):
            base = (0.6 * speed_mult, 0.6 * speed_mult)
        elif any(kw in instruction_lower for kw in ['slow down', 'slowly', 'carefully', 'cautiously', 'decelerate']):
            base = (0.25 * speed_mult, 0.25 * speed_mult)
        elif 'forward' in instruction_lower or 'straight' in instruction_lower or 'towards' in instruction_lower:
            base = (0.5 * speed_mult, 0.5 * speed_mult)
        elif 'turn left' in instruction_lower:
            base = (0.15, 0.45)
        elif 'turn right' in instruction_lower:
            base = (0.45, 0.15)
        elif 'left' in instruction_lower:
            base = (0.2, 0.5)
        elif 'right' in instruction_lower:
            base = (0.5, 0.2)
        elif 'backward' in instruction_lower or 'back' in instruction_lower:
            base = (-0.3, -0.3)
        elif 'stop' in instruction_lower or 'halt' in instruction_lower:
            base = (0.0, 0.0)
        elif 'around' in instruction_lower or 'avoid' in instruction_lower or 'swerve' in instruction_lower:
            # Random avoidance direction
            if np.random.random() > 0.5:
                base = (0.3, 0.5)
            else:
                base = (0.5, 0.3)
        elif 'follow' in instruction_lower:
            base = (0.4, 0.4)
        elif 'approach' in instruction_lower or 'navigate' in instruction_lower or 'go to' in instruction_lower:
            base = (0.45 * speed_mult, 0.45 * speed_mult)
        else:
            base = (0.3, 0.3)

        return (
            np.clip(base[0] + noise[0], -1, 1),
            np.clip(base[1] + noise[1], -1, 1)
        )

    def _get_speed_modifier(self, instruction: str) -> float:
        """Extract speed modifier from instruction text."""
        # Fast keywords
        fast_keywords = ['speed up', 'fast', 'quickly', 'rush', 'accelerate', 'full speed']
        for kw in fast_keywords:
            if kw in instruction:
                return np.random.uniform(1.2, 1.5)

        # Slow keywords
        slow_keywords = ['slow down', 'slowly', 'carefully', 'cautiously', 'decelerate', 'move slowly']
        for kw in slow_keywords:
            if kw in instruction:
                return np.random.uniform(0.4, 0.7)

        return 1.0


def main():
    parser = argparse.ArgumentParser(
        description='Language-Grounded Data Collection for VLA Training'
    )

    parser.add_argument(
        '--config',
        default='multi_target',
        help='Scene configuration'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of episodes'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=50,
        help='Steps per episode'
    )
    parser.add_argument(
        '--output',
        default='./dataset_language',
        help='Output directory'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without GUI'
    )
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Use mock collector (no Isaac Sim)'
    )

    args = parser.parse_args()

    if args.mock:
        collector = MockLanguageDataCollector(output_dir=args.output)
    else:
        collector = LanguageDataCollector(
            output_dir=args.output,
            headless=args.headless,
            scene_config=args.config
        )

    try:
        collector.setup()
        collector.collect_data(
            num_episodes=args.episodes,
            steps_per_episode=args.steps
        )
    finally:
        collector.cleanup()


if __name__ == '__main__':
    main()

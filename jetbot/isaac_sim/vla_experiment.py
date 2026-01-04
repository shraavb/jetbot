"""
VLA Language Grounding Experiment Runner

Runs experiments to test VLA model responses to natural language commands
with colored objects in the scene.

Commands tested:
- "go towards the red ball"
- "follow the person in front of you"
- "go around the chair"
- "go towards the door"
- "navigate to the blue box"
- etc.

Usage:
    python -m jetbot.isaac_sim.vla_experiment --config multi_target --episodes 10

    # Or with custom instructions
    python -m jetbot.isaac_sim.vla_experiment \
        --instructions "go towards the red ball" "avoid the chair" \
        --episodes 5
"""

import time
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict


@dataclass
class ExperimentTrial:
    """Results from a single trial."""
    instruction: str
    target_object: Optional[str]
    target_position: Optional[List[float]]
    start_position: List[float]
    final_position: List[float]
    trajectory: List[List[float]]
    actions: List[Tuple[float, float]]
    num_steps: int
    duration_seconds: float
    distance_to_target: Optional[float]
    initial_distance: Optional[float]
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentResults:
    """Results from a complete experiment run."""
    experiment_name: str
    scene_config: str
    timestamp: str
    num_trials: int
    trials: List[ExperimentTrial]
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'experiment_name': self.experiment_name,
            'scene_config': self.scene_config,
            'timestamp': self.timestamp,
            'num_trials': self.num_trials,
            'trials': [asdict(t) for t in self.trials],
            'aggregate_metrics': self.aggregate_metrics
        }


# Standard test instructions organized by command type
STANDARD_INSTRUCTIONS = {
    'color_target': [
        "go towards the red ball",
        "go to the red ball",
        "navigate to the blue ball",
        "move towards the green ball",
        "approach the yellow box",
        "drive to the orange cone",
    ],
    'object_target': [
        "go towards the door",
        "navigate to the door",
        "approach the chair",
        "go to the chair",
        "move towards the person",
        "drive to the box",
    ],
    'person_following': [
        "follow the person in front of you",
        "follow the person",
        "go towards the person",
        "approach the person in front",
        "move towards the human",
    ],
    'avoidance': [
        "go around the chair",
        "avoid the obstacle",
        "navigate around the box",
        "go around the obstacle ahead",
        "circumvent the chair",
    ],
    'robot_avoidance': [
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
    ],
    'speed_modulation': [
        "speed up as you go towards the red ball",
        "slow down as you approach the door",
        "go fast towards the blue ball",
        "slowly approach the person",
        "quickly navigate to the green box",
        "carefully approach the chair",
        "rush towards the red ball",
        "cautiously move towards the door",
        "accelerate as you head to the ball",
        "decelerate near the obstacle",
        "go full speed to the target",
        "move slowly towards the person",
    ],
    'conditional_actions': [
        "turn left near the person in front of you",
        "turn right when you reach the chair",
        "slow down as you go around the red ball",
        "speed up after passing the obstacle",
        "stop when you reach the door",
        "turn around at the red ball",
        "go left after the chair",
        "turn right before the door",
        "slow down near the jetbot",
        "speed up once past the person",
        "stop near the blue ball",
        "turn left at the cone",
    ],
    'complex_navigation': [
        "go quickly towards the red ball then slow down",
        "speed up as you go around the chair",
        "slowly approach the door while avoiding the jetbot",
        "fast approach to the ball then stop",
        "carefully navigate around the person then speed up",
        "rush to the red ball but slow near the chair",
        "go slow until you pass the jetbot then speed up",
        "accelerate towards the door avoiding obstacles",
    ],
    'combined': [
        "go towards the red ball while avoiding the chair",
        "navigate to the door, avoiding obstacles",
        "reach the blue box but avoid the person",
        "go to the green ball around the obstacle",
        "go to the red ball while avoiding the jetbot",
        "navigate to the door around the other robot",
        "speed up as you go around the red ball",
        "slow down as you go towards the door",
        "turn left near the person in front of you",
    ],
    'directional': [
        "go forward",
        "turn left",
        "turn right",
        "go straight ahead",
        "rotate left",
        "rotate right",
        "go forward fast",
        "go forward slowly",
        "turn left slowly",
        "turn right quickly",
    ],
}


class VLAExperiment:
    """
    Runs VLA language grounding experiments in Isaac Sim.

    Tests the VLA model's ability to understand and execute natural language
    commands referencing objects by color and type.
    """

    def __init__(
        self,
        vla_server_host: str = 'localhost',
        vla_server_port: int = 5555,
        headless: bool = True,
        output_dir: str = './experiment_results'
    ):
        """
        Initialize experiment runner.

        Args:
            vla_server_host: VLA server hostname
            vla_server_port: VLA server port
            headless: Run simulation without GUI
            output_dir: Directory to save results
        """
        self.vla_host = vla_server_host
        self.vla_port = vla_server_port
        self.headless = headless
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._sim = None
        self._vla_client = None
        self._scene = None
        self._world = None

    def setup(self, scene_config: str = 'multi_target') -> None:
        """
        Setup experiment environment.

        Args:
            scene_config: Name of predefined scene config or 'custom'
        """
        # Import here to avoid issues when Isaac Sim not available
        from .jetbot_sim import JetBotSim
        from .experiment_scene import ExperimentScene, SCENE_CONFIGS, load_scene_from_config

        try:
            from isaacsim import SimulationApp

            # Initialize simulation
            self._simulation_app = SimulationApp({
                "headless": self.headless,
                "width": 1280,
                "height": 720
            })

            from omni.isaac.core import World

            # Create world
            self._world = World(physics_dt=1/60.0, rendering_dt=1/60.0)

            # Setup experiment scene
            self._scene = ExperimentScene()
            self._scene.setup(self._world)

            # Load scene configuration
            if scene_config in SCENE_CONFIGS:
                config = SCENE_CONFIGS[scene_config]
                load_scene_from_config(self._scene, config)
                print(f"Loaded scene config: {scene_config}")
                print(f"Scene description: {self._scene.get_scene_description()}")
            else:
                print(f"Using empty scene. Add objects manually.")

            # Create JetBot (using existing sim code pattern)
            self._setup_jetbot()

            # Reset world to initialize
            self._world.reset()

            # Connect to VLA server
            self._connect_vla()

            print("Experiment setup complete")

        except ImportError as e:
            raise ImportError(
                f"Isaac Sim not available. Please install Isaac Sim. Error: {e}"
            )

    def _setup_jetbot(self) -> None:
        """Setup JetBot robot in the scene."""
        from omni.isaac.wheeled_robots.robots import WheeledRobot
        from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
        from omni.isaac.sensor import Camera

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

        # Controller
        self._controller = DifferentialController(
            name="jetbot_controller",
            wheel_radius=0.03,
            wheel_base=0.1125
        )

        # Camera
        self._camera = Camera(
            prim_path="/World/JetBot/chassis/camera",
            resolution=(224, 224)
        )
        self._camera.initialize()

    def _connect_vla(self) -> None:
        """Connect to VLA server."""
        try:
            from jetbot.vla.vla_client import VLAClient

            self._vla_client = VLAClient(
                server_host=self.vla_host,
                server_port=self.vla_port
            )
            print(f"Connected to VLA server at {self.vla_host}:{self.vla_port}")
        except ImportError:
            print("Warning: VLAClient not available, running in data collection mode only")
            self._vla_client = None

    def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, '_simulation_app') and self._simulation_app:
            self._simulation_app.close()
        self._sim = None
        self._vla_client = None

    def run_trial(
        self,
        instruction: str,
        max_steps: int = 200,
        success_threshold: float = 0.3
    ) -> ExperimentTrial:
        """
        Run a single experiment trial.

        Args:
            instruction: Natural language instruction
            max_steps: Maximum steps for the trial
            success_threshold: Distance threshold for success (meters)

        Returns:
            Trial results
        """
        # Reset robot position
        self._world.reset()

        # Parse instruction to find target object
        target_obj = self._find_target_object(instruction)
        target_position = target_obj.position.tolist() if target_obj else None

        # Get initial state
        start_pos, _ = self._jetbot.get_world_pose()
        start_position = start_pos.tolist()

        initial_distance = None
        if target_position:
            initial_distance = np.linalg.norm(
                np.array(start_position[:2]) - np.array(target_position[:2])
            )

        # Set instruction
        if self._vla_client:
            self._vla_client.instruction = instruction

        # Run trial
        trajectory = []
        actions = []
        start_time = time.time()

        for step in range(max_steps):
            # Get camera image
            rgba = self._camera.get_rgba()
            rgb = rgba[:, :, :3]

            # Get VLA action
            if self._vla_client:
                try:
                    left, right = self._vla_client.predict(rgb)
                except Exception as e:
                    print(f"VLA prediction failed at step {step}: {e}")
                    left, right = 0.0, 0.0
            else:
                # Random exploration if no VLA
                left = np.random.uniform(-0.5, 0.5)
                right = np.random.uniform(-0.5, 0.5)

            # Apply action
            linear = (left + right) / 2.0 * 0.3
            angular = (right - left) / 0.1125 * 0.3

            wheel_actions = self._controller.forward(command=[linear, angular])
            self._jetbot.apply_wheel_actions(wheel_actions)

            # Step simulation
            self._world.step(render=not self.headless)

            # Record trajectory
            pos, _ = self._jetbot.get_world_pose()
            trajectory.append(pos.tolist())
            actions.append((left, right))

            # Check if reached target
            if target_position:
                dist = np.linalg.norm(
                    np.array(pos[:2]) - np.array(target_position[:2])
                )
                if dist < success_threshold:
                    print(f"Reached target in {step + 1} steps")
                    break

        duration = time.time() - start_time

        # Get final state
        final_pos, _ = self._jetbot.get_world_pose()
        final_position = final_pos.tolist()

        # Calculate final distance to target
        final_distance = None
        if target_position:
            final_distance = np.linalg.norm(
                np.array(final_position[:2]) - np.array(target_position[:2])
            )

        # Determine success
        success = False
        if target_position and final_distance is not None:
            success = final_distance < success_threshold

        # Calculate metrics
        metrics = self._calculate_trial_metrics(
            trajectory, actions, target_position, initial_distance, final_distance
        )

        return ExperimentTrial(
            instruction=instruction,
            target_object=target_obj.name if target_obj else None,
            target_position=target_position,
            start_position=start_position,
            final_position=final_position,
            trajectory=trajectory,
            actions=actions,
            num_steps=len(trajectory),
            duration_seconds=duration,
            distance_to_target=final_distance,
            initial_distance=initial_distance,
            success=success,
            metrics=metrics
        )

    def _find_target_object(self, instruction: str):
        """Parse instruction to find the target object in the scene."""
        instruction_lower = instruction.lower()

        # Check for color + object combinations
        from .experiment_scene import ObjectColor

        for color in ObjectColor:
            color_name = color.name.lower()
            if color_name in instruction_lower:
                # Found a color reference
                colored_objects = self._scene.get_objects_by_color(color)
                if colored_objects:
                    # Check if object type is also mentioned
                    for obj in colored_objects:
                        if obj.object_type in instruction_lower:
                            return obj
                    # Return first matching color
                    return colored_objects[0]

        # Check for object type references (including jetbot/robot)
        object_types = ['ball', 'box', 'chair', 'door', 'person', 'cone', 'wall', 'jetbot', 'robot']
        for obj_type in object_types:
            if obj_type in instruction_lower:
                # 'robot' should also match 'jetbot' type
                search_type = 'jetbot' if obj_type == 'robot' else obj_type
                objects = self._scene.get_objects_by_type(search_type)
                if objects:
                    return objects[0]

        return None

    def _calculate_trial_metrics(
        self,
        trajectory: List[List[float]],
        actions: List[Tuple[float, float]],
        target_position: Optional[List[float]],
        initial_distance: Optional[float],
        final_distance: Optional[float]
    ) -> Dict[str, float]:
        """Calculate performance metrics for a trial."""
        metrics = {}

        # Path length
        path_length = 0.0
        for i in range(1, len(trajectory)):
            path_length += np.linalg.norm(
                np.array(trajectory[i][:2]) - np.array(trajectory[i-1][:2])
            )
        metrics['path_length'] = path_length

        # Progress towards target
        if initial_distance is not None and final_distance is not None:
            progress = initial_distance - final_distance
            metrics['progress'] = progress
            metrics['progress_ratio'] = progress / initial_distance if initial_distance > 0 else 0

        # Path efficiency (straight-line distance / path length)
        if target_position and len(trajectory) > 0:
            straight_line = np.linalg.norm(
                np.array(trajectory[-1][:2]) - np.array(trajectory[0][:2])
            )
            if path_length > 0:
                metrics['path_efficiency'] = straight_line / path_length

        # Action statistics
        if actions:
            left_speeds = [a[0] for a in actions]
            right_speeds = [a[1] for a in actions]
            metrics['mean_left_speed'] = float(np.mean(left_speeds))
            metrics['mean_right_speed'] = float(np.mean(right_speeds))
            metrics['speed_variance'] = float(np.var(left_speeds) + np.var(right_speeds))

        return metrics

    def run_experiment(
        self,
        instructions: List[str],
        episodes_per_instruction: int = 1,
        max_steps: int = 200,
        experiment_name: str = "vla_language_grounding"
    ) -> ExperimentResults:
        """
        Run a complete experiment with multiple instructions.

        Args:
            instructions: List of instructions to test
            episodes_per_instruction: Number of trials per instruction
            max_steps: Maximum steps per trial
            experiment_name: Name for this experiment

        Returns:
            Experiment results
        """
        trials = []
        total_trials = len(instructions) * episodes_per_instruction

        print(f"\n{'='*60}")
        print(f"Running experiment: {experiment_name}")
        print(f"Instructions: {len(instructions)}")
        print(f"Episodes per instruction: {episodes_per_instruction}")
        print(f"Total trials: {total_trials}")
        print(f"{'='*60}\n")

        trial_num = 0
        for instruction in instructions:
            for episode in range(episodes_per_instruction):
                trial_num += 1
                print(f"\n[{trial_num}/{total_trials}] Instruction: '{instruction}'")

                trial = self.run_trial(instruction, max_steps=max_steps)
                trials.append(trial)

                # Print trial summary
                status = "✓ SUCCESS" if trial.success else "✗ FAILED"
                print(f"  {status} - Steps: {trial.num_steps}, "
                      f"Final distance: {trial.distance_to_target:.3f}m" if trial.distance_to_target else "")

        # Calculate aggregate metrics
        aggregate = self._calculate_aggregate_metrics(trials)

        results = ExperimentResults(
            experiment_name=experiment_name,
            scene_config="custom",
            timestamp=datetime.now().isoformat(),
            num_trials=len(trials),
            trials=trials,
            aggregate_metrics=aggregate
        )

        # Print summary
        self._print_experiment_summary(results)

        return results

    def _calculate_aggregate_metrics(self, trials: List[ExperimentTrial]) -> Dict[str, float]:
        """Calculate aggregate metrics across all trials."""
        if not trials:
            return {}

        success_count = sum(1 for t in trials if t.success)
        metrics = {
            'success_rate': success_count / len(trials),
            'num_successes': success_count,
            'num_failures': len(trials) - success_count,
        }

        # Average metrics
        all_metrics = [t.metrics for t in trials]
        metric_keys = set()
        for m in all_metrics:
            metric_keys.update(m.keys())

        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                metrics[f'mean_{key}'] = float(np.mean(values))
                metrics[f'std_{key}'] = float(np.std(values))

        # Distance metrics
        distances = [t.distance_to_target for t in trials if t.distance_to_target is not None]
        if distances:
            metrics['mean_final_distance'] = float(np.mean(distances))
            metrics['min_final_distance'] = float(np.min(distances))
            metrics['max_final_distance'] = float(np.max(distances))

        return metrics

    def _print_experiment_summary(self, results: ExperimentResults) -> None:
        """Print experiment summary."""
        print(f"\n{'='*60}")
        print(f"EXPERIMENT SUMMARY: {results.experiment_name}")
        print(f"{'='*60}")
        print(f"Total trials: {results.num_trials}")
        print(f"Success rate: {results.aggregate_metrics.get('success_rate', 0):.1%}")
        print(f"Successes: {results.aggregate_metrics.get('num_successes', 0)}")
        print(f"Failures: {results.aggregate_metrics.get('num_failures', 0)}")

        if 'mean_final_distance' in results.aggregate_metrics:
            print(f"Mean final distance: {results.aggregate_metrics['mean_final_distance']:.3f}m")

        if 'mean_progress_ratio' in results.aggregate_metrics:
            print(f"Mean progress ratio: {results.aggregate_metrics['mean_progress_ratio']:.1%}")

        print(f"{'='*60}\n")

    def save_results(self, results: ExperimentResults, filename: Optional[str] = None) -> str:
        """
        Save experiment results to JSON file.

        Args:
            results: Experiment results
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{results.experiment_name}_{timestamp}.json"

        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)

        print(f"Results saved to: {filepath}")
        return str(filepath)

    def run_standard_tests(
        self,
        categories: Optional[List[str]] = None,
        episodes_per_instruction: int = 1
    ) -> ExperimentResults:
        """
        Run standard test suite.

        Args:
            categories: List of instruction categories to test
                       (default: all categories)
            episodes_per_instruction: Number of trials per instruction

        Returns:
            Experiment results
        """
        if categories is None:
            categories = list(STANDARD_INSTRUCTIONS.keys())

        instructions = []
        for category in categories:
            if category in STANDARD_INSTRUCTIONS:
                instructions.extend(STANDARD_INSTRUCTIONS[category])
            else:
                print(f"Warning: Unknown category '{category}'")

        return self.run_experiment(
            instructions=instructions,
            episodes_per_instruction=episodes_per_instruction,
            experiment_name="standard_language_tests"
        )


def main():
    """Main entry point for running experiments."""
    parser = argparse.ArgumentParser(
        description='VLA Language Grounding Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with predefined scene config
  python -m jetbot.isaac_sim.vla_experiment --config multi_target

  # Run standard test suite
  python -m jetbot.isaac_sim.vla_experiment --standard-tests

  # Run specific instruction categories
  python -m jetbot.isaac_sim.vla_experiment --categories color_target object_target

  # Run custom instructions
  python -m jetbot.isaac_sim.vla_experiment \\
      --instructions "go towards the red ball" "avoid the chair"
        """
    )

    parser.add_argument(
        '--config',
        choices=['color_navigation', 'obstacle_course', 'room_navigation', 'multi_target'],
        default='multi_target',
        help='Predefined scene configuration'
    )
    parser.add_argument(
        '--instructions',
        nargs='+',
        help='Custom instructions to test'
    )
    parser.add_argument(
        '--categories',
        nargs='+',
        choices=list(STANDARD_INSTRUCTIONS.keys()),
        help='Standard instruction categories to test'
    )
    parser.add_argument(
        '--standard-tests',
        action='store_true',
        help='Run full standard test suite'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=1,
        help='Episodes per instruction'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=200,
        help='Maximum steps per trial'
    )
    parser.add_argument(
        '--output-dir',
        default='./experiment_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--vla-host',
        default='localhost',
        help='VLA server host'
    )
    parser.add_argument(
        '--vla-port',
        type=int,
        default=5555,
        help='VLA server port'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run simulation without GUI'
    )
    parser.add_argument(
        '--name',
        default='vla_language_grounding',
        help='Experiment name'
    )

    args = parser.parse_args()

    # Create experiment runner
    experiment = VLAExperiment(
        vla_server_host=args.vla_host,
        vla_server_port=args.vla_port,
        headless=args.headless,
        output_dir=args.output_dir
    )

    try:
        # Setup scene
        experiment.setup(scene_config=args.config)

        # Determine which instructions to run
        if args.instructions:
            # Custom instructions
            results = experiment.run_experiment(
                instructions=args.instructions,
                episodes_per_instruction=args.episodes,
                max_steps=args.max_steps,
                experiment_name=args.name
            )
        elif args.categories:
            # Specific categories
            results = experiment.run_standard_tests(
                categories=args.categories,
                episodes_per_instruction=args.episodes
            )
        elif args.standard_tests:
            # Full standard test suite
            results = experiment.run_standard_tests(
                episodes_per_instruction=args.episodes
            )
        else:
            # Default: run a few example instructions
            default_instructions = [
                "go towards the red ball",
                "navigate to the door",
                "follow the person in front of you",
                "go around the chair",
            ]
            results = experiment.run_experiment(
                instructions=default_instructions,
                episodes_per_instruction=args.episodes,
                max_steps=args.max_steps,
                experiment_name=args.name
            )

        # Save results
        experiment.save_results(results)

    finally:
        experiment.cleanup()


if __name__ == '__main__':
    main()

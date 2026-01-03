"""
VLA-Simulation Interface

Connects the VLA model with Isaac Sim for:
1. Testing VLA models in simulation
2. Collecting synthetic training data
3. Reinforcement learning with VLA

Usage:
    from jetbot.isaac_sim import VLASimInterface

    interface = VLASimInterface(
        vla_server_host='localhost',
        vla_server_port=5555
    )

    # Run VLA-guided navigation in simulation
    interface.run_vla_navigation(instruction="go to the red box")

    # Collect training data
    interface.collect_data(num_episodes=100, save_dir='./sim_data')
"""

import time
import json
import uuid
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable
from PIL import Image

from .jetbot_sim import JetBotSim


class VLASimInterface:
    """
    Interface between VLA model and Isaac Sim.

    Enables testing VLA models in simulation and collecting
    synthetic training data.
    """

    def __init__(
        self,
        vla_server_host: str = 'localhost',
        vla_server_port: int = 5555,
        headless: bool = True,
        scene: str = 'simple_room'
    ):
        """
        Initialize VLA-Sim interface.

        Args:
            vla_server_host: VLA server hostname
            vla_server_port: VLA server port
            headless: Run simulation without GUI
            scene: Scene to load in simulation
        """
        self.vla_host = vla_server_host
        self.vla_port = vla_server_port
        self.headless = headless
        self.scene = scene

        self._sim: Optional[JetBotSim] = None
        self._vla_client = None

    def _ensure_sim_running(self) -> None:
        """Ensure simulation is running."""
        if self._sim is None or not self._sim.is_running:
            self._sim = JetBotSim(headless=self.headless)
            self._sim.start(scene=self.scene)

    def _ensure_vla_connected(self) -> None:
        """Ensure VLA client is connected."""
        if self._vla_client is None:
            try:
                from jetbot.vla.vla_client import VLAClient

                self._vla_client = VLAClient(
                    server_host=self.vla_host,
                    server_port=self.vla_port
                )
            except ImportError:
                raise ImportError(
                    "VLAClient not found. Ensure jetbot.vla module is available."
                )

    def start(self) -> None:
        """Start simulation and connect to VLA server."""
        self._ensure_sim_running()
        self._ensure_vla_connected()

    def stop(self) -> None:
        """Stop simulation and disconnect."""
        if self._sim is not None:
            self._sim.stop()
            self._sim = None

        if self._vla_client is not None:
            self._vla_client = None

    def run_vla_navigation(
        self,
        instruction: str,
        max_steps: int = 500,
        step_delay: float = 0.0,
        on_step: Optional[Callable[[int, np.ndarray, Tuple[float, float]], None]] = None
    ) -> Dict:
        """
        Run VLA-guided navigation in simulation.

        Args:
            instruction: Natural language instruction
            max_steps: Maximum simulation steps
            step_delay: Delay between steps (for visualization)
            on_step: Optional callback(step, image, action) called each step

        Returns:
            Dictionary with navigation results
        """
        self._ensure_sim_running()
        self._ensure_vla_connected()

        results = {
            'instruction': instruction,
            'num_steps': 0,
            'positions': [],
            'actions': [],
            'success': False
        }

        # Set instruction on client
        self._vla_client.instruction = instruction

        for step in range(max_steps):
            # Get camera image (VLA client expects BGR numpy array)
            image = self._sim.get_camera_image()

            # Query VLA for action
            try:
                left, right = self._vla_client.predict(image)
            except Exception as e:
                print(f"VLA prediction failed: {e}")
                break

            # Apply action
            self._sim.set_velocity(left, right)

            # Step simulation
            self._sim.step()

            # Record data
            position = self._sim.get_position()
            results['positions'].append(position.tolist())
            results['actions'].append((left, right))
            results['num_steps'] = step + 1

            # Callback
            if on_step:
                on_step(step, self._sim.get_camera_image(), (left, right))

            # Delay for visualization
            if step_delay > 0:
                time.sleep(step_delay)

        results['final_position'] = self._sim.get_position().tolist()
        return results

    def collect_data(
        self,
        num_episodes: int = 100,
        steps_per_episode: int = 50,
        save_dir: str = './sim_data',
        instructions: Optional[List[str]] = None,
        policy: str = 'random'
    ) -> str:
        """
        Collect synthetic training data in simulation.

        Args:
            num_episodes: Number of episodes to collect
            steps_per_episode: Steps per episode
            save_dir: Directory to save data
            instructions: List of instructions to use (random selection)
            policy: Control policy ('random', 'scripted', 'keyboard')

        Returns:
            Path to saved dataset
        """
        self._ensure_sim_running()

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if instructions is None:
            instructions = self._default_instructions()

        total_samples = 0

        for episode in range(num_episodes):
            # Reset simulation
            self._sim.reset()

            # Select random instruction
            instruction = np.random.choice(instructions)

            for step in range(steps_per_episode):
                # Get observation
                image = self._sim.get_camera_image()

                # Get action based on policy
                action = self._get_policy_action(policy, instruction)

                # Save sample
                sample_id = str(uuid.uuid4())

                # Save image
                img_pil = Image.fromarray(image)
                img_pil.save(save_path / f"{sample_id}.jpg", quality=95)

                # Save metadata
                metadata = {
                    'instruction': instruction,
                    'action': {
                        'left_speed': action[0],
                        'right_speed': action[1]
                    },
                    'episode': episode,
                    'step': step,
                    'timestamp': time.time(),
                    'source': 'isaac_sim'
                }
                with open(save_path / f"{sample_id}.json", 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Apply action and step
                self._sim.set_velocity(action[0], action[1])
                self._sim.step()

                total_samples += 1

            print(f"Episode {episode+1}/{num_episodes} complete "
                  f"({total_samples} samples)")

        print(f"Data collection complete: {total_samples} samples saved to {save_path}")
        return str(save_path)

    def _get_policy_action(
        self,
        policy: str,
        instruction: str
    ) -> Tuple[float, float]:
        """Get action from the specified policy."""

        if policy == 'random':
            # Random exploration
            left = np.random.uniform(-1.0, 1.0)
            right = np.random.uniform(-1.0, 1.0)

        elif policy == 'scripted':
            # Scripted policy based on instruction
            action = self._scripted_policy(instruction)
            left, right = action

        elif policy == 'keyboard':
            # Would require GUI input handling
            raise NotImplementedError("Keyboard policy not implemented")

        else:
            raise ValueError(f"Unknown policy: {policy}")

        return (left, right)

    def _scripted_policy(self, instruction: str) -> Tuple[float, float]:
        """Simple scripted policy based on instruction keywords."""
        instruction = instruction.lower()

        # Add some noise to make data more realistic
        noise = np.random.normal(0, 0.1, 2)

        if 'forward' in instruction or 'straight' in instruction:
            action = (0.5, 0.5)
        elif 'left' in instruction:
            action = (0.2, 0.5)
        elif 'right' in instruction:
            action = (0.5, 0.2)
        elif 'backward' in instruction or 'back' in instruction:
            action = (-0.3, -0.3)
        elif 'stop' in instruction:
            action = (0.0, 0.0)
        elif 'avoid' in instruction or 'around' in instruction:
            # Random turn direction
            if np.random.random() > 0.5:
                action = (0.3, 0.5)
            else:
                action = (0.5, 0.3)
        else:
            # Default: slow forward
            action = (0.3, 0.3)

        # Add noise
        return (
            np.clip(action[0] + noise[0], -1.0, 1.0),
            np.clip(action[1] + noise[1], -1.0, 1.0)
        )

    def _default_instructions(self) -> List[str]:
        """Return default set of instructions for data collection."""
        return [
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

    def test_vla_model(
        self,
        test_instructions: Optional[List[str]] = None,
        steps_per_test: int = 100
    ) -> Dict:
        """
        Test VLA model performance in simulation.

        Args:
            test_instructions: Instructions to test
            steps_per_test: Steps per test episode

        Returns:
            Test results dictionary
        """
        self._ensure_sim_running()
        self._ensure_vla_connected()

        if test_instructions is None:
            test_instructions = [
                "go forward",
                "turn left",
                "turn right",
                "stop"
            ]

        results = {
            'tests': [],
            'total_tests': len(test_instructions)
        }

        for instruction in test_instructions:
            print(f"Testing: '{instruction}'")
            self._sim.reset()

            test_result = self.run_vla_navigation(
                instruction=instruction,
                max_steps=steps_per_test
            )

            results['tests'].append({
                'instruction': instruction,
                'result': test_result
            })

        return results


def main():
    """Example usage of VLA-Sim interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='VLA-Simulation Interface'
    )
    parser.add_argument(
        '--mode',
        choices=['navigate', 'collect', 'test'],
        default='collect',
        help='Operation mode'
    )
    parser.add_argument(
        '--instruction',
        default='go forward',
        help='Navigation instruction'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of episodes for data collection'
    )
    parser.add_argument(
        '--output',
        default='./sim_data',
        help='Output directory for data'
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
    args = parser.parse_args()

    interface = VLASimInterface(
        vla_server_host=args.vla_host,
        vla_server_port=args.vla_port,
        headless=args.headless
    )

    try:
        interface.start()

        if args.mode == 'navigate':
            result = interface.run_vla_navigation(args.instruction)
            print(f"Navigation result: {result}")

        elif args.mode == 'collect':
            path = interface.collect_data(
                num_episodes=args.episodes,
                save_dir=args.output,
                policy='scripted'
            )
            print(f"Data saved to: {path}")

        elif args.mode == 'test':
            results = interface.test_vla_model()
            print(f"Test results: {results}")

    finally:
        interface.stop()


if __name__ == '__main__':
    main()

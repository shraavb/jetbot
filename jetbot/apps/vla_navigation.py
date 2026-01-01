import argparse
import traitlets
from jetbot import Robot, Camera
from jetbot.vla import VLAClient
import time
import signal


class VLANavigationApplication(traitlets.HasTraits):
    """
    Main application for VLA-guided navigation.

    Uses OpenVLA running on a remote GPU server to interpret natural language
    instructions and control the JetBot's motors for navigation tasks.

    Follows the camera.observe() pattern from wander.py for real-time control.

    Example usage:
        python vla_navigation.py --instruction "go to the red ball" \\
                                 --server_host 192.168.1.100

    Safety features:
        - Stops robot on connection timeout
        - Speed limiting (default max 50%)
        - Graceful shutdown on SIGINT
    """

    instruction = traitlets.Unicode(
        default_value="navigate forward avoiding obstacles"
    )
    server_host = traitlets.Unicode(default_value="localhost")
    server_port = traitlets.Integer(default_value=5555)

    # Safety parameters
    max_speed = traitlets.Float(default_value=0.5)
    connection_timeout_sec = traitlets.Float(default_value=1.0)

    def __init__(self, *args, **kwargs):
        super(VLANavigationApplication, self).__init__(*args, **kwargs)
        self._last_command_time = time.time()
        self._running = False

    def _update(self, change):
        """Callback for camera frame updates."""
        if not self._running:
            return

        image = change['new']

        # Get VLA prediction
        left_speed, right_speed = self.vla_client.predict(image)

        # Safety check: stop if connection lost
        if not self.vla_client.connected:
            time_since_command = time.time() - self._last_command_time
            if time_since_command > self.connection_timeout_sec:
                self.robot.stop()
                if int(time_since_command) % 5 == 0:  # Print every 5 seconds
                    print(f"Connection lost for {time_since_command:.1f}s - stopped")
                return
        else:
            self._last_command_time = time.time()

        # Apply speed limit
        left_speed = max(-self.max_speed, min(self.max_speed, left_speed))
        right_speed = max(-self.max_speed, min(self.max_speed, right_speed))

        # Set motor speeds
        self.robot.set_motors(left_speed, right_speed)

    def start(self):
        """Initialize components and start VLA-guided navigation."""
        print(f'Connecting to VLA server at {self.server_host}:{self.server_port}...')
        self.vla_client = VLAClient(
            server_host=self.server_host,
            server_port=self.server_port
        )
        self.vla_client.instruction = self.instruction

        print('Initializing robot...')
        self.robot = Robot()

        print('Initializing camera...')
        self.camera = Camera.instance(width=224, height=224)

        print(f'Instruction: "{self.instruction}"')
        print(f'Max speed: {self.max_speed}')
        print('Running... (Ctrl+C to stop)')

        self._running = True
        self.camera.observe(self._update, names='value')

        def shutdown(sig, frame):
            print('\nShutting down...')
            self._running = False
            self.robot.stop()
            self.camera.stop()
            self.vla_client.close()

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        # Wait for camera thread
        self.camera.thread.join()

    def stop(self):
        """Stop the application and clean up resources."""
        self._running = False
        if hasattr(self, 'robot'):
            self.robot.stop()
        if hasattr(self, 'camera'):
            self.camera.stop()
        if hasattr(self, 'vla_client'):
            self.vla_client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='VLA-guided navigation for JetBot',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--instruction',
        default='navigate forward avoiding obstacles',
        help='Natural language instruction for the robot'
    )
    parser.add_argument(
        '--server_host',
        default='localhost',
        help='VLA server hostname or IP address'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=5555,
        help='VLA server port'
    )
    parser.add_argument(
        '--max_speed',
        type=float,
        default=0.5,
        help='Maximum motor speed (0.0 to 1.0)'
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=1.0,
        help='Connection timeout in seconds before stopping'
    )
    args = parser.parse_args()

    application = VLANavigationApplication(
        instruction=args.instruction,
        server_host=args.server_host,
        server_port=args.server_port,
        max_speed=args.max_speed,
        connection_timeout_sec=args.timeout
    )
    application.start()

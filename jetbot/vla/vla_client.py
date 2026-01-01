import time
import traitlets
from traitlets.config.configurable import Configurable
import zmq
import numpy as np
import cv2


class VLAClient(Configurable):
    """
    Client for remote VLA (Vision-Language-Action) inference.

    Connects to a remote GPU server running OpenVLA and sends camera frames
    with natural language instructions, receiving motor commands in return.

    Follows JetBot's traitlets pattern for reactive state updates.

    Example usage:
        client = VLAClient(server_host='192.168.1.100', server_port=5555)
        client.instruction = "go to the red ball"
        left_speed, right_speed = client.predict(camera_frame)
        robot.set_motors(left_speed, right_speed)
    """

    # Reactive state traits
    instruction = traitlets.Unicode(
        default_value="navigate forward avoiding obstacles"
    ).tag(config=True)
    left_speed = traitlets.Float(default_value=0.0)
    right_speed = traitlets.Float(default_value=0.0)
    connected = traitlets.Bool(default_value=False)
    latency_ms = traitlets.Float(default_value=0.0)

    # Connection configuration
    server_host = traitlets.Unicode(default_value="localhost").tag(config=True)
    server_port = traitlets.Integer(default_value=5555).tag(config=True)
    timeout_ms = traitlets.Integer(default_value=500).tag(config=True)
    jpeg_quality = traitlets.Integer(default_value=85).tag(config=True)

    def __init__(self, *args, **kwargs):
        super(VLAClient, self).__init__(*args, **kwargs)
        self._context = None
        self._socket = None
        self._last_action_time = time.time()
        self._connect()

    def _connect(self):
        """Establish ZMQ connection to VLA server."""
        if self._context is not None:
            self.close()

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{self.server_host}:{self.server_port}")
        self.connected = True

    def predict(self, image):
        """
        Send image to VLA server and receive motor commands.

        Args:
            image: BGR uint8 numpy array from camera (typically 224x224x3)

        Returns:
            Tuple of (left_speed, right_speed) in range [-1.0, 1.0].
            Returns (0.0, 0.0) on connection error (safe stop).
        """
        try:
            # Encode image as JPEG for efficient transfer
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            _, jpeg = cv2.imencode('.jpg', image, encode_params)

            # Send multipart message: [image_bytes, instruction]
            start_time = time.time()
            self._socket.send_multipart([
                jpeg.tobytes(),
                self.instruction.encode('utf-8')
            ])

            # Receive response
            response = self._socket.recv_multipart()
            self.latency_ms = (time.time() - start_time) * 1000

            # Check for error response
            if response[0] == b'ERROR':
                error_msg = response[1].decode('utf-8') if len(response) > 1 else 'Unknown error'
                print(f"VLA server error: {error_msg}")
                self.connected = False
                self.left_speed = 0.0
                self.right_speed = 0.0
                return 0.0, 0.0

            # Parse motor commands
            left_speed = float(response[0].decode('utf-8'))
            right_speed = float(response[1].decode('utf-8'))

            # Update state
            self.left_speed = left_speed
            self.right_speed = right_speed
            self._last_action_time = time.time()
            self.connected = True

            return left_speed, right_speed

        except zmq.error.Again:
            # Timeout - return safe stop command
            print("VLA server timeout")
            self.connected = False
            self.left_speed = 0.0
            self.right_speed = 0.0
            return 0.0, 0.0

        except zmq.error.ZMQError as e:
            print(f"VLA ZMQ error: {e}")
            self.connected = False
            self.left_speed = 0.0
            self.right_speed = 0.0
            # Attempt to reconnect
            self._reconnect()
            return 0.0, 0.0

    def _reconnect(self):
        """Attempt to reconnect to server."""
        try:
            self._socket.close()
            self._socket = self._context.socket(zmq.REQ)
            self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.connect(f"tcp://{self.server_host}:{self.server_port}")
        except Exception as e:
            print(f"VLA reconnect failed: {e}")

    @property
    def time_since_last_action(self):
        """Time in seconds since last successful action."""
        return time.time() - self._last_action_time

    def close(self):
        """Close ZMQ connection and clean up resources."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        self.connected = False

    def __del__(self):
        self.close()

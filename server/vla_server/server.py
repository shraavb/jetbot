import zmq
import time
import io
import argparse
from PIL import Image
from .openvla_wrapper import OpenVLAWrapper


class VLAServer:
    """
    ZMQ server for VLA inference.

    Receives images and instructions from JetBot clients over ZMQ,
    runs OpenVLA inference, and returns motor commands.

    Protocol:
        Request: [jpeg_bytes, instruction_utf8]
        Response: [left_speed_str, right_speed_str] or [b'ERROR', error_msg]

    Example usage:
        server = VLAServer(port=5555)
        server.run()  # Blocks and serves requests
    """

    def __init__(
        self,
        port: int = 5555,
        model_id: str = "openvla/openvla-7b",
        fine_tuned: bool = False,
        device: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False
    ):
        """
        Initialize VLA server.

        Args:
            port: ZMQ port to listen on
            model_id: HuggingFace model ID or path to fine-tuned model
            fine_tuned: Whether using a fine-tuned JetBot model
            device: Device to run on ('auto', 'cuda', 'mps', 'cpu')
            load_in_4bit: Use 4-bit quantization (CUDA only)
            load_in_8bit: Use 8-bit quantization (CUDA only)
        """
        self.port = port
        self.model_id = model_id
        self.fine_tuned = fine_tuned

        # Statistics
        self.request_count = 0
        self.total_inference_time = 0.0

        # Initialize model
        print(f"Initializing VLA server on port {port}", flush=True)
        self.vla = OpenVLAWrapper(
            model_id=model_id,
            device=device,
            fine_tuned=fine_tuned,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit
        )

        # Initialize ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        print(f"VLA server listening on port {port}", flush=True)

    def process_request(self, request):
        """
        Process a single inference request.

        Args:
            request: ZMQ multipart message [jpeg_bytes, instruction_utf8]

        Returns:
            Response message [left_speed_str, right_speed_str]
        """
        try:
            # Parse request
            jpeg_bytes = request[0]
            instruction = request[1].decode('utf-8')
            print(f"Received request: '{instruction}' ({len(jpeg_bytes)} bytes)", flush=True)

            # Decode image
            image = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')

            # Run inference
            print("Running inference...", flush=True)
            start_time = time.time()
            left_speed, right_speed = self.vla.predict(image, instruction)
            inference_time = time.time() - start_time
            print(f"Inference completed in {inference_time:.2f}s", flush=True)

            # Update statistics
            self.request_count += 1
            self.total_inference_time += inference_time

            # Log periodically
            if self.request_count % 100 == 0:
                avg_time = self.total_inference_time / self.request_count
                hz = 1.0 / avg_time if avg_time > 0 else 0
                print(f"Requests: {self.request_count}, "
                      f"Avg inference: {avg_time*1000:.1f}ms ({hz:.1f} Hz)")

            # Return response
            return [
                f"{left_speed:.6f}".encode('utf-8'),
                f"{right_speed:.6f}".encode('utf-8')
            ]

        except Exception as e:
            print(f"Error processing request: {e}")
            return [b'ERROR', str(e).encode('utf-8')]

    def run(self):
        """
        Run the server main loop.

        Blocks and processes requests until interrupted.
        """
        print("VLA server running. Press Ctrl+C to stop.", flush=True)
        try:
            while True:
                # Wait for request
                request = self.socket.recv_multipart()

                # Process and respond
                response = self.process_request(request)
                self.socket.send_multipart(response)

        except KeyboardInterrupt:
            print("\nShutting down VLA server...")
        finally:
            self.socket.close()
            self.context.term()
            print("VLA server stopped.")

    def stats(self):
        """Get server statistics."""
        avg_time = (self.total_inference_time / self.request_count
                    if self.request_count > 0 else 0)
        return {
            'request_count': self.request_count,
            'total_inference_time': self.total_inference_time,
            'avg_inference_time': avg_time,
            'avg_hz': 1.0 / avg_time if avg_time > 0 else 0
        }


def main():
    parser = argparse.ArgumentParser(
        description='VLA inference server for JetBot',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5555,
        help='ZMQ port to listen on'
    )
    parser.add_argument(
        '--model',
        default='openvla/openvla-7b',
        help='HuggingFace model ID or path to fine-tuned model'
    )
    parser.add_argument(
        '--fine-tuned',
        action='store_true',
        help='Use fine-tuned model with 2-DoF output'
    )
    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Device to run inference on (auto-detects by default)'
    )
    parser.add_argument(
        '--4bit',
        dest='load_in_4bit',
        action='store_true',
        help='Use 4-bit quantization to reduce memory (CUDA only, ~4GB)'
    )
    parser.add_argument(
        '--8bit',
        dest='load_in_8bit',
        action='store_true',
        help='Use 8-bit quantization to reduce memory (CUDA only, ~8GB)'
    )
    args = parser.parse_args()

    server = VLAServer(
        port=args.port,
        model_id=args.model,
        fine_tuned=args.fine_tuned,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit
    )
    server.run()


if __name__ == '__main__':
    main()

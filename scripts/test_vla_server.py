#!/usr/bin/env python3
"""
Test script to verify the VLA server is working correctly.

Usage:
    1. Start the server in one terminal:
       python3 -m server.vla_server.server --port 5555

    2. Run this test in another terminal:
       python3 scripts/test_vla_server.py
"""

import zmq
import time
import numpy as np
import cv2
import sys


def create_test_image():
    """Create a simple test image (224x224)."""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    # Add some features
    img[:100, :] = (200, 200, 200)  # Sky/wall
    img[100:, :] = (80, 60, 40)     # Floor
    cv2.rectangle(img, (80, 80), (140, 180), (50, 50, 50), -1)  # Obstacle
    return img


def test_server(host='localhost', port=5555, timeout=120):
    """Send a test request to the VLA server."""

    print(f"\n{'='*50}")
    print("VLA Server Test")
    print(f"{'='*50}\n")

    # Create ZMQ socket
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, timeout * 1000)  # timeout in ms
    socket.setsockopt(zmq.SNDTIMEO, 5000)

    print(f"Timeout: {timeout}s (first inference may be slow on Apple Silicon)\n")

    print(f"Connecting to server at {host}:{port}...")
    socket.connect(f"tcp://{host}:{port}")

    # Create test image
    img = create_test_image()
    _, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # Test instructions
    test_cases = [
        "go forward",
        "turn left",
        "turn right",
        "avoid the obstacle",
        "stop"
    ]

    print("\nSending test requests...\n")

    for instruction in test_cases:
        try:
            # Send request
            start_time = time.time()
            socket.send_multipart([
                jpeg.tobytes(),
                instruction.encode('utf-8')
            ])

            # Receive response
            response = socket.recv_multipart()
            latency = (time.time() - start_time) * 1000

            if response[0] == b'ERROR':
                print(f"  ❌ '{instruction}': ERROR - {response[1].decode()}")
            else:
                left = float(response[0].decode())
                right = float(response[1].decode())
                print(f"  ✓ '{instruction}'")
                print(f"    → L: {left:+.3f}, R: {right:+.3f}  ({latency:.0f}ms)")

        except zmq.error.Again:
            print(f"  ❌ '{instruction}': TIMEOUT (server not responding)")
            break
        except Exception as e:
            print(f"  ❌ '{instruction}': {e}")
            break

    socket.close()
    context.term()

    print(f"\n{'='*50}")
    print("Test complete!")
    print(f"{'='*50}\n")


def check_server_status(host='localhost', port=5555):
    """Quick check if server is reachable."""
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 2000)
    socket.setsockopt(zmq.LINGER, 0)

    try:
        socket.connect(f"tcp://{host}:{port}")
        # Send minimal request
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode('.jpg', img)
        socket.send_multipart([jpeg.tobytes(), b"test"])
        socket.recv_multipart()
        return True
    except:
        return False
    finally:
        socket.close()
        context.term()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--timeout', type=int, default=120, help='Timeout in seconds (default: 120 for first run)')
    parser.add_argument('--check', action='store_true', help='Quick status check only')
    args = parser.parse_args()

    if args.check:
        if check_server_status(args.host, args.port):
            print("✓ Server is running")
            sys.exit(0)
        else:
            print("✗ Server not responding")
            sys.exit(1)
    else:
        test_server(args.host, args.port, args.timeout)
#!/usr/bin/env python3
"""
Simple data collection script for testing without a JetBot.

This creates sample training data using your laptop's webcam.
Use keyboard to simulate robot controls:
  W/S: Forward/Backward
  A/D: Turn Left/Right
  SPACE: Save sample
  Q: Quit

Usage:
    python scripts/collect_sample_data.py
"""

import os
import json
import time
import cv2
import numpy as np
from uuid import uuid1

DATASET_DIR = 'dataset_vla'
IMAGE_SIZE = 224

# Simulated motor state
left_speed = 0.0
right_speed = 0.0

def main():
    global left_speed, right_speed

    # Create dataset directory
    os.makedirs(DATASET_DIR, exist_ok=True)

    # Get current instruction
    print("\n=== VLA Data Collection (Laptop Mode) ===\n")
    instruction = input("Enter instruction (e.g., 'go forward'): ").strip()
    if not instruction:
        instruction = "navigate forward avoiding obstacles"

    print(f"\nInstruction: {instruction}")
    print("\nControls:")
    print("  W/S: Forward/Backward")
    print("  A/D: Turn Left/Right")
    print("  SPACE: Save sample")
    print("  R: Change instruction")
    print("  Q: Quit\n")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        print("Creating synthetic test data instead...")
        create_synthetic_data(instruction)
        return

    sample_count = len([f for f in os.listdir(DATASET_DIR) if f.endswith('.json')])
    print(f"Existing samples: {sample_count}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to model input size
        frame_resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))

        # Display frame with info overlay
        display = frame.copy()
        cv2.putText(display, f"L: {left_speed:.2f}  R: {right_speed:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Samples: {sample_count}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Instruction: {instruction[:40]}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(display, "SPACE=save, Q=quit",
                    (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Data Collection', display)

        key = cv2.waitKey(30) & 0xFF

        # Motor control simulation
        if key == ord('w'):
            left_speed = min(1.0, left_speed + 0.1)
            right_speed = min(1.0, right_speed + 0.1)
        elif key == ord('s'):
            left_speed = max(-1.0, left_speed - 0.1)
            right_speed = max(-1.0, right_speed - 0.1)
        elif key == ord('a'):
            left_speed = max(-1.0, left_speed - 0.1)
            right_speed = min(1.0, right_speed + 0.1)
        elif key == ord('d'):
            left_speed = min(1.0, left_speed + 0.1)
            right_speed = max(-1.0, right_speed - 0.1)
        elif key == ord(' '):
            # Save sample
            sample_id = str(uuid1())

            # Save image
            image_path = os.path.join(DATASET_DIR, f'{sample_id}.jpg')
            cv2.imwrite(image_path, frame_resized)

            # Save metadata
            meta = {
                'instruction': instruction,
                'action': {
                    'left_speed': float(left_speed),
                    'right_speed': float(right_speed)
                },
                'timestamp': time.time()
            }
            meta_path = os.path.join(DATASET_DIR, f'{sample_id}.json')
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            sample_count += 1
            print(f"Saved sample {sample_count}: L={left_speed:.2f}, R={right_speed:.2f}")

        elif key == ord('r'):
            # Change instruction
            cv2.destroyAllWindows()
            instruction = input("\nEnter new instruction: ").strip()
            if not instruction:
                instruction = "navigate forward avoiding obstacles"
            print(f"New instruction: {instruction}\n")

        elif key == ord('q'):
            break

        # Decay motor speeds toward zero
        left_speed *= 0.95
        right_speed *= 0.95

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! Collected {sample_count} samples in {DATASET_DIR}/")


def create_synthetic_data(instruction, num_samples=10):
    """Create synthetic test data without a camera."""
    print(f"Creating {num_samples} synthetic samples...")

    for i in range(num_samples):
        sample_id = str(uuid1())

        # Create a synthetic image (colored rectangle)
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        color = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
        cv2.rectangle(img, (20, 20), (200, 200), color, -1)
        cv2.putText(img, f"Sample {i+1}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Random motor values
        left_speed = np.random.uniform(-0.5, 0.5)
        right_speed = np.random.uniform(-0.5, 0.5)

        # Save
        image_path = os.path.join(DATASET_DIR, f'{sample_id}.jpg')
        cv2.imwrite(image_path, img)

        meta = {
            'instruction': instruction,
            'action': {
                'left_speed': float(left_speed),
                'right_speed': float(right_speed)
            },
            'timestamp': time.time()
        }
        meta_path = os.path.join(DATASET_DIR, f'{sample_id}.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"  Created sample {i+1}/{num_samples}")

    print(f"\nDone! Created {num_samples} synthetic samples in {DATASET_DIR}/")


if __name__ == '__main__':
    main()

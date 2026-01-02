#!/usr/bin/env python3
"""
Download and convert existing robot datasets for JetBot VLA training.

Supported datasets:
1. BridgeData V2 - Real robot navigation/manipulation
2. Open X-Embodiment subsets
3. Custom video annotation

Usage:
    python download_dataset.py --dataset bridge --output ./data/bridge_jetbot
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
from uuid import uuid1


def download_bridge_dataset(output_dir, max_samples=1000):
    """
    Download and convert BridgeData V2 to JetBot format.

    BridgeData contains robot manipulation data that can be adapted
    for navigation by using x,y movements as differential drive.
    """
    try:
        import tensorflow_datasets as tfds
    except ImportError:
        print("Installing tensorflow_datasets...")
        os.system("pip install tensorflow_datasets")
        import tensorflow_datasets as tfds

    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading BridgeData V2 (max {max_samples} samples)...")

    # Load dataset
    dataset = tfds.load(
        'bridge_dataset',
        split=f'train[:{max_samples}]',
        shuffle_files=True
    )

    sample_count = 0

    for episode in dataset:
        steps = episode['steps']

        for step in steps:
            # Extract image
            image = step['observation']['image'].numpy()

            # Extract action (7-DoF: x, y, z, roll, pitch, yaw, gripper)
            action = step['action'].numpy()

            # Convert to differential drive
            # Use x (forward) and y (lateral) for JetBot
            forward = float(action[0])  # x movement
            lateral = float(action[1])  # y movement (turning)

            # Map to differential drive
            left_speed = np.clip(forward + lateral, -1.0, 1.0)
            right_speed = np.clip(forward - lateral, -1.0, 1.0)

            # Get language instruction if available
            instruction = step.get('language_instruction', b'navigate forward')
            if isinstance(instruction, bytes):
                instruction = instruction.decode('utf-8')
            if not instruction:
                instruction = 'navigate forward'

            # Save sample
            sample_id = str(uuid1())

            # Save image (resize to 224x224)
            img = Image.fromarray(image)
            img = img.resize((224, 224), Image.LANCZOS)
            img.save(os.path.join(output_dir, f'{sample_id}.jpg'))

            # Save metadata
            meta = {
                'instruction': instruction,
                'action': {
                    'left_speed': float(left_speed),
                    'right_speed': float(right_speed)
                },
                'source': 'bridge_dataset'
            }
            with open(os.path.join(output_dir, f'{sample_id}.json'), 'w') as f:
                json.dump(meta, f, indent=2)

            sample_count += 1

            if sample_count >= max_samples:
                break

        if sample_count >= max_samples:
            break

        if sample_count % 100 == 0:
            print(f"Processed {sample_count} samples...")

    print(f"Done! Saved {sample_count} samples to {output_dir}")
    return sample_count


def create_navigation_dataset_from_videos(video_dir, output_dir, fps=2):
    """
    Create dataset from navigation videos.

    Expects videos in video_dir with corresponding .txt files containing
    instructions. You'll need to manually label actions.
    """
    import cv2

    os.makedirs(output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

    print(f"Found {len(video_files)} videos")
    print("\nFor each video, you'll be asked to label the motor actions.")
    print("Use W/A/S/D to set direction, SPACE to save frame, Q to skip to next video\n")

    sample_count = 0

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)

        # Check for instruction file
        txt_file = video_path.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(txt_file):
            with open(txt_file, 'r') as f:
                instruction = f.read().strip()
        else:
            instruction = input(f"Enter instruction for {video_file}: ").strip()
            if not instruction:
                instruction = "navigate forward"

        print(f"\nProcessing: {video_file}")
        print(f"Instruction: {instruction}")

        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps)

        frame_num = 0
        left_speed = 0.0
        right_speed = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            if frame_num % frame_interval != 0:
                continue

            # Display frame
            display = frame.copy()
            cv2.putText(display, f"L: {left_speed:.2f}  R: {right_speed:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "WASD=control, SPACE=save, Q=next",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.imshow('Label Video', display)

            key = cv2.waitKey(0) & 0xFF

            if key == ord('w'):
                left_speed = min(1.0, left_speed + 0.2)
                right_speed = min(1.0, right_speed + 0.2)
            elif key == ord('s'):
                left_speed = max(-1.0, left_speed - 0.2)
                right_speed = max(-1.0, right_speed - 0.2)
            elif key == ord('a'):
                left_speed = max(-1.0, left_speed - 0.2)
                right_speed = min(1.0, right_speed + 0.2)
            elif key == ord('d'):
                left_speed = min(1.0, left_speed + 0.2)
                right_speed = max(-1.0, right_speed - 0.2)
            elif key == ord(' '):
                # Save frame
                sample_id = str(uuid1())

                frame_resized = cv2.resize(frame, (224, 224))
                cv2.imwrite(os.path.join(output_dir, f'{sample_id}.jpg'), frame_resized)

                meta = {
                    'instruction': instruction,
                    'action': {
                        'left_speed': float(left_speed),
                        'right_speed': float(right_speed)
                    },
                    'source': video_file
                }
                with open(os.path.join(output_dir, f'{sample_id}.json'), 'w') as f:
                    json.dump(meta, f, indent=2)

                sample_count += 1
                print(f"Saved sample {sample_count}")

            elif key == ord('q'):
                break

            # Decay
            left_speed *= 0.9
            right_speed *= 0.9

        cap.release()

    cv2.destroyAllWindows()
    print(f"\nDone! Saved {sample_count} samples to {output_dir}")
    return sample_count


def generate_synthetic_navigation_data(output_dir, num_samples=500):
    """
    Generate synthetic navigation training data.

    Creates simple geometric scenes with corresponding navigation actions.
    Useful for initial testing before real data collection.
    """
    import cv2

    os.makedirs(output_dir, exist_ok=True)

    instructions_and_actions = [
        ("go forward", (0.4, 0.4)),
        ("move ahead", (0.4, 0.4)),
        ("navigate forward avoiding obstacles", (0.35, 0.35)),
        ("turn left", (-0.2, 0.4)),
        ("go left", (-0.1, 0.3)),
        ("rotate left", (-0.3, 0.3)),
        ("turn right", (0.4, -0.2)),
        ("go right", (0.3, -0.1)),
        ("rotate right", (0.3, -0.3)),
        ("stop", (0.0, 0.0)),
        ("wait", (0.0, 0.0)),
        ("go backwards", (-0.3, -0.3)),
        ("reverse", (-0.3, -0.3)),
    ]

    print(f"Generating {num_samples} synthetic samples...")

    for i in range(num_samples):
        # Pick random instruction
        instruction, (left_base, right_base) = instructions_and_actions[i % len(instructions_and_actions)]

        # Add some noise to actions
        left_speed = np.clip(left_base + np.random.normal(0, 0.05), -1, 1)
        right_speed = np.clip(right_base + np.random.normal(0, 0.05), -1, 1)

        # Create synthetic image
        img = np.zeros((224, 224, 3), dtype=np.uint8)

        # Random background color (floor)
        floor_color = (
            np.random.randint(60, 120),
            np.random.randint(40, 80),
            np.random.randint(30, 60)
        )
        img[:] = floor_color

        # Add horizon line
        horizon = np.random.randint(80, 140)
        wall_color = (
            np.random.randint(150, 220),
            np.random.randint(150, 220),
            np.random.randint(150, 220)
        )
        img[:horizon, :] = wall_color

        # Add random obstacles
        num_obstacles = np.random.randint(0, 4)
        for _ in range(num_obstacles):
            ox = np.random.randint(20, 200)
            oy = np.random.randint(horizon, 200)
            ow = np.random.randint(20, 60)
            oh = np.random.randint(20, 80)
            obs_color = (
                np.random.randint(0, 100),
                np.random.randint(0, 100),
                np.random.randint(0, 100)
            )
            cv2.rectangle(img, (ox, oy), (ox + ow, oy + oh), obs_color, -1)

        # Maybe add a target object for "go to" instructions
        if "go to" in instruction or "approach" in instruction:
            tx, ty = np.random.randint(60, 160), np.random.randint(horizon + 20, 180)
            cv2.circle(img, (tx, ty), 15, (0, 0, 255), -1)  # Red ball

        # Save
        sample_id = str(uuid1())
        cv2.imwrite(os.path.join(output_dir, f'{sample_id}.jpg'), img)

        meta = {
            'instruction': instruction,
            'action': {
                'left_speed': float(left_speed),
                'right_speed': float(right_speed)
            },
            'source': 'synthetic'
        }
        with open(os.path.join(output_dir, f'{sample_id}.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_samples} samples")

    print(f"Done! Saved {num_samples} synthetic samples to {output_dir}")
    return num_samples


def main():
    parser = argparse.ArgumentParser(
        description='Download/create datasets for JetBot VLA training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dataset',
        choices=['bridge', 'video', 'synthetic'],
        default='synthetic',
        help='Dataset source'
    )
    parser.add_argument(
        '--output',
        default='./dataset_vla',
        help='Output directory'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=500,
        help='Number of samples to generate/download'
    )
    parser.add_argument(
        '--video-dir',
        default='./videos',
        help='Directory containing videos (for video mode)'
    )
    args = parser.parse_args()

    if args.dataset == 'bridge':
        download_bridge_dataset(args.output, args.samples)
    elif args.dataset == 'video':
        create_navigation_dataset_from_videos(args.video_dir, args.output)
    elif args.dataset == 'synthetic':
        generate_synthetic_navigation_data(args.output, args.samples)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Download robot navigation datasets from HuggingFace.

This script downloads datasets that are compatible with JetBot VLA training.
"""

import os
import json
import numpy as np
from uuid import uuid1
from PIL import Image


def download_pusht_dataset(output_dir, max_samples=1000):
    """
    Download PushT dataset - a simple 2D navigation/pushing task.
    Good for testing differential drive control.
    """
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)

    print("Downloading PushT dataset from HuggingFace...")

    # Load dataset
    dataset = load_dataset("lerobot/pusht", split="train")

    print(f"Dataset size: {len(dataset)} samples")

    sample_count = 0
    instructions = [
        "push the T to the target",
        "navigate to push the object",
        "move toward the T shape",
        "go to the target location",
    ]

    for i, sample in enumerate(dataset):
        if sample_count >= max_samples:
            break

        # Get image
        image = sample.get('observation.image') or sample.get('image')
        if image is None:
            continue

        # Get action (typically [x, y] or similar)
        action = sample.get('action')
        if action is None:
            continue

        action = np.array(action)

        # Convert 2D action to differential drive
        if len(action) >= 2:
            forward = float(action[0])  # x movement
            lateral = float(action[1])  # y movement

            # Normalize to [-1, 1] range
            forward = np.clip(forward / max(abs(forward), 1e-6) * 0.5 if abs(forward) > 0.1 else forward, -1, 1)
            lateral = np.clip(lateral / max(abs(lateral), 1e-6) * 0.3 if abs(lateral) > 0.1 else lateral, -1, 1)

            left_speed = float(np.clip(forward + lateral, -1, 1))
            right_speed = float(np.clip(forward - lateral, -1, 1))
        else:
            continue

        # Convert image
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image.astype(np.uint8))
        else:
            img = image

        img = img.convert('RGB').resize((224, 224), Image.LANCZOS)

        # Save
        sample_id = str(uuid1())
        img.save(os.path.join(output_dir, f'{sample_id}.jpg'))

        meta = {
            'instruction': instructions[sample_count % len(instructions)],
            'action': {
                'left_speed': left_speed,
                'right_speed': right_speed
            },
            'source': 'pusht'
        }
        with open(os.path.join(output_dir, f'{sample_id}.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        sample_count += 1

        if sample_count % 100 == 0:
            print(f"Processed {sample_count}/{max_samples} samples...")

    print(f"Done! Saved {sample_count} samples to {output_dir}")
    return sample_count


def download_xarm_dataset(output_dir, max_samples=1000):
    """
    Download xArm dataset - robot manipulation with images.
    """
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)

    print("Downloading xArm dataset from HuggingFace...")

    try:
        dataset = load_dataset("lerobot/xarm_lift_medium", split="train")
    except Exception as e:
        print(f"Could not load xarm dataset: {e}")
        print("Trying aloha dataset instead...")
        return download_aloha_dataset(output_dir, max_samples)

    print(f"Dataset size: {len(dataset)} samples")

    sample_count = 0
    instructions = [
        "lift the object",
        "move the arm forward",
        "navigate to the target",
        "approach the object",
        "go to the goal position",
    ]

    for i, sample in enumerate(dataset):
        if sample_count >= max_samples:
            break

        # Try different image keys
        image = None
        for key in ['observation.image', 'image', 'observation.images.top']:
            if key in sample and sample[key] is not None:
                image = sample[key]
                break

        if image is None:
            continue

        # Get action
        action = sample.get('action')
        if action is None:
            continue

        action = np.array(action)

        # Use first two dimensions for differential drive
        if len(action) >= 2:
            left_speed = float(np.clip(action[0], -1, 1))
            right_speed = float(np.clip(action[1], -1, 1))
        else:
            continue

        # Convert image
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image.astype(np.uint8))
        else:
            img = image

        img = img.convert('RGB').resize((224, 224), Image.LANCZOS)

        # Save
        sample_id = str(uuid1())
        img.save(os.path.join(output_dir, f'{sample_id}.jpg'))

        meta = {
            'instruction': instructions[sample_count % len(instructions)],
            'action': {
                'left_speed': left_speed,
                'right_speed': right_speed
            },
            'source': 'xarm'
        }
        with open(os.path.join(output_dir, f'{sample_id}.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        sample_count += 1

        if sample_count % 100 == 0:
            print(f"Processed {sample_count}/{max_samples} samples...")

    print(f"Done! Saved {sample_count} samples to {output_dir}")
    return sample_count


def download_aloha_dataset(output_dir, max_samples=1000):
    """
    Download ALOHA dataset - bimanual manipulation with camera views.
    """
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)

    print("Downloading ALOHA dataset from HuggingFace...")

    try:
        dataset = load_dataset("lerobot/aloha_sim_transfer_cube_human", split="train")
    except Exception as e:
        print(f"Could not load aloha dataset: {e}")
        return 0

    print(f"Dataset size: {len(dataset)} samples")

    sample_count = 0
    instructions = [
        "transfer the cube",
        "move to the target",
        "navigate forward",
        "approach the object",
        "go to the goal",
    ]

    for i, sample in enumerate(dataset):
        if sample_count >= max_samples:
            break

        # Try different image keys
        image = None
        for key in ['observation.images.top', 'observation.image', 'image']:
            if key in sample and sample[key] is not None:
                image = sample[key]
                break

        if image is None:
            continue

        # Get action
        action = sample.get('action')
        if action is None:
            continue

        action = np.array(action)

        # Use first two dimensions
        if len(action) >= 2:
            left_speed = float(np.clip(action[0] * 2, -1, 1))
            right_speed = float(np.clip(action[1] * 2, -1, 1))
        else:
            continue

        # Convert image
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image.astype(np.uint8))
        else:
            img = image

        img = img.convert('RGB').resize((224, 224), Image.LANCZOS)

        # Save
        sample_id = str(uuid1())
        img.save(os.path.join(output_dir, f'{sample_id}.jpg'))

        meta = {
            'instruction': instructions[sample_count % len(instructions)],
            'action': {
                'left_speed': left_speed,
                'right_speed': right_speed
            },
            'source': 'aloha'
        }
        with open(os.path.join(output_dir, f'{sample_id}.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        sample_count += 1

        if sample_count % 100 == 0:
            print(f"Processed {sample_count}/{max_samples} samples...")

    print(f"Done! Saved {sample_count} samples to {output_dir}")
    return sample_count


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['pusht', 'xarm', 'aloha', 'all'], default='pusht')
    parser.add_argument('--output', default='./dataset_vla_hf')
    parser.add_argument('--samples', type=int, default=500)
    args = parser.parse_args()

    if args.dataset == 'pusht':
        download_pusht_dataset(args.output, args.samples)
    elif args.dataset == 'xarm':
        download_xarm_dataset(args.output, args.samples)
    elif args.dataset == 'aloha':
        download_aloha_dataset(args.output, args.samples)
    elif args.dataset == 'all':
        total = 0
        total += download_pusht_dataset(args.output, args.samples // 3)
        total += download_xarm_dataset(args.output, args.samples // 3)
        total += download_aloha_dataset(args.output, args.samples // 3)
        print(f"\nTotal samples: {total}")
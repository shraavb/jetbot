import os
import json
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, List, Optional


class JetBotVLADataset(Dataset):
    """
    Dataset for JetBot VLA fine-tuning.

    Loads (image, instruction, action) tuples collected via teleoperation.

    Expected data format:
        data_dir/
            sample1.jpg
            sample1.json  # {"instruction": str, "action": {"left_speed": float, "right_speed": float}}
            sample2.jpg
            sample2.json
            ...

    Example usage:
        dataset = JetBotVLADataset('data/jetbot_train')
        sample = dataset[0]
        # sample = {'image': PIL.Image, 'instruction': str, 'action': np.array([left, right])}
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 224,
        transform=None
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing image and JSON files
            image_size: Target image size (images will be resized)
            transform: Optional transform to apply to images
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transform
        self.samples = []

        # Find all JSON metadata files
        json_files = glob.glob(os.path.join(data_dir, '*.json'))

        for json_path in json_files:
            try:
                with open(json_path, 'r') as f:
                    meta = json.load(f)

                # Get corresponding image path
                base_name = os.path.splitext(os.path.basename(json_path))[0]
                image_path = os.path.join(data_dir, f"{base_name}.jpg")

                if not os.path.exists(image_path):
                    # Try PNG
                    image_path = os.path.join(data_dir, f"{base_name}.png")

                if not os.path.exists(image_path):
                    print(f"Warning: Image not found for {json_path}")
                    continue

                # Extract action
                action = meta.get('action', {})
                left_speed = action.get('left_speed', 0.0)
                right_speed = action.get('right_speed', 0.0)

                self.samples.append({
                    'image_path': image_path,
                    'instruction': meta.get('instruction', 'navigate forward'),
                    'action': np.array([left_speed, right_speed], dtype=np.float32)
                })

            except Exception as e:
                print(f"Error loading {json_path}: {e}")
                continue

        print(f"Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Load and resize image
        image = Image.open(sample['image_path']).convert('RGB')
        if image.size != (self.image_size, self.image_size):
            image = image.resize(
                (self.image_size, self.image_size),
                Image.LANCZOS
            )

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'instruction': sample['instruction'],
            'action': sample['action']
        }

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if len(self.samples) == 0:
            return {}

        actions = np.array([s['action'] for s in self.samples])
        instructions = [s['instruction'] for s in self.samples]

        return {
            'num_samples': len(self.samples),
            'action_mean': actions.mean(axis=0).tolist(),
            'action_std': actions.std(axis=0).tolist(),
            'action_min': actions.min(axis=0).tolist(),
            'action_max': actions.max(axis=0).tolist(),
            'unique_instructions': len(set(instructions)),
            'instruction_examples': list(set(instructions))[:10]
        }


def create_train_val_split(
    data_dir: str,
    val_ratio: float = 0.1,
    seed: int = 42
) -> tuple:
    """
    Create train/validation split from a dataset directory.

    Args:
        data_dir: Directory containing the full dataset
        val_ratio: Fraction of data to use for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    np.random.seed(seed)

    # Load full dataset
    full_dataset = JetBotVLADataset(data_dir)

    # Create indices
    n_samples = len(full_dataset)
    indices = np.random.permutation(n_samples)

    n_val = int(n_samples * val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    # Create split datasets
    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples = [full_dataset.samples[i] for i in val_indices]

    train_dataset = JetBotVLADataset.__new__(JetBotVLADataset)
    train_dataset.samples = train_samples
    train_dataset.image_size = full_dataset.image_size
    train_dataset.transform = full_dataset.transform
    train_dataset.data_dir = data_dir

    val_dataset = JetBotVLADataset.__new__(JetBotVLADataset)
    val_dataset.samples = val_samples
    val_dataset.image_size = full_dataset.image_size
    val_dataset.transform = full_dataset.transform
    val_dataset.data_dir = data_dir

    print(f"Split: {len(train_samples)} train, {len(val_samples)} validation")

    return train_dataset, val_dataset

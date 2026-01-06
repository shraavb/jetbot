"""
Training Curves Visualization

Generate plots from training logs for SmolVLA experiments.

Usage:
    python experiments/visualize_training.py --log experiments/logs/training_log.json
    python experiments/visualize_training.py --log models/smolvla_jetbot/training_log.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plots will be saved as text summaries.")


def load_training_log(log_path: Path) -> Dict:
    """Load training log from JSON file."""
    with open(log_path) as f:
        return json.load(f)


def plot_loss_curves(log: Dict, output_path: Path):
    """Plot training and validation loss curves."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plot generation")
        return

    epochs = log.get('epochs', [])
    if not epochs:
        print("No epoch data found in log")
        return

    epoch_nums = [e['epoch'] for e in epochs]
    train_losses = [e['train_loss'] for e in epochs]
    val_losses = [e.get('val_loss') for e in epochs if e.get('val_loss') is not None]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epoch_nums, train_losses, 'b-', label='Train Loss', linewidth=2)
    if val_losses and len(val_losses) == len(epoch_nums):
        ax.plot(epoch_nums, val_losses, 'r-', label='Val Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('SmolVLA Training Curves', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to {output_path}")


def plot_learning_rate(log: Dict, output_path: Path):
    """Plot learning rate schedule."""
    if not HAS_MATPLOTLIB:
        return

    steps = log.get('steps', [])
    if not steps:
        print("No step data found in log")
        return

    step_nums = [s['step'] for s in steps]
    lrs = [s['lr'] for s in steps]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(step_nums, lrs, 'g-', linewidth=1.5)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule (OneCycleLR)', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Learning rate plot saved to {output_path}")


def plot_step_loss(log: Dict, output_path: Path):
    """Plot step-level loss (smoothed)."""
    if not HAS_MATPLOTLIB:
        return

    steps = log.get('steps', [])
    if not steps:
        return

    step_nums = [s['step'] for s in steps]
    losses = [s['loss'] for s in steps]

    # Smooth with moving average
    window = min(50, len(losses) // 10) if len(losses) > 100 else 5
    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
    smoothed_steps = step_nums[window-1:]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(step_nums, losses, 'b-', alpha=0.3, linewidth=0.5, label='Raw')
    ax.plot(smoothed_steps, smoothed, 'b-', linewidth=2, label=f'Smoothed (window={window})')

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Step-Level Training Loss', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Step loss plot saved to {output_path}")


def generate_text_summary(log: Dict, output_path: Path):
    """Generate text summary when matplotlib is not available."""
    lines = []
    lines.append("=" * 60)
    lines.append("SMOLVLA TRAINING SUMMARY")
    lines.append("=" * 60)

    config = log.get('config', {})
    lines.append(f"\nConfiguration:")
    lines.append(f"  Model: {config.get('model_id', 'N/A')}")
    lines.append(f"  Epochs: {config.get('num_epochs', 'N/A')}")
    lines.append(f"  Batch Size: {config.get('batch_size', 'N/A')}")
    lines.append(f"  Learning Rate: {config.get('learning_rate', 'N/A')}")
    lines.append(f"  Train Samples: {config.get('train_samples', 'N/A')}")
    lines.append(f"  Val Samples: {config.get('val_samples', 'N/A')}")

    lines.append(f"\nTraining Time:")
    lines.append(f"  Start: {log.get('start_time', 'N/A')}")
    lines.append(f"  End: {log.get('end_time', 'N/A')}")

    epochs = log.get('epochs', [])
    if epochs:
        lines.append(f"\nLoss by Epoch:")
        lines.append(f"  {'Epoch':>6} {'Train':>10} {'Val':>10}")
        lines.append(f"  {'-'*6} {'-'*10} {'-'*10}")
        for e in epochs:
            val = f"{e['val_loss']:.4f}" if e.get('val_loss') else "N/A"
            lines.append(f"  {e['epoch']:>6} {e['train_loss']:>10.4f} {val:>10}")

    final = log.get('final_metrics', {})
    if final:
        lines.append(f"\nFinal Metrics:")
        lines.append(f"  Best Val Loss: {final.get('best_val_loss', 'N/A')}")
        lines.append(f"  Final Train Loss: {final.get('final_train_loss', 'N/A')}")
        lines.append(f"  Total Steps: {final.get('total_steps', 'N/A')}")

    summary = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(summary)
    print(f"Text summary saved to {output_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Visualize SmolVLA training curves',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--log',
        type=Path,
        required=True,
        help='Path to training log JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('experiments/visualizations'),
        help='Output directory for plots'
    )
    args = parser.parse_args()

    if not args.log.exists():
        print(f"Log file not found: {args.log}")
        return

    log = load_training_log(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    stem = args.log.stem
    plot_loss_curves(log, args.output_dir / f"{stem}_loss_curves.png")
    plot_learning_rate(log, args.output_dir / f"{stem}_lr_schedule.png")
    plot_step_loss(log, args.output_dir / f"{stem}_step_loss.png")
    generate_text_summary(log, args.output_dir / f"{stem}_summary.txt")

    print(f"\nVisualizations saved to {args.output_dir}")


if __name__ == '__main__':
    main()

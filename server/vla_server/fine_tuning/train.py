import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict
import yaml
import argparse
from tqdm import tqdm

from .dataset import JetBotVLADataset, create_train_val_split


def create_jetbot_openvla(
    base_model_id: str = "openvla/openvla-7b",
    action_dim: int = 2,
    freeze_backbone: bool = True
):
    """
    Create OpenVLA model with modified action head for JetBot.

    Replaces the 7-DoF action output with 2-DoF (left_speed, right_speed).

    Args:
        base_model_id: HuggingFace model ID
        action_dim: Output action dimension (2 for differential drive)
        freeze_backbone: Whether to freeze the vision-language backbone

    Returns:
        Modified model ready for fine-tuning
    """
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print(f"Loading base model: {base_model_id}")
    processor = AutoProcessor.from_pretrained(
        base_model_id,
        trust_remote_code=True
    )
    model = AutoModelForVision2Seq.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # Get hidden size from model config
    hidden_size = model.config.hidden_size

    # Replace action head with JetBot-specific head
    # Note: The actual attribute name depends on OpenVLA's architecture
    # This may need adjustment based on the model's implementation
    model.action_head = nn.Sequential(
        nn.Linear(hidden_size, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, action_dim),
        nn.Tanh()  # Output in [-1, 1] for motor speeds
    )

    # Initialize new layers
    for module in model.action_head.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    if freeze_backbone:
        # Freeze everything except action head
        for name, param in model.named_parameters():
            if 'action_head' not in name:
                param.requires_grad = False
        print("Backbone frozen, only training action head")
    else:
        print("Training full model")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100*trainable_params/total_params:.2f}%)")

    return model, processor


def fine_tune(
    model,
    processor,
    train_dataset: JetBotVLADataset,
    val_dataset: Optional[JetBotVLADataset] = None,
    config: Optional[Dict] = None
):
    """
    Fine-tune OpenVLA for JetBot navigation.

    Args:
        model: OpenVLA model with modified action head
        processor: OpenVLA processor
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        config: Training configuration dict

    Returns:
        Fine-tuned model
    """
    # Default config
    default_config = {
        'num_epochs': 10,
        'batch_size': 4,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'warmup_steps': 100,
        'save_every_n_epochs': 2,
        'save_dir': './checkpoints',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    if config:
        default_config.update(config)
    config = default_config

    device = config['device']
    model = model.to(device)

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2
        )

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        total_steps=total_steps,
        pct_start=config['warmup_steps'] / total_steps
    )

    # Training loop
    os.makedirs(config['save_dir'], exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch in pbar:
            images = batch['image']
            instructions = batch['instruction']
            target_actions = batch['action'].to(device)

            # Process inputs
            # Note: This needs to match OpenVLA's expected input format
            inputs = processor(
                text=[f"In: What action should the robot take to {inst}?\nOut:"
                      for inst in instructions],
                images=images,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(device) if hasattr(v, 'to') else v
                      for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs)

            # Get predicted actions (depends on model architecture)
            # This may need adjustment based on OpenVLA's output format
            if hasattr(outputs, 'actions'):
                predicted_actions = outputs.actions
            else:
                predicted_actions = model.action_head(outputs.last_hidden_state[:, -1, :])

            # MSE loss for action regression
            loss = F.mse_loss(predicted_actions, target_actions)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")

        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image']
                    instructions = batch['instruction']
                    target_actions = batch['action'].to(device)

                    inputs = processor(
                        text=[f"In: What action should the robot take to {inst}?\nOut:"
                              for inst in instructions],
                        images=images,
                        return_tensors="pt",
                        padding=True
                    )
                    inputs = {k: v.to(device) if hasattr(v, 'to') else v
                              for k, v in inputs.items()}

                    outputs = model(**inputs)
                    if hasattr(outputs, 'actions'):
                        predicted_actions = outputs.actions
                    else:
                        predicted_actions = model.action_head(
                            outputs.last_hidden_state[:, -1, :]
                        )

                    loss = F.mse_loss(predicted_actions, target_actions)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(config['save_dir'], 'best_model')
                model.save_pretrained(save_path)
                processor.save_pretrained(save_path)
                print(f"Saved best model to {save_path}")

        # Periodic checkpoint
        if (epoch + 1) % config['save_every_n_epochs'] == 0:
            save_path = os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}')
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"Saved checkpoint to {save_path}")

    # Save final model
    final_path = os.path.join(config['save_dir'], 'final_model')
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"Saved final model to {final_path}")

    return model


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune OpenVLA for JetBot navigation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--data-dir',
        required=True,
        help='Directory containing training data'
    )
    parser.add_argument(
        '--config',
        default=None,
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--base-model',
        default='openvla/openvla-7b',
        help='Base model ID'
    )
    parser.add_argument(
        '--output-dir',
        default='./models/jetbot_openvla',
        help='Output directory for fine-tuned model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Training batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    args = parser.parse_args()

    # Load config
    config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'save_dir': args.output_dir
    }

    if args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        config.update(yaml_config.get('training', {}))

    # Create datasets
    train_dataset, val_dataset = create_train_val_split(args.data_dir)

    # Create model
    model, processor = create_jetbot_openvla(
        base_model_id=args.base_model
    )

    # Fine-tune
    fine_tune(
        model,
        processor,
        train_dataset,
        val_dataset,
        config
    )


if __name__ == '__main__':
    main()

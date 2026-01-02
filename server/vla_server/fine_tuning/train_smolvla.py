"""
SmolVLA Fine-tuning Script for JetBot

Train the SmolVLA action head on JetBot navigation data.
Optimized for resource-constrained environments like Jetson Nano.

Usage:
    python -m server.vla_server.fine_tuning.train_smolvla \
        --data-dir dataset_vla_synthetic_large \
        --output-dir models/smolvla_jetbot \
        --epochs 20
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path

from .dataset import JetBotVLADataset, create_train_val_split
from .jetbot_action_head import JetBotActionHead


def smolvla_collate_fn(batch):
    """Custom collate function for SmolVLA training."""
    images = [item['image'] for item in batch]
    instructions = [item['instruction'] for item in batch]
    actions = torch.stack([torch.from_numpy(item['action']) for item in batch])
    return {
        'image': images,
        'instruction': instructions,
        'action': actions
    }


class SmolVLATrainer:
    """
    Trainer for SmolVLA fine-tuning on JetBot data.

    This trainer only updates the action head while keeping
    the vision-language backbone frozen, making it feasible
    to train on resource-constrained hardware.
    """

    def __init__(
        self,
        model_id: str = "HuggingFaceM4/SmolVLM-Instruct",
        output_dir: str = "./models/smolvla_jetbot",
        device: str = "auto",
        use_fp16: bool = True,
        gradient_checkpointing: bool = True
    ):
        """
        Initialize trainer.

        Args:
            model_id: Base model HuggingFace ID
            output_dir: Directory to save fine-tuned model
            device: Computation device
            use_fp16: Use mixed precision training
            gradient_checkpointing: Use gradient checkpointing
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.use_fp16 = use_fp16 and self.device == "cuda"
        self.gradient_checkpointing = gradient_checkpointing

        # Will be initialized in setup()
        self.model = None
        self.processor = None
        self.action_head = None
        self.scaler = None

    def setup(self):
        """Load model and initialize training components."""
        from transformers import AutoProcessor, AutoModelForVision2Seq

        print(f"Loading base model: {self.model_id}", flush=True)

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        # Load model
        dtype = torch.float16 if self.use_fp16 else torch.float32
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        self.model = self.model.to(self.device)
        self.model.eval()  # Freeze backbone

        # Enable gradient checkpointing
        if self.gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        # Freeze all backbone parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Get hidden size
        if hasattr(self.model.config, 'hidden_size'):
            hidden_size = self.model.config.hidden_size
        elif hasattr(self.model.config, 'text_config'):
            hidden_size = self.model.config.text_config.hidden_size
        else:
            hidden_size = 768

        # Create action head
        self.action_head = JetBotActionHead(
            hidden_size=hidden_size,
            intermediate_size=128,
            dropout=0.1
        ).to(self.device)

        # Mixed precision scaler
        if self.use_fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        # Log info
        trainable = sum(p.numel() for p in self.action_head.parameters())
        total = sum(p.numel() for p in self.model.parameters()) + trainable
        print(f"Trainable parameters: {trainable:,} / {total:,} "
              f"({100*trainable/total:.4f}%)", flush=True)

    def get_hidden_states(self, images, instructions):
        """
        Extract hidden states from the model.

        Args:
            images: List of PIL images
            instructions: List of instruction strings

        Returns:
            Hidden states tensor
        """
        # Format prompts
        prompts = [f"What action should the robot take to {inst}?"
                   for inst in instructions]

        # Process inputs
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v
                  for k, v in inputs.items()}

        # Forward pass (no grad for backbone)
        with torch.no_grad():
            if self.use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        **inputs,
                        output_hidden_states=True,
                        return_dict=True
                    )
            else:
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )

        # Extract last hidden state
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            hidden = outputs.hidden_states[-1][:, -1, :]
        elif hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states:
            hidden = outputs.decoder_hidden_states[-1][:, -1, :]
        else:
            raise RuntimeError("Could not extract hidden states")

        return hidden.float()  # Ensure float32 for action head

    def train(
        self,
        train_dataset: JetBotVLADataset,
        val_dataset: Optional[JetBotVLADataset] = None,
        config: Optional[Dict] = None
    ):
        """
        Train the action head.

        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            config: Training configuration
        """
        # Default config
        default_config = {
            'num_epochs': 20,
            'batch_size': 2,
            'learning_rate': 5e-5,
            'weight_decay': 0.01,
            'warmup_ratio': 0.1,
            'save_every_n_epochs': 5,
            'log_every_n_steps': 10
        }
        if config:
            default_config.update(config)
        config = default_config

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
            collate_fn=smolvla_collate_fn
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=0,
                collate_fn=smolvla_collate_fn
            )

        # Optimizer (only for action head)
        optimizer = torch.optim.AdamW(
            self.action_head.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # Scheduler
        total_steps = len(train_loader) * config['num_epochs']
        warmup_steps = int(total_steps * config['warmup_ratio'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['learning_rate'],
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps
        )

        # Training loop
        best_val_loss = float('inf')
        global_step = 0

        for epoch in range(config['num_epochs']):
            self.action_head.train()
            epoch_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
            for batch in pbar:
                images = batch['image']
                instructions = batch['instruction']
                target_actions = batch['action'].to(self.device)

                # Get hidden states (frozen backbone)
                hidden = self.get_hidden_states(images, instructions)

                # Forward through action head
                if self.use_fp16:
                    with torch.cuda.amp.autocast():
                        predicted_actions = self.action_head(hidden)
                        loss = F.mse_loss(predicted_actions, target_actions)
                else:
                    predicted_actions = self.action_head(hidden)
                    loss = F.mse_loss(predicted_actions, target_actions)

                # Backward
                optimizer.zero_grad()
                if self.use_fp16:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.action_head.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.action_head.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                epoch_loss += loss.item()
                global_step += 1

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_train_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}", flush=True)

            # Validation
            if val_loader:
                val_loss = self.validate(val_loader)
                print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}", flush=True)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save("best")
                    print(f"Saved best model (val_loss={val_loss:.4f})", flush=True)

            # Periodic checkpoint
            if (epoch + 1) % config['save_every_n_epochs'] == 0:
                self.save(f"checkpoint_epoch_{epoch+1}")

        # Save final model
        self.save("final")
        print(f"Training complete. Model saved to {self.output_dir}", flush=True)

    def validate(self, val_loader: DataLoader) -> float:
        """Run validation and return average loss."""
        self.action_head.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image']
                instructions = batch['instruction']
                target_actions = batch['action'].to(self.device)

                hidden = self.get_hidden_states(images, instructions)
                predicted_actions = self.action_head(hidden)
                loss = F.mse_loss(predicted_actions, target_actions)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def save(self, name: str):
        """Save action head checkpoint."""
        save_path = self.output_dir / name
        save_path.mkdir(parents=True, exist_ok=True)

        # Save action head
        torch.save(
            self.action_head.state_dict(),
            save_path / "jetbot_action_head.pt"
        )

        # Save config
        config = {
            'model_id': self.model_id,
            'hidden_size': self.action_head.hidden_size,
            'action_dim': self.action_head.action_dim
        }
        with open(save_path / "config.yaml", 'w') as f:
            yaml.dump(config, f)


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune SmolVLA for JetBot navigation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--data-dir',
        required=True,
        help='Directory containing training data'
    )
    parser.add_argument(
        '--output-dir',
        default='./models/smolvla_jetbot',
        help='Output directory for fine-tuned model'
    )
    parser.add_argument(
        '--model',
        default='HuggingFaceM4/SmolVLM-Instruct',
        help='Base model HuggingFace ID'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Training batch size (keep small for Jetson)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=5e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--no-fp16',
        action='store_true',
        help='Disable mixed precision training'
    )
    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to train on'
    )
    args = parser.parse_args()

    # Create trainer
    trainer = SmolVLATrainer(
        model_id=args.model,
        output_dir=args.output_dir,
        device=args.device,
        use_fp16=not args.no_fp16
    )

    # Setup model
    trainer.setup()

    # Create datasets
    train_dataset, val_dataset = create_train_val_split(args.data_dir)

    # Training config
    config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr
    }

    # Train
    trainer.train(train_dataset, val_dataset, config)


if __name__ == '__main__':
    main()

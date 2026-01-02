"""
SmolVLA Wrapper for JetBot

Lightweight Vision-Language-Action model wrapper optimized for
Jetson Nano and resource-constrained environments.

SmolVLA is a 450M parameter model that can run on consumer hardware
while maintaining good performance on robotics tasks.

Usage:
    from server.vla_server.smolvla_wrapper import SmolVLAWrapper

    vla = SmolVLAWrapper()
    left, right = vla.predict(image, "go forward")
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union
from pathlib import Path

from .fine_tuning.jetbot_action_head import JetBotActionHead


class SmolVLAWrapper:
    """
    SmolVLA wrapper with JetBot action head adapter.

    This wrapper provides a drop-in replacement for OpenVLAWrapper
    with significantly reduced memory footprint (~1.5GB vs ~15GB).

    Attributes:
        model_id: HuggingFace model ID or local path
        device: Computation device (cuda, cpu, auto)
        fine_tuned: Whether using a fine-tuned JetBot model
    """

    # Default model configurations
    DEFAULT_MODEL_ID = "HuggingFaceM4/SmolVLM-Instruct"  # Base VLM, will add action head
    SMOLVLA_MODEL_ID = "lerobot/smolvla"  # If available

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: str = "auto",
        fine_tuned: bool = False,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        use_flash_attention: bool = False
    ):
        """
        Initialize SmolVLA wrapper.

        Args:
            model_id: HuggingFace model ID or path to fine-tuned model
            device: Device to run on ('auto', 'cuda', 'cpu')
            fine_tuned: Whether using a fine-tuned JetBot model
            load_in_4bit: Use 4-bit quantization
            load_in_8bit: Use 8-bit quantization
            use_flash_attention: Use flash attention if available
        """
        self.fine_tuned = fine_tuned
        self.use_flash_attention = use_flash_attention

        # Determine model ID
        if model_id is None:
            model_id = self.DEFAULT_MODEL_ID
        self.model_id = model_id

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Determine dtype
        if self.device == "cuda":
            self.dtype = torch.float16  # FP16 for GPU
        else:
            self.dtype = torch.float32  # FP32 for CPU

        print(f"SmolVLA: Loading model {model_id}", flush=True)
        print(f"SmolVLA: Device={self.device}, dtype={self.dtype}", flush=True)

        # Load model and processor
        self._load_model(load_in_4bit, load_in_8bit)

        # Initialize action head
        self._init_action_head()

        print("SmolVLA: Model loaded successfully", flush=True)

    def _load_model(self, load_in_4bit: bool, load_in_8bit: bool):
        """Load the SmolVLA model and processor."""
        try:
            # Try loading SmolVLA if available
            self._load_smolvla_model(load_in_4bit, load_in_8bit)
        except Exception as e:
            print(f"SmolVLA: Could not load SmolVLA model: {e}", flush=True)
            print("SmolVLA: Falling back to SmolVLM base model", flush=True)
            self._load_smolvlm_model(load_in_4bit, load_in_8bit)

    def _load_smolvla_model(self, load_in_4bit: bool, load_in_8bit: bool):
        """Load SmolVLA from lerobot if available."""
        try:
            from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig

            config = SmolVLAConfig()
            self.model = SmolVLAPolicy(config)
            self.model.to(self.device)
            self.processor = None  # SmolVLA has built-in processing
            self.model_type = "smolvla"

        except ImportError:
            raise ImportError("lerobot not installed, cannot load SmolVLA")

    def _load_smolvlm_model(self, load_in_4bit: bool, load_in_8bit: bool):
        """Load SmolVLM as base model with custom action head."""
        from transformers import AutoProcessor, AutoModelForVision2Seq

        # Quantization config
        quantization_config = None
        if load_in_4bit or load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    bnb_4bit_compute_dtype=torch.float16
                )
            except ImportError:
                print("SmolVLA: bitsandbytes not available, skipping quantization")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        # Load model
        model_kwargs = {
            "torch_dtype": self.dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = None

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            **model_kwargs
        )

        if not quantization_config:
            self.model = self.model.to(self.device)

        self.model.eval()
        self.model_type = "smolvlm"

    def _init_action_head(self):
        """Initialize JetBot action head."""
        # Get hidden size from model config
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'hidden_size'):
                hidden_size = self.model.config.hidden_size
            elif hasattr(self.model.config, 'text_config'):
                hidden_size = self.model.config.text_config.hidden_size
            else:
                hidden_size = 768  # Default for small models
        else:
            hidden_size = 768

        self.action_head = JetBotActionHead(
            hidden_size=hidden_size,
            intermediate_size=128,
            dropout=0.1,
            action_dim=2
        ).to(self.device)

        # Load fine-tuned action head if available
        if self.fine_tuned:
            self._load_action_head()

    def _load_action_head(self):
        """Load fine-tuned action head weights."""
        model_path = Path(self.model_id)
        action_head_path = model_path / "jetbot_action_head.pt"

        if action_head_path.exists():
            state_dict = torch.load(action_head_path, map_location=self.device)
            self.action_head.load_state_dict(state_dict)
            print(f"SmolVLA: Loaded action head from {action_head_path}", flush=True)
        else:
            print("SmolVLA: No fine-tuned action head found, using random init", flush=True)

    def predict(
        self,
        image: Union[Image.Image, np.ndarray],
        instruction: str
    ) -> Tuple[float, float]:
        """
        Predict motor commands from image and instruction.

        Args:
            image: Input image (PIL Image or numpy array)
            instruction: Natural language instruction

        Returns:
            Tuple of (left_speed, right_speed) in range [-1, 1]
        """
        try:
            # Convert numpy to PIL if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            # Ensure RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize to expected size
            if image.size != (224, 224):
                image = image.resize((224, 224), Image.LANCZOS)

            # Get hidden states and predict
            with torch.no_grad():
                if self.model_type == "smolvla":
                    return self._predict_smolvla(image, instruction)
                else:
                    return self._predict_smolvlm(image, instruction)

        except Exception as e:
            print(f"SmolVLA: Prediction error: {e}", flush=True)
            return (0.0, 0.0)  # Safe stop on error

    def _predict_smolvla(
        self,
        image: Image.Image,
        instruction: str
    ) -> Tuple[float, float]:
        """Predict using SmolVLA model."""
        # SmolVLA native prediction
        observation = {
            "observation.images.top": torch.from_numpy(
                np.array(image)
            ).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0,
            "observation.state": torch.zeros(1, 7).to(self.device)  # Placeholder
        }

        # Add instruction to observation
        observation["language_instruction"] = instruction

        # Get action
        action = self.model.select_action(observation)

        # Map to differential drive
        if action.shape[-1] >= 2:
            left = float(action[0, 0].cpu())
            right = float(action[0, 1].cpu())
        else:
            # Fallback
            left = right = 0.0

        return (
            np.clip(left, -1.0, 1.0),
            np.clip(right, -1.0, 1.0)
        )

    def _predict_smolvlm(
        self,
        image: Image.Image,
        instruction: str
    ) -> Tuple[float, float]:
        """Predict using SmolVLM + action head."""
        # Format prompt
        prompt = f"What action should the robot take to {instruction}?"

        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v
                  for k, v in inputs.items()}

        # Forward pass to get hidden states
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

        # Get last hidden state
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            hidden = outputs.hidden_states[-1][:, -1, :]
        elif hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states:
            hidden = outputs.decoder_hidden_states[-1][:, -1, :]
        else:
            # Fallback: use logits shape to infer hidden size
            raise RuntimeError("Could not extract hidden states from model")

        # Predict action
        left, right = self.action_head.get_actions(hidden)

        return (
            np.clip(left, -1.0, 1.0),
            np.clip(right, -1.0, 1.0)
        )

    def save_action_head(self, save_path: str):
        """Save the action head weights."""
        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.action_head.state_dict(),
            path / "jetbot_action_head.pt"
        )
        print(f"SmolVLA: Saved action head to {path}", flush=True)

    def get_memory_usage(self) -> dict:
        """Get current memory usage."""
        if self.device == "cuda":
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9
            }
        else:
            return {"device": "cpu", "note": "Memory tracking not available for CPU"}


def create_smolvla(
    model_type: str = "auto",
    device: str = "auto",
    fine_tuned: bool = False,
    **kwargs
) -> SmolVLAWrapper:
    """
    Factory function to create SmolVLA wrapper.

    Args:
        model_type: "smolvla", "smolvlm", or "auto"
        device: Computation device
        fine_tuned: Whether to load fine-tuned weights
        **kwargs: Additional arguments

    Returns:
        SmolVLAWrapper instance
    """
    if model_type == "smolvla":
        model_id = SmolVLAWrapper.SMOLVLA_MODEL_ID
    elif model_type == "smolvlm":
        model_id = SmolVLAWrapper.DEFAULT_MODEL_ID
    else:
        # Auto: try SmolVLA first, fall back to SmolVLM
        model_id = None

    return SmolVLAWrapper(
        model_id=model_id,
        device=device,
        fine_tuned=fine_tuned,
        **kwargs
    )

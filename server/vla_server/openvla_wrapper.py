import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import platform


def get_device():
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class OpenVLAWrapper:
    """
    Wrapper for OpenVLA model inference.

    Loads the OpenVLA model from HuggingFace and provides a simple interface
    for predicting motor commands from images and instructions.

    For the base (non-fine-tuned) model, maps the 7-DoF arm action output
    to 2-DoF differential drive commands using a simple heuristic.

    After fine-tuning for JetBot, the model outputs (left_speed, right_speed)
    directly.

    Supports:
    - NVIDIA GPUs (CUDA)
    - Apple Silicon (MPS)
    - CPU fallback
    - 4-bit/8-bit quantization for reduced memory

    Example usage:
        wrapper = OpenVLAWrapper(device="auto")
        left, right = wrapper.predict(pil_image, "go to the door")
        robot.set_motors(left, right)
    """

    # Available unnorm_key options from OpenVLA training datasets
    UNNORM_KEYS = [
        'bridge_orig',  # Good for navigation-like tasks
        'fractal20220817_data',  # Google RT-1 data
        'kuka',
        'bc_z',
        'jaco_play',
        'berkeley_cable_routing',
        'roboturk',
        'viola',
        'toto',
        'berkeley_autolab_ur5',
    ]

    def __init__(
        self,
        model_id: str = "openvla/openvla-7b",
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        fine_tuned: bool = False,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        unnorm_key: str = "bridge_orig"
    ):
        """
        Initialize OpenVLA model.

        Args:
            model_id: HuggingFace model ID or path to fine-tuned model
            device: Device to run inference on ('auto', 'cuda', 'mps', or 'cpu')
            torch_dtype: Torch dtype for model weights (auto-detected if None)
            fine_tuned: Whether this is a fine-tuned JetBot model (2-DoF output)
            load_in_4bit: Use 4-bit quantization (requires bitsandbytes, CUDA only)
            load_in_8bit: Use 8-bit quantization (requires bitsandbytes, CUDA only)
            unnorm_key: Dataset key for action unnormalization (default: 'bridge_orig')
        """
        # Auto-detect device
        if device == "auto":
            device = get_device()
        self.device = device
        self.fine_tuned = fine_tuned
        self.unnorm_key = unnorm_key

        # Auto-detect dtype based on device
        if torch_dtype is None:
            if device == "cuda":
                torch_dtype = torch.bfloat16
            elif device == "mps":
                torch_dtype = torch.float16  # MPS works better with float16
            else:
                torch_dtype = torch.float32

        print(f"Loading OpenVLA model: {model_id}", flush=True)
        print(f"Device: {device}, dtype: {torch_dtype}", flush=True)

        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        # Build model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "eager",  # Avoid SDPA compatibility issues
        }

        # Quantization (CUDA only)
        if load_in_4bit and device == "cuda":
            print("Loading with 4-bit quantization...")
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["device_map"] = "auto"
        elif load_in_8bit and device == "cuda":
            print("Loading with 8-bit quantization...")
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True
            )
            model_kwargs["device_map"] = "auto"
        elif device == "cuda":
            model_kwargs["torch_dtype"] = torch_dtype
            model_kwargs["device_map"] = "auto"
        elif device == "mps":
            # MPS: load to CPU first, then move
            model_kwargs["torch_dtype"] = torch_dtype
            model_kwargs["low_cpu_mem_usage"] = True
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            **model_kwargs
        )

        # Move to device if needed (MPS or explicit CUDA without device_map)
        if device == "mps":
            print("Moving model to MPS (Apple Silicon)...", flush=True)
            self.model = self.model.to(device)
        elif device == "cuda" and not hasattr(self.model, 'hf_device_map'):
            self.model = self.model.to(device)

        self.model.eval()

        # Print memory usage
        if device == "cuda":
            mem_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"GPU memory used: {mem_gb:.1f} GB")

        print("OpenVLA model loaded successfully", flush=True)

    def predict(
        self,
        image: Image.Image,
        instruction: str
    ) -> Tuple[float, float]:
        """
        Predict motor commands from image and instruction.

        Args:
            image: PIL Image (will be resized to 224x224 if needed)
            instruction: Natural language instruction

        Returns:
            Tuple of (left_speed, right_speed) in range [-1.0, 1.0]
        """
        # Format prompt as expected by OpenVLA
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"

        # Ensure image is correct size
        if image.size != (224, 224):
            image = image.resize((224, 224), Image.LANCZOS)

        # Process inputs
        inputs = self.processor(prompt, image, return_tensors="pt")

        # Move to device
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v
                  for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            if self.fine_tuned:
                # Fine-tuned model doesn't need unnorm_key
                action = self.model.predict_action(**inputs)
            else:
                # Base model needs unnorm_key to select normalization statistics
                action = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key)

        # Convert to numpy
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        # Flatten if needed
        action = np.array(action).flatten()

        if self.fine_tuned:
            # Fine-tuned model outputs (left_speed, right_speed) directly
            left_speed = float(np.clip(action[0], -1.0, 1.0))
            right_speed = float(np.clip(action[1], -1.0, 1.0))
        else:
            # Base model outputs 7-DoF: [x, y, z, roll, pitch, yaw, gripper]
            # Map to differential drive:
            # - action[0] (x-translation) -> forward/backward
            # - action[1] (y-translation) -> left/right turning
            forward = float(action[0]) if len(action) > 0 else 0.0
            turn = float(action[1]) if len(action) > 1 else 0.0

            # Convert to differential drive
            # Scale factors may need tuning based on the model's output range
            left_speed = float(np.clip(forward + turn, -1.0, 1.0))
            right_speed = float(np.clip(forward - turn, -1.0, 1.0))

        return left_speed, right_speed

    def predict_raw(
        self,
        image: Image.Image,
        instruction: str
    ) -> np.ndarray:
        """
        Get raw action output from model (for debugging/analysis).

        Args:
            image: PIL Image
            instruction: Natural language instruction

        Returns:
            Raw action array from model
        """
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"

        if image.size != (224, 224):
            image = image.resize((224, 224), Image.LANCZOS)

        inputs = self.processor(prompt, image, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            if self.fine_tuned:
                action = self.model.predict_action(**inputs)
            else:
                action = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key)

        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        return np.array(action).flatten()

"""
JetBot Action Head Module

Provides a reusable 2-DoF action head for VLA models to control
differential drive robots like JetBot.

The action head maps from the VLA model's hidden states to
(left_speed, right_speed) motor commands in the range [-1, 1].
"""

import torch
import torch.nn as nn
from typing import Optional


class JetBotActionHead(nn.Module):
    """
    Action head for JetBot differential drive control.

    Maps VLA hidden states to 2-DoF motor commands:
    - left_speed: Left wheel velocity [-1, 1]
    - right_speed: Right wheel velocity [-1, 1]

    Architecture:
        hidden_states -> Linear -> ReLU -> Dropout -> Linear -> Tanh -> [left, right]

    Example:
        >>> head = JetBotActionHead(hidden_size=768)
        >>> hidden = torch.randn(1, 768)
        >>> actions = head(hidden)  # Shape: (1, 2)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int = 128,
        dropout: float = 0.1,
        action_dim: int = 2
    ):
        """
        Initialize the action head.

        Args:
            hidden_size: Input dimension from VLA model
            intermediate_size: Size of hidden layer (default: 128)
            dropout: Dropout probability (default: 0.1)
            action_dim: Output action dimension (default: 2 for differential drive)
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.action_dim = action_dim

        self.head = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: VLA model hidden states
                Shape: (batch_size, hidden_size) or (batch_size, seq_len, hidden_size)

        Returns:
            actions: Motor commands in [-1, 1]
                Shape: (batch_size, action_dim)
        """
        # If sequence dimension present, take last token
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, -1, :]

        return self.head(hidden_states)

    def get_actions(self, hidden_states: torch.Tensor) -> tuple:
        """
        Get actions as a tuple of floats.

        Args:
            hidden_states: VLA model hidden states

        Returns:
            Tuple of (left_speed, right_speed)
        """
        with torch.no_grad():
            actions = self.forward(hidden_states)
            actions = actions.squeeze().cpu().numpy()

            if self.action_dim == 2:
                return float(actions[0]), float(actions[1])
            else:
                return tuple(float(a) for a in actions)


class JetBotActionHeadWithHistory(nn.Module):
    """
    Action head with temporal history for smoother control.

    Uses an LSTM to incorporate past hidden states for
    more temporally coherent action predictions.
    """

    def __init__(
        self,
        hidden_size: int,
        lstm_hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        action_dim: int = 2
    ):
        """
        Initialize action head with LSTM.

        Args:
            hidden_size: Input dimension from VLA model
            lstm_hidden: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            action_dim: Output action dimension
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.action_dim = action_dim

        # Project to LSTM input size
        self.input_proj = nn.Linear(hidden_size, lstm_hidden)

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Action output
        self.output = nn.Sequential(
            nn.Linear(lstm_hidden, action_dim),
            nn.Tanh()
        )

        # Hidden state
        self.h = None
        self.c = None

    def reset_history(self):
        """Reset LSTM hidden state."""
        self.h = None
        self.c = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        reset: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with temporal history.

        Args:
            hidden_states: VLA hidden states (batch, hidden_size)
            reset: Whether to reset LSTM state

        Returns:
            actions: Motor commands (batch, action_dim)
        """
        if reset:
            self.reset_history()

        # Project input
        x = self.input_proj(hidden_states)

        # Add sequence dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # LSTM forward
        if self.h is None:
            out, (self.h, self.c) = self.lstm(x)
        else:
            out, (self.h, self.c) = self.lstm(x, (self.h, self.c))

        # Detach hidden state to prevent backprop through time
        self.h = self.h.detach()
        self.c = self.c.detach()

        # Output action
        actions = self.output(out[:, -1, :])

        return actions


def create_action_head(
    hidden_size: int,
    head_type: str = "simple",
    **kwargs
) -> nn.Module:
    """
    Factory function to create action heads.

    Args:
        hidden_size: VLA model hidden size
        head_type: "simple" or "lstm"
        **kwargs: Additional arguments for the head

    Returns:
        Action head module
    """
    if head_type == "simple":
        return JetBotActionHead(hidden_size, **kwargs)
    elif head_type == "lstm":
        return JetBotActionHeadWithHistory(hidden_size, **kwargs)
    else:
        raise ValueError(f"Unknown head type: {head_type}")

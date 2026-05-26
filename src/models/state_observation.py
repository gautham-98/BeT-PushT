import torch
import torch.nn as nn


class StateObservation(nn.Module):
    """
    Encodes the 5-D state [agent_x, agent_y, block_x, block_y, block_angle]
    into a fixed-dim embedding for BeT.

    Input  : (B, T, state_dim)  — normalised to [-1, 1]
    Output : (B, T, embed_dim)
    """

    def __init__(self, state_dim: int = 5, embed_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(8, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, embed_dim),
            nn.GELU(),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.mlp(states)

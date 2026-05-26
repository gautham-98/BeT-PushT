import torch
import torch.nn as nn


class KeypointStateObservation(nn.Module):
    """
    Encodes (flattened contour keypoints, agent state) into a fixed-dim embedding.
    Replaces the image encoder when using detected keypoints as observations.

    Input:
      keypoints : (B, T, n_pts * 2)  normalised to [-1, 1]
      states    : (B, T, state_dim)  normalised by dataset stats
    Output:
      (B, T, embed_dim)
    """

    def __init__(self, n_pts: int = 16, state_dim: int = 2, embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        input_dim = n_pts * 2 + state_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
            nn.GELU(),
        )

    def forward(self, keypoints: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        x = torch.cat([keypoints, states], dim=-1)  # (B, T, n_pts*2 + state_dim)
        return self.mlp(x)

from typing_extensions import Literal
import torch.nn as nn
import torch
import torchvision.transforms as T
from src.models.resnet import resnet18


class ImageStateCrossAttention(nn.Module):
    def __init__(self, seq_len, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_embd = nn.Parameter(torch.zeros(seq_len, embed_dim))

    def forward(self, state_embeddings, image_embeddings):
        # image_embeddings: (B*T, E, H, W)
        # state_embeddings: (B*T, E)
        img_emb_reshaped = image_embeddings.flatten(2).transpose(1, 2)  # (B*T, H*W, E)
        img_emb_reshaped = img_emb_reshaped + self.pos_embd.unsqueeze(0)
        state_emb_reshaped = state_embeddings.unsqueeze(1)  # (B*T, 1, E)

        attn_output, _ = self.attention(state_emb_reshaped, img_emb_reshaped, img_emb_reshaped)
        attn_output = attn_output.squeeze(1)  # (B*T, E)

        output = self.norm(state_embeddings + self.dropout(attn_output))
        return output


def _build_cnn_flat(dropout):
    """Simple CNN → flat 32-dim vector. For use with concat fusion."""
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # 48x48
        nn.GELU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 24x24
        nn.GELU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 12x12
        nn.GELU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),# 6x6
        nn.GELU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 32),
        nn.GELU(),
        nn.Dropout(dropout),
    )


def _build_cnn_spatial(dropout):
    """Simple CNN → spatial (B*T, 64, 3, 3) feature map. For use with cross_attention fusion."""
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # 48x48
        nn.GELU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 24x24
        nn.GELU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 12x12
        nn.GELU(),
        nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1), # 6x6
        nn.GELU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 3x3
        nn.GELU(),
        nn.Dropout(dropout),
    )


class ImageStateObservation(nn.Module):
    """
    Combines image and state observations.
    encoder_type: "resnet" (pretrained ResNet18) or "cnn" (lightweight CNN trained from scratch).
    fusion_type:  "concat" or "cross_attention".
    """

    def __init__(
        self,
        use_states=True,
        fusion_type: Literal["concat", "cross_attention"] = "concat",
        encoder_type: Literal["resnet", "cnn"] = "resnet",
        dropout=0.1,
    ):
        super().__init__()
        self.use_states = use_states
        self.fusion_type = fusion_type
        self.encoder_type = encoder_type

        # Default flat image projection (used by concat / no-state paths)
        if encoder_type == "resnet":
            self.image_projection = nn.Sequential(
                resnet18(),
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.GELU(),
            )
        else:  # cnn
            self.image_projection = _build_cnn_flat(dropout)

        if use_states:
            match fusion_type:
                case "cross_attention":
                    self.cross_attention = ImageStateCrossAttention(
                        seq_len=9,  # 3x3 spatial grid from encoder
                        embed_dim=64,
                        num_heads=4,
                        dropout=dropout,
                    )
                    self.state_projection = nn.Sequential(
                        nn.Linear(2, 8),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(8, 32),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(32, 64),
                        nn.GELU(),
                    )

                    # Override image_projection with spatial variant
                    if encoder_type == "resnet":
                        resnet = resnet18().resnet[:-1]
                        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        self.image_projection = nn.Sequential(
                            normalize,
                            resnet,
                            nn.Conv2d(512, 128, kernel_size=3, padding=1),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Conv2d(128, 64, kernel_size=3, padding=1),
                            nn.GELU(),
                        )
                    else:  # cnn
                        self.image_projection = _build_cnn_spatial(dropout)

                case "concat":
                    self.state_projection = nn.Sequential(
                        nn.Linear(2, 4),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(4, 8),
                        nn.GELU(),
                    )

                case _:
                    raise ValueError(f"Unsupported fusion type: {fusion_type}")

    def forward(self, images, states):
        B, T, C, H, W = images.shape
        images = images.reshape(B * T, C, H, W)

        if self.use_states:
            _, _, C_state = states.shape
            states = states.reshape(B * T, C_state)
            state_embeddings = self.state_projection(states)
            image_embeddings = self.image_projection(images)

            match self.fusion_type:
                case "cross_attention":
                    output = self.cross_attention(state_embeddings, image_embeddings)
                case "concat":
                    output = torch.cat([image_embeddings, state_embeddings], dim=-1)

            output = output.reshape(B, T, -1)

        else:
            output = self.image_projection(images)
            output = output.reshape(B, T, -1)

        return output


if __name__ == "__main__":
    from src.data.dataloader import get_dataloaders

    trainloader, testloader = get_dataloaders()
    batch = next(iter(trainloader))
    images, states, actions = batch["observation.image"], batch["observation.state"], batch["action"]

    observation_history_maker = ImageStateObservation(True)
    observation_history = observation_history_maker(images, states)
    pass

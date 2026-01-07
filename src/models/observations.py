from typing_extensions import Literal
import torch.nn as nn
import torch
from src.models.resnet import resnet18


class ImageStateCrossAttention(nn.Module):
    def __init__(self, seq_len, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_embd = nn.Parameter(torch.zeros(seq_len, embed_dim))
    
    def forward(self, state_embeddings, image_embeddings):
        # image_embeddings: (B*T,E, H, W)
        # state_embeddings: (B*T, E)
        # Reshape to (B*T, H*W, E) and (B*T, 1, E) for attention
        img_emb_reshaped = image_embeddings.flatten(2).transpose(1,2)  # (B*T, H*W, E)
        img_emb_reshaped = img_emb_reshaped + self.pos_embd.unsqueeze(0) # provide position embedding
        state_emb_reshaped = state_embeddings.unsqueeze(1)  # (B*T, 1, E)

        attn_output, _ = self.attention(state_emb_reshaped, img_emb_reshaped, img_emb_reshaped)
        attn_output = attn_output.squeeze(1)  # (B*T, E)

        # Add & Norm
        output = self.norm(state_embeddings + self.dropout(attn_output))
        return output


class ImageStateObservation(nn.Module):
    """
    A simple module that passes the observed image through resnet and combines the feature vector with states.
    The module can also be used as resnet encoder alone without using the observation.states by setting use_states = False.
    The module can combine image with state via simple concatenation and MLP modules.
    """

    def __init__(self, use_states=True, fusion_type: Literal["concat", "cross_attention"] = "concat", dropout=0.1):
        super().__init__()
        self.use_states = use_states
        self.fusion_type = fusion_type

        self.image_projection = nn.Sequential(
            resnet18(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),  # final embedding size
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
        )

        if use_states:
            match fusion_type:
                case "cross_attention":
                    self.cross_attention = ImageStateCrossAttention(
                        seq_len=9,  # assuming resnet output spatial size is 3x3
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
                    
                    resnet = resnet18().resnet[:-1] 

                    self.image_projection = nn.Sequential(
                        resnet, 
                        nn.Conv2d(512, 128, kernel_size=3, padding=1),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Conv2d(128, 64, kernel_size=3, padding=1),  # final embedding size
                        nn.GELU(),
                    )

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
        # get image features
        B, T, C, H, W = images.shape
        images = images.reshape(B * T, C, H, W)

        if self.use_states:
            # project states so that they talk the resnet language
            _, _, C_state = states.shape
            states = states.reshape(B * T, C_state)
            state_embeddings = self.state_projection(states)
            image_embeddings = self.image_projection(images)
           
            match self.fusion_type:
                case "cross_attention":                    
                    output = self.cross_attention(state_embeddings, image_embeddings)

                case "concat":
                    output = torch.cat([image_embeddings, state_embeddings], dim=-1)

            # reshape output to batch, sequence, dim
            output = output.reshape(B, T, -1)

        else:
            output = self.image_projection(images)
            output = output.reshape(B, T, -1)

        return output


if __name__ == "__main__":
    from src.data.dataloader import get_dataloaders

    trainloader, testloader = get_dataloaders()

    batch = next(iter(trainloader))  # get a single batch of data

    images, states, actions = (
        batch["observation.image"],
        batch["observation.state"],
        batch["action"],
    )

    observation_history_maker = ImageStateObservation(True)
    observation_history = observation_history_maker(images, states)
    pass

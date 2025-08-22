import torch
import torch.nn as nn
from sklearn.cluster import KMeans

class EncoderDecoder(nn.Module):
    def __init__(self, action_dim, num_bins, actions, device):
        super().__init__()
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.device = device

        # Fit KMeans once
        kmeans = KMeans(n_clusters=num_bins)
        kmeans.fit(actions.cpu())

        # Save centroids as buffer to be included in state_dict
        self.register_buffer("centroids", torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))

    def encode(self, action):
        B, T, D = action.shape
        action_flat = action.reshape(B*T, D)

        # Compute nearest cluster centers
        dists = torch.cdist(action_flat.cpu(), self.centroids.cpu())
        action_bin = torch.argmin(dists, dim=1)
        action_center = self.centroids[action_bin]

        action_residual = action_flat.to(self.device) - action_center.to(self.device)
        action_residual = action_residual.reshape(B, T, D)
        action_bin = action_bin.reshape(B, T, 1)

        return action_bin.to(torch.int64), action_residual

    def decode(self, action_bin, action_residual):
        action_center = self.centroids[action_bin.squeeze(-1)].to(self.device)
        return action_center + action_residual

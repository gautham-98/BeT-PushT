import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class EncoderDecoder(nn.Module):
    def __init__(self, action_dim, num_bins, actions=None, device="cuda"):
        super().__init__()
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.device = device

        if actions is not None:
            # Fit KMeans once
            kmeans = KMeans(n_clusters=num_bins)
            kmeans.fit(actions.cpu())
            cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        else:
            # Load centroids from saved buffer during inference
            cluster_centers = torch.rand(size=(num_bins, action_dim), dtype=torch.float32)

        # Save centroids as buffer to be included in state_dict
        self.register_buffer("centroids", cluster_centers)
        
        # Save K-Means plot
        if actions is not None and action_dim == 2:  # Only plot for 2D actions
            self._save_kmeans_plot(actions.cpu().numpy(), kmeans, 'kmeans_action_clusters.png')
    
    def _save_kmeans_plot(self, actions, kmeans, plot_path):
        """Visualize and save K-Means clustering of actions"""
        import numpy as np
        
        # Get cluster assignments
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        
        # Calculate cluster populations
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique_labels, counts))
        
        # Map each point to its cluster's population
        population_colors = np.array([cluster_sizes[label] for label in labels])
        
        plt.figure(figsize=(10, 8))
        
        # Plot all actions colored by cluster population
        scatter = plt.scatter(actions[:, 0], actions[:, 1], 
                            c=population_colors, cmap='plasma', 
                            alpha=0.5, s=10, label='Actions')
        
        # Plot centroids sized by cluster population
        centroid_colors = [cluster_sizes[i] for i in range(len(centroids))]
        plt.scatter(centroids[:, 0], centroids[:, 1], 
                   c=centroid_colors, cmap='plasma',
                   marker='X', s=300, 
                   edgecolors='green', linewidths=2,
                   label='Centroids')
        
        # Add centroid labels with population counts
        for i, centroid in enumerate(centroids):
            count = cluster_sizes.get(i, 0)
            plt.annotate(f'Bin {i}\n({count})', 
                        xy=centroid, 
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.3', 
                                 facecolor='yellow', 
                                 alpha=0.7))
        
        plt.colorbar(scatter, label='Cluster Population')
        plt.xlabel('Action Dimension 0 (x)')
        plt.ylabel('Action Dimension 1 (y)')
        plt.title(f'K-Means Clustering of Actions (K={self.num_bins})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # Set (0,0) at top-left to match image coordinates
        plt.tight_layout()
        
        # Save plot
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"K-Means plot saved to: {plot_path}")
        
        # Calculate cluster spread (standard deviation from centroid)
        cluster_spreads = {}
        for i in range(len(centroids)):
            cluster_points = actions[labels == i]
            if len(cluster_points) > 0:
                # Calculate distances from centroid
                distances = np.linalg.norm(cluster_points - centroids[i], axis=1)
                cluster_spreads[i] = {
                    'mean_dist': np.mean(distances),
                    'std_dist': np.std(distances),
                    'max_dist': np.max(distances)
                }
        
        # Print cluster statistics
        print(f"\nCluster Statistics:")
        print(f"Total actions: {len(actions)}")
        
        # Calculate average spread across all clusters
        avg_mean_dist = np.mean([s['mean_dist'] for s in cluster_spreads.values()])
        avg_std_dist = np.mean([s['std_dist'] for s in cluster_spreads.values()])
        avg_max_dist = np.mean([s['max_dist'] for s in cluster_spreads.values()])
        
        print(f"Average cluster spread:")
        print(f"  mean_dist={avg_mean_dist:.4f}, std={avg_std_dist:.4f}, max={avg_max_dist:.4f}")


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

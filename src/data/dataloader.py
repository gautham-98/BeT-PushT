import numpy as np
import torch
import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from types import MethodType

def _make_subset_dataset(repo_id, episodes, delta_timestamps):
    """Create a LeRobotDataset subset dataset on specified episodes.
        This is necessary because we have to remap the global episode idx (referred with full dataset) 
        to local episode idx referring to subset dataset. 
    """
    ds = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps, episodes=episodes)

    # global to local map
    g2l = {int(g): i for i, g in enumerate(episodes)}

    # keep original method
    _orig = ds._get_query_indices

    # patch to covert the global ep_idx to local ep_idx_local
    def _patched(self, idx, ep_idx):
        ep_idx_local = g2l.get(int(ep_idx), ep_idx)
        return _orig(idx, ep_idx_local)

    ds._get_query_indices = MethodType(_patched, ds)
    return ds

def get_dataloaders(repo_id="lerobot/pusht", timestep=0.1, h=10, batch_size=32, num_workers=0): 

    timestamps = [round(-timestep*count,1) for count in range(1,h+1)][::-1]

    delta_timestamps = {
        "observation.image": timestamps,
        "observation.state": timestamps,
        "action": timestamps,
    }
    
    # create train validation splits
    ds_meta = LeRobotDatasetMetadata(repo_id)
    all_episodes = list(range(ds_meta.total_episodes))

    train_eps, val_eps = train_test_split(all_episodes, train_size=0.8, test_size=0.2, shuffle=True) 
    
    # create train and val dataset
    train_dataset = _make_subset_dataset(repo_id, delta_timestamps=delta_timestamps, episodes=train_eps)
    val_dataset = _make_subset_dataset(repo_id, delta_timestamps=delta_timestamps, episodes=val_eps)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
    )        
    return train_dataloader, val_dataloader

if __name__ == "__main__":
    trainloader, testloader = get_dataloaders()
    
    for batch in testloader:
        states = batch['observation.state']
        print(states.shape)
import numpy as np
import torch
import torchvision.transforms.v2 as T
import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import hf_transform_to_torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from types import MethodType
from src.utils.data_utils import Normalise

train_augmentation = T.Compose([
    T.ToImage(),  # Ensures input is a tensor image
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Always applies, but with random factors
    T.RandomGrayscale(p=0.3),
    # T.RandomSolarize(threshold=0.5, p=0.3),
    T.RandomAutocontrast(p=0.3),
    T.RandomEqualize(p=0.3),
    T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    # T.GaussianNoise(mean=0.0, sigma=0.1),
    T.ToDtype(torch.float32, scale=True),  # Ensures output is float tensor in [0,1]
])


def normalise_coords(sample:dict) -> dict:
    """Normalize the observations and actions in the sample dictionary."""
    for name in ['observation.state', 'action']:
        if name in sample:
            sample[name] = Normalise.forward(sample[name], name).astype(np.float32)
    return hf_transform_to_torch(sample)

def _make_subset_dataset(repo_id, episodes, delta_timestamps, augment=None):
    """
    Create a LeRobotDataset subset dataset on specified episodes.
    This is necessary because we have to remap the global episode idx (referred with full dataset)
    to local episode idx referring to subset dataset.
    """
    ds = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps, episodes=episodes, image_transforms=augment)
    
    # normalise coords
    ds.hf_dataset.set_transform(normalise_coords)

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


def get_dataloaders(
    repo_id="lerobot/pusht", timestep=0.1, h=40, batch_size=32, num_workers=0
):
    timestamps = [round(-timestep * count, 1) for count in range(1, h + 1)][::-1]

    delta_timestamps = {
        "observation.image": timestamps,
        "observation.state": timestamps,
        "action": timestamps,
    }

    # create train validation splits
    ds_meta = LeRobotDatasetMetadata(repo_id)
    
    # set normalisation stats 
    Normalise.set_stats(ds_meta.stats)

    # train val split
    all_episodes = list(range(ds_meta.total_episodes))
    train_eps, val_eps = train_test_split(
        all_episodes, train_size=0.8, test_size=0.2, shuffle=True
    )

    # create train and val dataset
    train_dataset = _make_subset_dataset(
        repo_id, delta_timestamps=delta_timestamps, episodes=train_eps, augment=None
    )
    val_dataset = _make_subset_dataset(
        repo_id, delta_timestamps=delta_timestamps, episodes=val_eps
    )

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


def main():
    import matplotlib.pyplot as plt

    trainloader, testloader = get_dataloaders()

    for i, batch in enumerate(trainloader):
        image = (
            (batch["observation.image"].permute(0, 1, 3, 4, 2).cpu().numpy() * 255)
            .astype(np.uint8)
            .reshape(32, 96 * 40, 96, 3)
            .transpose(1, 0, 2, 3)
            .reshape(96 * 40, 32 * 96, 3)
            .transpose(1, 0, 2)
        )
        # img = batch["observation.image"][0, 0]  # shape: (C, H, W)
        # action = batch["observation.state"][0, 0]
        # print(action)
        # img = img.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        plt.imsave(f"batch{i}.png", image)
        action = batch["action"]
        print(action)
        observation = batch["observation.state"]
        print(observation)
        break
        

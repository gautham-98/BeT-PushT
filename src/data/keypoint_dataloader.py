import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import hf_transform_to_torch
from sklearn.model_selection import train_test_split
from types import MethodType

from src.utils.data_utils import Normalise
from detect_keypoints_mine import detect_keypoints

N_PTS = 16


def _normalise_coords(sample: dict) -> dict:
    for name in ["observation.state", "action"]:
        if name in sample:
            sample[name] = Normalise.forward(sample[name], name).astype(np.float32)
    return hf_transform_to_torch(sample)


def _make_subset_dataset(repo_id, episodes, delta_timestamps):
    ds = LeRobotDataset(
        repo_id, delta_timestamps=delta_timestamps, episodes=episodes, video_backend="pyav"
    )
    ds.hf_dataset.set_transform(_normalise_coords)

    g2l = {int(g): i for i, g in enumerate(episodes)}
    _orig = ds._get_query_indices

    def _patched(self, idx, ep_idx):
        return _orig(idx, g2l.get(int(ep_idx), ep_idx))

    ds._get_query_indices = MethodType(_patched, ds)
    return ds


class KeypointDataset(Dataset):
    """
    Wraps a LeRobotDataset, extracts contour keypoints from each image frame,
    and returns them in place of the images to keep batch memory small.
    """

    def __init__(self, base_dataset, n_pts: int = N_PTS):
        self.base = base_dataset
        self.n_pts = n_pts

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]

        images = sample["observation.image"]  # (T, C, H, W) float32 [0, 1]
        T_len = images.shape[0]

        # (T, n_pts*2) normalised to [-1, 1]; zeros on detection failure
        kps = torch.zeros(T_len, self.n_pts * 2, dtype=torch.float32)
        for t in range(T_len):
            img_np = (images[t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            kp, _ = detect_keypoints(img_np, n_pts=self.n_pts)
            if kp is not None:
                # world coords [0, 512] → [-1, 1]
                kps[t] = torch.from_numpy(kp.reshape(-1) / 256.0 - 1.0)

        sample["observation.keypoints"] = kps
        del sample["observation.image"]  # not needed for training
        return sample


def get_keypoint_dataloaders(
    repo_id="lerobot/pusht",
    timestep=0.1,
    h=5,
    batch_size=256,
    num_workers=8,
    n_pts=N_PTS,
):
    timestamps = [round(-timestep * i, 1) for i in range(1, h + 1)][::-1]
    delta_timestamps = {
        "observation.image": timestamps,
        "observation.state": timestamps,
        "action": timestamps,
    }

    ds_meta = LeRobotDatasetMetadata(repo_id)
    Normalise.set_stats(ds_meta.stats)

    all_eps = list(range(ds_meta.total_episodes))
    train_eps, val_eps = train_test_split(all_eps, train_size=0.8, test_size=0.2, shuffle=True, random_state=42)

    train_dataset = KeypointDataset(_make_subset_dataset(repo_id, train_eps, delta_timestamps), n_pts=n_pts)
    val_dataset   = KeypointDataset(_make_subset_dataset(repo_id, val_eps,   delta_timestamps), n_pts=n_pts)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    return train_loader, val_loader

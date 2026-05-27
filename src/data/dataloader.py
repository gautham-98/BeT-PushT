import zarr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# positions in [0, 512] → [-1, 1];  angle in [0, 2π] → [-1, 1]
_POS_SCALE   = 256.0
_ANGLE_SCALE = np.pi


def normalise_state(state: np.ndarray) -> np.ndarray:
    out = state.copy().astype(np.float32)
    out[..., :4] = out[..., :4] / _POS_SCALE - 1.0
    out[..., 4]  = out[..., 4]  / _ANGLE_SCALE - 1.0
    return out


def normalise_action(action: np.ndarray) -> np.ndarray:
    return (action.astype(np.float32) / _POS_SCALE - 1.0)


def denormalise_action(action: np.ndarray) -> np.ndarray:
    return (action + 1.0) * _POS_SCALE


class PushTStateDataset(Dataset):
    def __init__(self, states: np.ndarray, actions: np.ndarray,
                 episode_ends: np.ndarray, h: int = 5):
        """
        states       : (N, 5)  normalised — [agent_xy, block_xy, block_angle]
        actions      : (N, 2)  normalised — target agent position
        episode_ends : (E,)    exclusive end index of each episode
        h            : sequence length
        """
        self.states  = states
        self.actions = actions
        self.h       = h

        # Only include timesteps where a full h-length history is available
        # (skip the first h-1 steps of each episode to avoid padding)
        self.samples = []
        ep_start = 0
        for ep_end in episode_ends:
            for t in range(ep_start + h - 1, int(ep_end)):
                self.samples.append(t)
            ep_start = int(ep_end)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t = self.samples[idx]
        start = t - self.h + 1
        return {
            "observation.state": torch.from_numpy(self.states[start:t + 1]),  # (h, 5)
            "action":            torch.from_numpy(self.actions[start:t + 1]),  # (h, 2)
        }


def get_dataloaders(
    zarr_path: str = "pusht.zarr",
    h: int = 5,
    batch_size: int = 256,
    num_workers: int = 8,
    train_ratio: float = 0.8,
    random_state: int = 42,
):
    root = zarr.open(zarr_path, mode="r")

    states_raw    = root["data/state"][:]         # (N, 5)
    actions_raw   = root["data/action"][:]        # (N, 2)
    episode_ends  = root["meta/episode_ends"][:]  # (E,)

    num_episodes  = len(episode_ends)
    all_eps       = list(range(num_episodes))
    train_eps, val_eps = train_test_split(
        all_eps, train_size=train_ratio, shuffle=True, random_state=random_state
    )

    # Rebuild per-episode end arrays for each split
    def _split_data(ep_indices):
        ep_indices = sorted(ep_indices)
        new_ends, new_states, new_actions = [], [], []
        cursor = 0
        for ep in ep_indices:
            start = int(episode_ends[ep - 1]) if ep > 0 else 0
            end   = int(episode_ends[ep])
            s = normalise_state(states_raw[start:end])
            a = normalise_action(actions_raw[start:end])
            new_states.append(s)
            new_actions.append(a)
            cursor += (end - start)
            new_ends.append(cursor)
        return (np.concatenate(new_states),
                np.concatenate(new_actions),
                np.array(new_ends))

    tr_s, tr_a, tr_e = _split_data(train_eps)
    va_s, va_a, va_e = _split_data(val_eps)

    train_ds = PushTStateDataset(tr_s, tr_a, tr_e, h=h)
    val_ds   = PushTStateDataset(va_s, va_a, va_e, h=h)

    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=True,
              prefetch_factor=2 if num_workers > 0 else None,
              persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    return train_loader, val_loader

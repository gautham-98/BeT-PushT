import argparse
import os
import yaml
import collections
from pathlib import Path

import numpy as np
import torch
import cv2
import gym_pusht  # noqa: F401 – registers gym_pusht namespace
import gymnasium

from src.models.bet import BeT
from src.models.keypoint_observation import KeypointStateObservation
from src.utils.data_utils import Normalise
from detect_keypoints_mine import detect_keypoints


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(Path(path)) as f:
        return yaml.safe_load(f)


def build_models(config: dict, device: str):
    obs_conf    = config["observation"]
    action_conf = config["action"]
    bet_conf    = config["bet_model"]
    data_conf   = config["data"]

    observation_module = KeypointStateObservation(
        n_pts=obs_conf["n_pts"],
        state_dim=obs_conf["state_dim"],
        embed_dim=obs_conf["observation_dim"],
        dropout=obs_conf.get("dropout", 0.1),
    ).to(device)

    # actions=None → inference mode (_train=False), centroids loaded from ckpt
    bet = BeT(
        observation_dim=obs_conf["observation_dim"],
        embedding_dim=obs_conf["embedding_dim"],
        num_transformer_layers=bet_conf["num_transformer_layers"],
        num_attention_heads=bet_conf["num_attention_heads"],
        action_dim=action_conf["action_dim"],
        num_bins=action_conf["num_bins"],
        sequence_length=data_conf["sequence_length"],
        actions=None,
        dropout=bet_conf.get("dropout", 0.1),
        device=device,
    ).to(device)

    return observation_module, bet


def load_checkpoint(ckpt_path: str, observation_module, bet, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    bet.load_state_dict(ckpt["bet_state_dict"])
    observation_module.load_state_dict(ckpt["observation_state_dict"])
    print(f"Loaded checkpoint from {ckpt_path}  (epoch {ckpt.get('epoch', '?')}, val_loss {ckpt.get('val_loss', '?'):.4f})")


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------

def run_rollout(env, observation_module, bet, sequence_length: int, n_pts: int,
                device: str, max_steps: int = 300, render_frames: bool = False):
    """
    Returns:
        success   : bool
        coverage  : float (final)
        total_reward : float
        frames    : list[np.ndarray] RGB frames if render_frames else []
    """
    obs_img, info = env.reset()   # obs_img: (96, 96, 3) uint8

    obs_buf = _make_obs(obs_img, info, n_pts, device)  # (1, obs_dim)
    window  = collections.deque([obs_buf] * sequence_length, maxlen=sequence_length)

    total_reward = 0.0
    coverage     = 0.0
    success      = False
    frames       = []

    for _ in range(max_steps):
        if render_frames:
            frames.append(env.render())

        # Stack window → (1, T, obs_dim) and run model
        obs_seq = torch.cat(list(window), dim=0).unsqueeze(0)  # (1, T, obs_dim)
        with torch.no_grad():
            action_norm = bet(obs_seq)   # (1, action_dim) normalised

        # Denormalise → raw [0, 512], then clamp to action space
        action_raw = Normalise.inverse(action_norm.cpu().numpy()[0], "action")
        action_raw = np.clip(action_raw, 0.0, 512.0).astype(np.float32)

        obs_img, reward, terminated, truncated, info = env.step(action_raw)

        total_reward += reward
        coverage = info.get("coverage", 0.0)
        success  = bool(info.get("is_success", False))

        # Update observation window
        obs_buf = _make_obs(obs_img, info, n_pts, device)
        window.append(obs_buf)

        if terminated or truncated:
            break

    if render_frames:
        frames.append(env.render())

    return success, coverage, total_reward, frames


def _make_obs(obs_img: np.ndarray, info: dict, n_pts: int, device: str) -> torch.Tensor:
    """
    Build a single-step observation tensor (1, obs_dim) from a raw env step.
    Keypoints: world coords → [-1,1].  State: normalised by dataset stats.
    """
    # Keypoints
    kp, _ = detect_keypoints(obs_img, n_pts=n_pts)
    if kp is not None:
        kp_norm = (kp.reshape(-1) / 256.0 - 1.0).astype(np.float32)
    else:
        kp_norm = np.zeros(n_pts * 2, dtype=np.float32)

    # Agent state (pos_agent is in [0, 512] world coords)
    state_raw = info["pos_agent"].astype(np.float32)
    state_norm = Normalise.forward(state_raw, "observation.state").astype(np.float32)

    obs = np.concatenate([kp_norm, state_norm])  # (n_pts*2 + 2,)
    return torch.from_numpy(obs).unsqueeze(0).to(device)  # (1, obs_dim)


# ---------------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------------

def save_video(frames: list, path: str, fps: int = 15):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, default="config_keypoint.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pth checkpoint. Defaults to evaluation.checkpoint_path in config.")
    parser.add_argument("--num_rollouts", type=int, default=None,
                        help="Override evaluation.num_rollout from config.")
    parser.add_argument("--max_steps",  type=int, default=None,
                        help="Override evaluation.max_episode_steps from config.")
    parser.add_argument("--save_video", action="store_true", default=None,
                        help="Override evaluation.save_video from config.")
    parser.add_argument("--video_dir",  type=str, default=None)
    args = parser.parse_args()

    config   = load_config(args.config)
    device   = config["device"]
    eval_conf = config["evaluation"]
    data_conf = config["data"]

    ckpt_path    = args.checkpoint   or eval_conf["checkpoint_path"]
    num_rollouts = args.num_rollouts or eval_conf["num_rollout"]
    max_steps    = args.max_steps    or eval_conf["max_episode_steps"]
    do_video     = args.save_video if args.save_video is not None else eval_conf["save_video"]
    video_dir    = args.video_dir  or eval_conf["video_dir"]

    if do_video:
        os.makedirs(video_dir, exist_ok=True)

    # Normalisation stats (written by the dataloader on first training run)
    Normalise.set_stats_from_file("normaliser.stats")

    # Models
    observation_module, bet = build_models(config, device)
    load_checkpoint(ckpt_path, observation_module, bet, device)
    observation_module.eval()
    bet.eval()

    # Wrap BeT forward to run through observation_module first
    def _policy(obs_seq_tensor):
        """obs_seq_tensor: (1, T, n_pts*2 + state_dim)"""
        # Split keypoints and state
        n_pts = data_conf["n_pts"]
        kps    = obs_seq_tensor[:, :, :n_pts * 2]    # (1, T, n_pts*2)
        states = obs_seq_tensor[:, :, n_pts * 2:]    # (1, T, state_dim)
        obs_emb = observation_module(kps, states)    # (1, T, obs_dim)
        return bet(obs_emb)                           # (1, action_dim)

    # Gym env
    gymnasium.register_envs(gym_pusht)
    # obs_type='pixels' always needs render_mode='rgb_array' to produce frames
    env = gymnasium.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels",
        render_mode="rgb_array",
    )

    n_pts = data_conf["n_pts"]
    seq_len = data_conf["sequence_length"]

    successes, coverages, rewards = [], [], []

    for ep in range(num_rollouts):
        obs_img, info = env.reset()

        obs_buf = _make_obs(obs_img, info, n_pts, device)
        window  = collections.deque([obs_buf] * seq_len, maxlen=seq_len)

        total_reward = 0.0
        coverage     = 0.0
        success      = False
        frames       = [] if do_video else None

        for step in range(max_steps):
            if do_video:
                frames.append(env.render())

            # (1, T, obs_dim) — raw concat of kps+state, not yet through obs module
            obs_seq = torch.cat(list(window), dim=0).unsqueeze(0)  # (1, T, n_pts*2+2)

            with torch.no_grad():
                action_norm = _policy(obs_seq)   # (1, action_dim)

            action_raw = Normalise.inverse(action_norm.cpu().numpy()[0], "action")
            action_raw = np.clip(action_raw, 0.0, 512.0).astype(np.float32)

            obs_img, reward, terminated, truncated, info = env.step(action_raw)
            total_reward += reward
            coverage = info.get("coverage", 0.0)
            success  = bool(info.get("is_success", False))

            window.append(_make_obs(obs_img, info, n_pts, device))

            if terminated or truncated:
                break

        if do_video:
            frames.append(env.render())
            vpath = os.path.join(video_dir, f"rollout_{ep:04d}.mp4")
            save_video(frames, vpath)

        successes.append(int(success))
        coverages.append(coverage)
        rewards.append(total_reward)

        print(f"  ep {ep+1:>4d}/{num_rollouts}  success={success}  coverage={coverage:.3f}  reward={total_reward:.2f}")

    env.close()

    print("\n=== Evaluation Results ===")
    print(f"  Success rate : {np.mean(successes)*100:.1f}%  ({sum(successes)}/{num_rollouts})")
    print(f"  Mean coverage: {np.mean(coverages):.4f} ± {np.std(coverages):.4f}")
    print(f"  Mean reward  : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")


if __name__ == "__main__":
    main()

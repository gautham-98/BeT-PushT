import argparse
import os
import yaml
import collections
from pathlib import Path

import numpy as np
import torch
import cv2
import gym_pusht  # noqa: F401
import gymnasium

from src.models.bet import BeT
from src.models.state_observation import StateObservation
from src.data.dataloader import normalise_state, denormalise_action


def load_config(path: str) -> dict:
    with open(Path(path)) as f:
        return yaml.safe_load(f)


def build_models(config: dict, device: str):
    obs_conf    = config["observation"]
    action_conf = config["action"]
    bet_conf    = config["bet_model"]
    data_conf   = config["data"]

    observation_module = StateObservation(
        state_dim=obs_conf["state_dim"],
        embed_dim=obs_conf["observation_dim"],
        dropout=obs_conf.get("dropout", 0.1),
    ).to(device)

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
    print(f"Loaded checkpoint: {ckpt_path}  (epoch {ckpt.get('epoch','?')}, val_loss {ckpt.get('val_loss', 0):.4f})")


def save_video(frames: list, path: str, fps: int = 10):
    if not frames:
        return
    path = str(path).rsplit(".", 1)[0] + ".avi"
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def _make_obs(info: dict, device: str) -> torch.Tensor:
    """Build a (1, 5) normalised state tensor from env step info."""
    state_raw = np.array([
        info["pos_agent"][0], info["pos_agent"][1],
        info["block_pose"][0], info["block_pose"][1],
        info["block_pose"][2] % (2 * np.pi),   # wrap to [0, 2π] — matches zarr training data
    ], dtype=np.float32)
    state_norm = normalise_state(state_raw[None])[0]   # (5,)
    return torch.from_numpy(state_norm).unsqueeze(0).to(device)  # (1, 5)


def _obs_from_reset(obs_raw: np.ndarray, info: dict, device: str) -> torch.Tensor:
    """Build a (1, 5) normalised state tensor from env.reset() output.
    Prefers the direct observation (always present) over reconstructing from info."""
    if obs_raw is not None and obs_raw.shape == (5,):
        state_norm = normalise_state(obs_raw[None])[0]
        return torch.from_numpy(state_norm).unsqueeze(0).to(device)
    return _make_obs(info, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       type=str, default="config.yaml")
    parser.add_argument("--checkpoint",   type=str, default=None)
    parser.add_argument("--num_rollouts", type=int, default=None)
    parser.add_argument("--max_steps",    type=int, default=None)
    parser.add_argument("--save_video",   action="store_true", default=None)
    parser.add_argument("--video_dir",    type=str, default=None)
    args = parser.parse_args()

    config    = load_config(args.config)
    device    = config["device"]
    eval_conf = config["evaluation"]
    data_conf = config["data"]

    ckpt_path    = args.checkpoint   or eval_conf["checkpoint_path"]
    num_rollouts = args.num_rollouts or eval_conf["num_rollout"]
    max_steps    = args.max_steps    or eval_conf["max_episode_steps"]
    do_video     = args.save_video if args.save_video is not None else eval_conf["save_video"]
    video_dir    = args.video_dir  or eval_conf["video_dir"]

    if do_video:
        os.makedirs(video_dir, exist_ok=True)

    observation_module, bet = build_models(config, device)
    load_checkpoint(ckpt_path, observation_module, bet, device)
    observation_module.eval()
    bet.eval()

    seq_len = data_conf["sequence_length"]

    gymnasium.register_envs(gym_pusht)
    env = gymnasium.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array", max_episode_steps=max_steps)

    successes, coverages, rewards = [], [], []

    for ep in range(num_rollouts):
        obs_raw, info = env.reset(seed=ep)

        obs_buf = _obs_from_reset(obs_raw, info, device)           # (1, 5)
        window  = collections.deque([obs_buf] * seq_len, maxlen=seq_len)

        total_reward = 0.0
        coverage     = 0.0
        success      = False
        frames       = [] if do_video else None

        for _ in range(max_steps):
            if do_video:
                frames.append(env.render())

            obs_seq = torch.cat(list(window), dim=0).unsqueeze(0)  # (1, T, 5)
            with torch.no_grad():
                obs_emb     = observation_module(obs_seq)           # (1, T, obs_dim)
                action_norm = bet(obs_emb)                          # (1, 2)

            action_raw = denormalise_action(action_norm.cpu().numpy()[0])
            action_raw = np.clip(action_raw, 0.0, 512.0).astype(np.float32)

            _, reward, terminated, truncated, info = env.step(action_raw)
            total_reward += reward
            coverage = info.get("coverage", 0.0)
            success  = bool(info.get("is_success", False))

            window.append(_make_obs(info, device))

            if terminated or truncated:
                break

        if do_video:
            frames.append(env.render())
            save_video(frames, os.path.join(video_dir, f"rollout_{ep:04d}.mp4"))

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

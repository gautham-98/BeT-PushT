from pathlib import Path
import argparse

import gym_pusht
import gymnasium as gym
import imageio
import numpy as np
import torch

from src.utils.config_utils import load_config, init_models
from src.utils import load_models_from_checkpoint


def evaluate_policy(config):
    # Create video directory
    output_directory = Path(config["evaluation"]["video_dir"])
    output_directory.mkdir(parents=True, exist_ok=True)

    # init and load models
    observation_module, bet = init_models(config)
    observation_module, bet, _ = load_models_from_checkpoint(
        config["evaluation"]["checkpoint_path"], observation_module, bet, config["device"]
    )
    observation_module.eval()
    bet.eval()

    # env
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=config["evaluation"]["max_episode_steps"],
    )

    successes, max_overlaps = [], []

    for run_id in range(config["evaluation"]["num_rollout"]):
        frames = []
        rewards = []
        obs_history_images = []
        obs_history_states = []

        obs, info = env.reset(seed=42 + run_id)
        done, step = False, 0
        frames.append(env.render())

        while not done:
            # extract state + image from env observation
            state = torch.tensor(obs["agent_pos"], dtype=torch.float32).unsqueeze(0).to(config["device"])
            image = torch.tensor(obs["pixels"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            image = image.to(config["device"])

            # store in history
            obs_history_images.append(image)
            obs_history_states.append(state)

            # keep only last h
            if len(obs_history_images) > config["data"]["sequence_length"]:
                obs_history_images.pop(0)
                obs_history_states.pop(0)

            if len(obs_history_images) == config["data"]["sequence_length"]:
                # stack into tensors
                images_tensor = torch.stack(obs_history_images, dim=1)   # (B=1, T, C, H, W)
                states_tensor = torch.stack(obs_history_states, dim=1)   # (B=1, T, state_dim)

                # encode observations + predict
                with torch.no_grad():
                    obs_encoded = observation_module(images_tensor, states_tensor)
                    action = bet(obs_encoded)

                numpy_action = action.squeeze(0).cpu().numpy()
            else:
                # warmup: zero action
                numpy_action = np.zeros(env.action_space.shape, dtype=np.float32)

            # step env
            obs, reward, terminated, truncated, info = env.step(numpy_action)
            rewards.append(reward)
            frames.append(env.render())

            done = terminated or truncated
            step += 1

        # success = terminated flag
        success = terminated
        successes.append(success)

        # max overlap reward
        max_overlap = max(rewards) if rewards else 0.0
        max_overlaps.append(max_overlap)

        print(f"[Run {run_id+1}/{config['evaluation']['max_episode_steps']}] "
              f"Success={success} | Max Overlap={max_overlap:.3f} | Steps={step}")

        # Save video
        if config["evaluation"]["save_video"]:
            fps = env.metadata["render_fps"]
            video_path = output_directory / f"bet_rollout_{run_id+1}.mp4"
            imageio.mimsave(str(video_path), np.stack(frames), fps=fps)
            print(f"  Saved video at {video_path}")

    # compute metrics
    success_rate = 100.0 * np.mean(successes)
    avg_max_overlap = np.mean(max_overlaps)

    print("\n=== Evaluation Results ===")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Avg Max Overlap Reward: {avg_max_overlap:.3f}")

    return success_rate, avg_max_overlap


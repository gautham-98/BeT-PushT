import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import os

from src.data.keypoint_dataloader import get_keypoint_dataloaders
from src.models.keypoint_observation import KeypointStateObservation
from src.models.bet import BeT
from src.training.trainer import Trainer
from src.training.losses import FocalLoss


class KeypointTrainer(Trainer):
    """Trainer subclass that reads observation.keypoints instead of observation.image."""

    def _step(self, batch):
        device = self.bet.device
        keypoints = batch["observation.keypoints"].to(device)  # (B, T, n_pts*2)
        states    = batch["observation.state"].to(device)      # (B, T, 2)
        actions   = batch["action"].to(device)                 # (B, T, 2)

        target_bins, target_residuals = self.bet.encoderDecoder.encode(actions)
        target_bins      = target_bins.to(device)
        target_residuals = target_residuals.to(device)

        observations = self.observation_module(keypoints, states)  # (B, T, embed_dim)
        preds        = self.bet(observations)

        logits       = preds["seq_action_bins_logits"]
        logits_flat  = logits.reshape(-1, logits.shape[-1])
        bins_flat    = target_bins.reshape(-1, 1)

        bin_loss      = self.focal_loss(logits_flat, bins_flat)
        residual_loss = self.mse(preds["selected_seq_action_residuals"], target_residuals)
        loss          = bin_loss + residual_loss * self.residual_loss_scale

        return loss, bin_loss, residual_loss


def load_config(path: str) -> dict:
    with open(Path(path)) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_keypoint.yaml")
    parser.add_argument("--run_name", type=str, default="bet_keypoint")
    args = parser.parse_args()

    config     = load_config(args.config)
    device     = config["device"]
    data_conf  = config["data"]
    obs_conf   = config["observation"]
    action_conf = config["action"]
    bet_conf   = config["bet_model"]
    train_conf = config["training"]

    # Data
    trainloader, valloader = get_keypoint_dataloaders(
        batch_size=data_conf["batch_size"],
        h=data_conf["sequence_length"],
        num_workers=data_conf["num_workers"],
        n_pts=data_conf["n_pts"],
    )

    # Observation encoder
    observation_module = KeypointStateObservation(
        n_pts=obs_conf["n_pts"],
        state_dim=obs_conf["state_dim"],
        embed_dim=obs_conf["observation_dim"],
        dropout=obs_conf["dropout"],
    ).to(device)

    # BeT model — action collection from training set for k-means init
    action_collection = torch.stack(trainloader.dataset.base.hf_dataset["action"], dim=0)
    bet = BeT(
        observation_dim=obs_conf["observation_dim"],
        embedding_dim=obs_conf["embedding_dim"],
        num_transformer_layers=bet_conf["num_transformer_layers"],
        num_attention_heads=bet_conf["num_attention_heads"],
        action_dim=action_conf["action_dim"],
        num_bins=action_conf["num_bins"],
        sequence_length=data_conf["sequence_length"],
        actions=action_collection,
        dropout=bet_conf.get("dropout", 0.1),
        device=device,
    ).to(device)

    # Train
    trainer = KeypointTrainer(
        observation_module=observation_module,
        bet=bet,
        trainloader=trainloader,
        valloader=valloader,
        epochs=train_conf["epochs"],
        learning_rate=train_conf["learning_rate"],
        weight_decay=train_conf["weight_decay"],
        betas=train_conf["betas"],
        gamma=train_conf["gamma"],
        residual_loss_scale=train_conf["residual_loss_scale"],
        eval_interval=train_conf["eval_interval"],
        save_interval=train_conf["save_interval"],
        ckpt_dir=train_conf["checkpoint_path"],
        run_name=args.run_name,
    )
    trainer.train()


if __name__ == "__main__":
    main()

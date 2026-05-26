import argparse
import yaml
from pathlib import Path
import torch

from src.data.dataloader import get_dataloaders
from src.models.state_observation import StateObservation
from src.models.bet import BeT
from src.training.trainer import Trainer


def load_config(path: str) -> dict:
    with open(Path(path)) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, default="config.yaml")
    parser.add_argument("--run_name", type=str, default="bet_pusht")
    args = parser.parse_args()

    config     = load_config(args.config)
    device     = config["device"]
    data_conf  = config["data"]
    obs_conf   = config["observation"]
    action_conf = config["action"]
    bet_conf   = config["bet_model"]
    train_conf = config["training"]

    # Data
    trainloader, valloader = get_dataloaders(
        zarr_path=data_conf["zarr_path"],
        h=data_conf["sequence_length"],
        batch_size=data_conf["batch_size"],
        num_workers=data_conf["num_workers"],
    )

    # Action collection for K-means init (normalised actions from training set)
    action_collection = torch.from_numpy(trainloader.dataset.actions).float()

    # Observation encoder
    observation_module = StateObservation(
        state_dim=obs_conf["state_dim"],
        embed_dim=obs_conf["observation_dim"],
        dropout=obs_conf["dropout"],
    ).to(device)

    # BeT model
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
    trainer = Trainer(
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

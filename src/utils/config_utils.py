import yaml
from pathlib import Path
import torch
from src.models.observations import ImageStateObservation
from src.models.bet import BeT
from src.data.dataloader import get_dataloaders
from src.training.trainer import Trainer
import src.utils as utils

def load_config(path="config.yaml"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file {path} not found.")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def init_data(config):
    batch_size = config["data"]["batch_size"]
    sequence_length = config["data"]["sequence_length"]
    trainloader, valloader = get_dataloaders(batch_size=batch_size, h=sequence_length, num_workers=config["data"]["num_workers"])
    return trainloader, valloader

def init_models(config, trainloader=None, device=None):
    device = device or config["device"]
    
    # Observation encoder
    obs_conf = config["observation"]
    observation_module = ImageStateObservation(use_states=obs_conf["use_states"]).to(device)
    
    # Action collection (needed by BeT)
    if trainloader is not None:
        action_collection = torch.stack(trainloader.dataset.hf_dataset['action'], dim=0)
    else: # load a dummy, the bin centroids can later be loaded from the checkpoint file
        action_collection = torch.rand(size=(10000,2))
    # BeT model
    bet_conf = config["bet_model"]
    action_conf = config["action"]
    data_conf = config["data"]
    
    bet = BeT(
        observation_dim=obs_conf["observation_dim"],
        embedding_dim=obs_conf["embedding_dim"],
        num_transformer_layers=bet_conf["num_transformer_layers"],
        num_attention_heads=bet_conf["num_attention_heads"],
        action_dim=action_conf["action_dim"],
        num_bins=action_conf["num_bins"],
        sequence_length=data_conf["sequence_length"],
        actions=action_collection,
        device=device
    ).to(device)
    return observation_module, bet

def init_trainer(config, observation_module, bet, trainloader, valloader):
    train_conf = config["training"]
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
        ckpt_dir=train_conf["checkpoint_path"]
    )
    return trainer

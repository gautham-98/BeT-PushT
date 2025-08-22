import torch
import torch.nn as nn

def freeze_module(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()
    return module

def load_models_from_checkpoint(ckpt_path, observation_module, bet, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    bet.load_state_dict(ckpt["bet_state_dict"])
    observation_module.load_state_dict(ckpt["observation_state_dict"])
    return observation_module, bet, ckpt
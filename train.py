import argparse
from src.utils.config_utils import load_config, init_data, init_models, init_trainer

# parse config file path
parser = argparse.ArgumentParser(description="Load YAML config")
parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
args = parser.parse_args()


# Load YAML config
config = load_config(args.config)

# Initialize data
trainloader, valloader = init_data(config)

# Initialize models
observation_module, bet = init_models(config, trainloader)

# Initialize trainer
trainer = init_trainer(config, observation_module, bet, trainloader, valloader)

# Start training
trainer.train()

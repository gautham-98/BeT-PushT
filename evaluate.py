import argparse

from src.utils.config_utils import load_config
from src.evaluation.evaluation import evaluate_policy

parser = argparse.ArgumentParser(description="Evaluate BeT policy")
parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
args = parser.parse_args()

config = load_config(args.config)
evaluate_policy(config)

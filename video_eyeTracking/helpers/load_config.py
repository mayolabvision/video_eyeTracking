import os
import json

def load_config(config_path=None):
    # Default to looking for config.json in the parent directory of this script
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}. Please provide a valid path to the config.json file.")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

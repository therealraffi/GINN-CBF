import argparse
import yaml
import os
from datetime import datetime
import ast

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Adapter-CBF Training Script")
    
    parser.add_argument("--config_path", type=str, default="config_adapter.yml", 
                        help="Path to the YAML config file")
    parser.add_argument("--hp_dict", type=str, default="", 
                        help="Override config hyperparameters using 'key:val;key2:val2' format")

    return parser.parse_args()

def load_config(config_path):
    """Loads the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def update_config_from_args(config, hp_dict):
    """Updates the config with command-line hyperparameter overrides."""
    if not hp_dict:
        return config  # No overrides provided
    
    overrides = dict(item.split(":") for item in hp_dict.split(";"))  # Parse key-value pairs
    
    def update_dict(config_section, overrides):
        """Helper to update nested config sections."""
        for key, val in overrides.items():
            for section in config_section:
                if key in config_section[section]:  
                    # Convert value type dynamically (int, float, bool, str)
                    try:
                        if "[" == val[0]:
                            config_section[section][key] = ast.literal_eval(val)
                        elif "." in val:
                            config_section[section][key] = float(val)
                        else:
                            config_section[section][key] = int(val)
                    except ValueError:
                        config_section[section][key] = val  # Default to string
        return config_section

    return update_dict(config, overrides)

def build_config():
    args = parse_args()
    config = load_config(args.config_path)
    config = update_config_from_args(config, args.hp_dict)
    return config

if __name__ == "__main__":
    # Parse CLI arguments
    print(build_config())
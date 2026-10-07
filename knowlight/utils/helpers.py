import os
import random
import yaml
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across torch, numpy, and python."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml_config(config_path: str, dataset_name: str = None) -> dict:
    """
    Load YAML configuration file safely with automatic dataset-specific hyperparameter overrides.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Determine target dataset name from argument or config
    target_dataset = dataset_name or config.get("dataset", {}).get("name")
    if target_dataset:
        norm_name = str(target_dataset).lower().strip().replace("-", "_").replace(" ", "_")
        
        # Dataset name aliases mapping
        aliases = {
            "movielens": "movielens",
            "ml_1m": "movielens",
            "ml1m": "movielens",
            "amazon_cd": "amazon_cd_2014",
            "amazon_cd_2014": "amazon_cd_2014",
            "cd": "amazon_cd_2014",
            "amazon_toys": "amazon_toys_and_games",
            "amazon_toys_and_games": "amazon_toys_and_games",
            "toys": "amazon_toys_and_games",
            "toys_and_games": "amazon_toys_and_games",
            "amazon_beauty": "amazon_beauty",
            "beauty": "amazon_beauty",
            "yelp": "yelp"
        }
        canonical_name = aliases.get(norm_name, norm_name)
        
        # Known optimal dataset-specific hyperparameters
        default_dataset_hyperparams = {
            "movielens": {"lr": 0.01, "weight_decay": 0.001},
            "amazon_cd_2014": {"lr": 0.001, "weight_decay": 0.0001},
            "amazon_toys_and_games": {"lr": 0.001, "weight_decay": 0.001},
            "amazon_beauty": {"lr": 0.001, "weight_decay": 0.001},
            "yelp": {"lr": 0.001, "weight_decay": 0.0001}
        }
        
        # Merge YAML overrides if present in the config file
        yaml_overrides = config.get("dataset_hyperparams", {})
        merged_hyperparams = {**default_dataset_hyperparams, **yaml_overrides}
        
        if canonical_name in merged_hyperparams:
            ds_cfg = merged_hyperparams[canonical_name]
            config.setdefault("training", {})
            for param, val in ds_cfg.items():
                config["training"][param] = val
                
        config.setdefault("dataset", {})["name"] = canonical_name
        lr_val = config.get("training", {}).get("lr")
        wd_val = config.get("training", {}).get("weight_decay")
        print(f"[Config] Dataset: '{canonical_name}' -> Applied hyperparameters: lr={lr_val}, weight_decay={wd_val}")

    return config


def get_device(device_setting: str = "cuda") -> torch.device:
    """Get PyTorch torch.device object with automatic fallback if CUDA is unavailable."""
    if device_setting == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

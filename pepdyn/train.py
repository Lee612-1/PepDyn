from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> Dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["_config_path"] = str(path.resolve())
    config_dir = path.resolve().parent

    for key in ["output_dir"]:
        value = Path(config[key])
        if not value.is_absolute():
            config[key] = str((config_dir.parent / value).resolve())

    if "data" in config and "lmdb_path" in config["data"]:
        lmdb_path = Path(config["data"]["lmdb_path"])
        if not lmdb_path.is_absolute():
            config["data"]["lmdb_path"] = str((config_dir.parent / lmdb_path).resolve())
    for section, key in [
        ("split", "train_keys_path"),
        ("split", "test_keys_path"),
        ("split", "val_keys_path"),
    ]:
        if section in config and key in config[section] and config[section][key]:
            value = Path(config[section][key])
            if not value.is_absolute():
                config[section][key] = str((config_dir.parent / value).resolve())
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denormalize(value: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return value * std + mean


def pick_device(preferred: str) -> torch.device:
    return torch.device(preferred if torch.cuda.is_available() else "cpu")


def save_json(payload: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))

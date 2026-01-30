import os
import torch
from matplotlib import pyplot as plt
import json


def save_config(cfg):
    base_logs_path = os.path.join("logs", "logs_upload", cfg["exp_name"])
    base_weights_path = os.path.join("logs", "logs_weights", cfg["exp_name"], "weights")

    if not os.path.exists(base_logs_path):
        os.makedirs(base_logs_path)
    if not os.path.exists(base_weights_path):
        os.makedirs(base_weights_path)

    config_path = os.path.join(base_logs_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=4)
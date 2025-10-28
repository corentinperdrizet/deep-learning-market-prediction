# src/training/utils.py
import os
import random
from typing import Dict, Any
import numpy as np
import torch


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cudnn deterministic at the cost of some speed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    # Order: CUDA > MPS > CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(state: Dict[str, Any], path: str):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)

# src/training/utils.py
# Utility helpers for reproducible training, device selection, and checkpoints.
# Comments are in English only.

import os
import random
from typing import Dict, Any, Optional

import numpy as np
import torch


# --------------------------- Reproducibility ---------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set global random seeds for reproducibility across Python, NumPy, and PyTorch.
    Also configures cuDNN for deterministic behavior (slower but repeatable).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_global_seed(seed: int = 42) -> None:
    """
    Backward-compatible alias used elsewhere in the project.
    """
    set_seed(seed)


# --------------------------- Device selection ---------------------------

def get_device() -> torch.device:
    """
    Select the best available device in order: CUDA > MPS (Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Guard against rare cases where MPS is reported but unusable
        try:
            _ = torch.zeros(1, device="mps")
            return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


def select_device(prefer_cuda: bool = True, prefer_mps: bool = True) -> torch.device:
    """
    API used by MLflow runner scripts. Keeps the same semantics as get_device()
    but allows toggling CUDA/MPS preference via flags.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            _ = torch.zeros(1, device="mps")
            return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


def available_device_name() -> str:
    """
    Human-friendly device description (useful for logs).
    """
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------- Filesystem helpers ---------------------------

def ensure_dir(path: str) -> None:
    """
    Create a directory (and parents) if it does not exist.
    """
    if path:
        os.makedirs(path, exist_ok=True)


# --------------------------- Checkpointing ---------------------------

def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """
    Save a training checkpoint safely. Ensures parent directory exists.
    """
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)


# --------------------------- Model utilities (optional) ---------------------------

def count_parameters(model) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def move_to_device(obj: Any, device: Optional[torch.device] = None) -> Any:
    """
    Recursively move tensors (or dict/list/tuple containers of tensors) to a device.
    """
    if device is None:
        device = get_device()

    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [move_to_device(x, device) for x in obj]
        return type(obj)(t) if isinstance(obj, tuple) else t
    return obj

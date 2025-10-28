# src/training/dataloaders.py
from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from .utils import get_device


def make_dataset(
    X: np.ndarray,
    y: np.ndarray,
) -> TensorDataset:
    """
    X: (N, seq_len, n_features) float32
    y: (N,) in {0,1}
    """
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    return TensorDataset(X_t, y_t)


def make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Returns train_loader, val_loader, test_loader (test may be None).
    We DON'T shuffle sequences order globally; batching is fine.
    """
    # Deactivate warnings
    device = get_device()
    if str(device) == "mps":
        pin_memory = False

    train_set = make_dataset(X_train, y_train)
    val_set = make_dataset(X_val, y_val)
    test_set = make_dataset(X_test, y_test) if X_test is not None and y_test is not None else None

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,           # ok: mélange les séquences entre elles, pas l'ordre intra-séquence
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = (
        DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if test_set is not None
        else None
    )
    return train_loader, val_loader, test_loader

# src/loader_utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple
import torch
import random
import numpy as np

def collate_h5(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Collate para H5:
      Entrada: lista de B tuplas (x_i, y_i)
        - x_i: (T,C,H,W)
        - y_i: (1,)
      Devuelve:
        - X: (T,B,C,H,W)
        - y: (B,1)
    """
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)                 # (B,T,C,H,W)
    if x.ndim != 5:
        raise ValueError(f"Esperaba 5D (B,T,C,H,W); recibido {tuple(x.shape)}")
    x = x.permute(1, 0, 2, 3, 4).contiguous()  # (T,B,C,H,W)
    y = torch.stack(ys, dim=0)                 # (B,1)
    return x, y

def seed_worker(worker_id: int) -> None:
    """Inicialización determinista por worker (recomendación PyTorch)."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

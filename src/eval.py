# src/eval.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, Dict, Optional

import torch
from torch import nn
from torch.amp import autocast

from .nn_io import _forward_with_cached_orientation

def _align_target_shape(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Alinea la forma del target con la predicción (B,)↔(B,1)."""
    if y_hat.ndim == 2 and y_hat.shape[1] == 1 and y.ndim == 1:
        return y.unsqueeze(1)
    if y_hat.ndim == 1 and y.ndim == 2 and y.shape[1] == 1:
        return y.squeeze(1)
    return y

@torch.no_grad()
def eval_loader(
    loader,
    model: nn.Module,
    loss_fn: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    use_amp: Optional[bool] = None,
) -> Tuple[float, float]:
    """
    Devuelve (mse_medio_por_batch, mae_medio_por_batch).

    Compatibilidad hacia atrás:
      - Si 'loss_fn' es None, se usa nn.MSELoss().
      - Si 'device' y/o 'use_amp' no se pasan, se infieren:
          device  := device del primer parámetro del modelo (o 'cuda' si hay)
          use_amp := (device.type == 'cuda')
    """
    # Inferencias
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_amp is None:
        use_amp = (device.type == "cuda")
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    model.eval()
    v_running_mse = 0.0
    v_running_mae = 0.0
    n_val_batches = 0
    phase_hint: Dict[str, str] = {"val": None}

    for x, y in loader:
        y = y.to(device, non_blocking=True)
        y_hat = _forward_with_cached_orientation(
            model=model, x=x, y=y, device=device, use_amp=bool(use_amp), phase_hint=phase_hint, phase="val"
        )
        y_aligned = _align_target_shape(y_hat, y).to(device=y_hat.device, dtype=y_hat.dtype, non_blocking=True)
        with autocast("cuda", enabled=bool(use_amp)):
            v_loss = loss_fn(y_hat, y_aligned)
        v_running_mse += float(v_loss.detach().item())
        mae_batch = torch.abs(y_hat.to(torch.float32) - y_aligned.to(torch.float32)).mean()
        v_running_mae += float(mae_batch.detach().item())
        n_val_batches += 1

    mse = v_running_mse / max(1, n_val_batches)
    mae = v_running_mae / max(1, n_val_batches)
    return mse, mae

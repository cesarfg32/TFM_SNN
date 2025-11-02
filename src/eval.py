# src/eval.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, Dict

import torch
from torch import nn
from torch.amp import autocast

# Reutilizamos el helper de train para garantizar
# la misma lógica de orientación y runtime-encode.
from .training import _forward_with_cached_orientation


def _align_target_shape(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Alinea la forma del target con la predicción:
    - Si y_hat=(B,1) y y=(B,) -> y=(B,1)
    - Si y_hat=(B,)  y y=(B,1) -> y=(B,)
    """
    if y_hat.ndim == 2 and y_hat.shape[1] == 1 and y.ndim == 1:
        return y.unsqueeze(1)
    if y_hat.ndim == 1 and y.ndim == 2 and y.shape[1] == 1:
        return y.squeeze(1)
    return y


@torch.no_grad()
def eval_loader(
    loader,
    model: nn.Module,
    device: torch.device,
    use_amp: bool | None = None,
) -> Tuple[float, float]:
    """
    Evalúa MAE y MSE sobre un DataLoader.

    - Soporta entradas 4D (B,C,H,W) y 5D en ambas orientaciones:
      (T,B,C,H,W) y (B,T,C,H,W). La orientación correcta se decide
      con el mismo helper usado en training.
    - Si runtime-encode está activo (lo gestiona el runner con set_encode_runtime),
      el helper lo aplicará igual que en training.
    - Alinea y y_hat en forma, device y dtype antes de calcular métricas.
    """
    model.eval()
    if use_amp is None:
        use_amp = bool(torch.cuda.is_available())

    sum_abs = 0.0
    sum_sq = 0.0
    n = 0

    # Usamos la misma clave de fase que en validación ("val")
    phase_hint: Dict[str, str] = {"val": None}

    for x, y in loader:
        # y al device antes de pasar al helper (este lo usa para inferir B)
        y = y.to(device, non_blocking=True)

        # MISMA ruta que en training: corrige (B,T,...) -> (T,B,...)
        # y aplica runtime-encode si procede.
        y_hat = _forward_with_cached_orientation(
            model=model,
            x=x,
            y=y,
            device=device,
            use_amp=use_amp,
            phase_hint=phase_hint,
            phase="val",
        )

        # Alinea formas/dtypes/devices para el cálculo
        y_aligned = _align_target_shape(y_hat, y).to(
            device=y_hat.device, dtype=y_hat.dtype, non_blocking=True
        )

        # Métricas por elementos
        diff = (y_hat - y_aligned).reshape(-1)
        sum_abs += diff.abs().sum().item()
        sum_sq += (diff * diff).sum().item()
        n += diff.numel()

    mae = sum_abs / max(1, n)
    mse = sum_sq / max(1, n)
    return float(mae), float(mse)

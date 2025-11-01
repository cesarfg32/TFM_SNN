# src/eval.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple, Dict

import torch
from torch import nn
from torch.amp import autocast


def _align_target_shape(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Alinea la forma del target con la predicción:
    - Si y_hat=(B,1) y y=(B,) -> y=(B,1)
    - Si y_hat=(B,)   y y=(B,1) -> y=(B,)
    """
    if y_hat.ndim == 2 and y_hat.shape[1] == 1 and y.ndim == 1:
        return y.unsqueeze(1)
    if y_hat.ndim == 1 and y.ndim == 2 and y.shape[1] == 1:
        return y.squeeze(1)
    return y


def _forward_with_cached_orientation_eval(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    phase_hint: Dict[str, str],
) -> torch.Tensor:
    """
    Forward para evaluación con detección de orientación 5D cacheada.
    - Mueve x,y a device (y se asegura después que y matchea y_hat en device/dtype).
    - Si x es 5D, prueba una vez 'ok' (as is) y si el batch no cuadra, prueba con permuta (1,0,2,3,4).
    - El resultado ('ok'|'permute') se cachea en phase_hint["eval"].
    """
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    if x.ndim != 5:
        with autocast("cuda", enabled=use_amp):
            return model(x)

    hint = phase_hint.get("eval", None)
    if hint is None:
        # Intento A: sin permutar
        with autocast("cuda", enabled=use_amp):
            y_hat_A = model(x)
        if y_hat_A.shape[0] == y.shape[0]:
            phase_hint["eval"] = "ok"
            return y_hat_A

        # Intento B: permutar a (T,B,...) -> (B,T,...)
        x_perm = x.permute(1, 0, 2, 3, 4).contiguous()
        with autocast("cuda", enabled=use_amp):
            y_hat_B = model(x_perm)
        if y_hat_B.shape[0] == y.shape[0]:
            phase_hint["eval"] = "permute"
            return y_hat_B

        # Si nada encaja, nos quedamos con A (caso raro)
        phase_hint["eval"] = "ok"
        return y_hat_A

    if hint == "ok":
        with autocast("cuda", enabled=use_amp):
            return model(x)
    else:
        x_perm = x.permute(1, 0, 2, 3, 4).contiguous()
        with autocast("cuda", enabled=use_amp):
            return model(x_perm)


@torch.no_grad()
def eval_loader(
    loader,
    model: nn.Module,
    device: torch.device,
    use_amp: bool | None = None,
) -> Tuple[float, float]:
    """
    Evalúa MAE y MSE sobre un DataLoader.
    - Soporta entradas 4D (offline) y 5D (B,T,...) o (T,B,...) automáticamente.
    - Alinea y y y_hat en forma, device y dtype.
    """
    model.eval()
    if use_amp is None:
        use_amp = bool(torch.cuda.is_available())

    sum_abs = 0.0
    sum_sq = 0.0
    n = 0

    # cache de orientación 5D para no probar en cada batch
    phase_hint: Dict[str, str] = {"eval": None}

    for x, y in loader:
        # forward con orientación cacheada
        y_hat = _forward_with_cached_orientation_eval(
            model=model, x=x, y=y, device=device, use_amp=use_amp, phase_hint=phase_hint
        )

        # Alinear target y asegurar mismo device/dtype que y_hat
        y = y.to(device, non_blocking=True)
        y_aligned = _align_target_shape(y_hat, y).to(device=y_hat.device, dtype=y_hat.dtype, non_blocking=True)

        # Diferencia y acumulación
        diff = (y_hat - y_aligned).reshape(-1)
        sum_abs += diff.abs().sum().item()
        sum_sq  += (diff * diff).sum().item()
        n       += diff.numel()

    mae = sum_abs / max(1, n)
    mse = sum_sq  / max(1, n)
    return float(mae), float(mse)

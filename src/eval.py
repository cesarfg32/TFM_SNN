# src/eval.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import src.training as training

def eval_loader(loader, model, device) -> tuple[float, float]:
    """Calcula MAE y MSE promedio sobre todo el loader."""
    model.eval()
    mae_sum = 0.0
    mse_sum = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = training._permute_if_needed(x).to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_hat = model(x)
            mae_sum += torch.mean(torch.abs(y_hat - y)).item() * len(y)
            mse_sum += torch.mean((y_hat - y) ** 2).item() * len(y)
            n += len(y)
    return (mae_sum / max(n, 1)), (mse_sum / max(n, 1))

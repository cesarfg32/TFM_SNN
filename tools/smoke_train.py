# tools/smoke_train.py
# -*- coding: utf-8 -*-
"""
Smoke test de entrenamiento sintético:
- Verifica forward, backward y AMP en tu GPU.
- Útil para depurar instalación sin tocar datasets reales.

Uso:
  python tools/smoke_train.py --steps 50 --T 10 --batch 8 --amp --seed 42
"""
from __future__ import annotations

# --- asegurar que la raíz del repo está en sys.path ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -------------------------------------------------------------

import argparse, time
import torch
from torch import nn, optim
from torch.amp import autocast, GradScaler

from src.models import SNNVisionRegressor
from src.utils import set_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50, help="Iteraciones de entrenamiento")
    parser.add_argument("--T", type=int, default=10, help="Ventana temporal (timesteps)")
    parser.add_argument("--batch", type=int, default=8, help="Tamaño de batch")
    parser.add_argument("--amp", action="store_true", help="Activar AMP (float16) en CUDA")
    parser.add_argument("--seed", type=int, default=42, help="Semilla global para reproducibilidad")
    args = parser.parse_args()

    set_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device.type)

    # Modelo simple (entrada sintética de 1 canal)
    model = SNNVisionRegressor(in_channels=1, lif_beta=0.95).to(device)

    # Datos sintéticos: (B,T,C,H,W) -> SNNVisionRegressor espera (T,B,C,H,W)
    B, T, C, H, W = args.batch, args.T, 1, 66, 200

    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(args.amp and device_type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    t0 = time.time()
    for i in range(1, args.steps + 1):
        # Batch sintético determinista para reproducibilidad
        x = torch.randn(B, T, C, H, W, device=device)
        y = torch.randn(B, 1, device=device) * 0.1

        # (B,T,C,H,W) -> (T,B,C,H,W)
        x = x.permute(1, 0, 2, 3, 4).contiguous()

        opt.zero_grad(set_to_none=True)
        with autocast(device_type=device_type, enabled=use_amp):
            y_hat = model(x)
            loss = loss_fn(y_hat, y)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        if i % 10 == 0 or i == args.steps:
            print(f"Iter {i}/{args.steps} - loss {loss.item():.6f}")

    elapsed = time.time() - t0
    print(f"OK: entrenamiento sintético completado en {elapsed:.2f}s, última loss={loss.item():.6f}")


if __name__ == "__main__":
    main()

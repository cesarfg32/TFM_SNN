#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Smoke test de entrenamiento sin datos reales.

- Genera un dataset sintético (spikes rate) con una etiqueta correlacionada.
- Carga el SNN del proyecto y ejecuta un minibucle de entrenamiento (pocas steps).
- Útil para validar el stack (PyTorch, snnTorch, AMP, GPU) sin depender del dataset.

Uso:
  python tools/smoke_train.py --steps 50 --T 10 --H 80 --W 160 --batch 8
"""
import argparse, time, sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

# Asegura que podemos importar src/...
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.models import SNNVisionRegressor  # backbone del proyecto

class DummySpikesDataset(Dataset):
    """Dataset sintético: genera spikes (T,1,H,W) ~ Bernoulli(p)
       y un target y = k * mean(x) + ruido, para que el modelo pueda aprender algo.
    """
    def __init__(self, n=128, T=10, H=80, W=160, p=0.1, noise=0.05, seed=42):
        g = torch.Generator().manual_seed(seed)
        self.X = torch.rand((n, T, 1, H, W), generator=g) < p
        self.X = self.X.float()
        m = self.X.mean(dim=(1,2,3,4))  # (n,)
        self.y = (0.5 * m + noise * torch.randn(n, generator=g)).unsqueeze(1).float()

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50, help="nº de iteraciones (batches) a entrenar")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--H", type=int, default=80)
    ap.add_argument("--W", type=int, default=160)
    ap.add_argument("--p", type=float, default=0.1, help="densidad de spikes (rate)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--amp", action="store_true", help="usar mixed precision (si hay CUDA)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # Dataset + loader
    ds = DummySpikesDataset(n=args.steps * args.batch, T=args.T, H=args.H, W=args.W, p=args.p)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    # Modelo
    model = SNNVisionRegressor(in_channels=1, lif_beta=0.95).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())
    loss_fn = torch.nn.MSELoss()

    t0 = time.time()
    model.train(True)
    it = 0
    for x, y in dl:
        x = x.to(device)   # (B,T,1,H,W) -> el modelo espera (T,B,C,H,W)
        y = y.to(device)
        x = x.permute(1,0,2,3,4).contiguous()  # (T,B,C,H,W)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp and torch.cuda.is_available()):
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        it += 1
        if it % 10 == 0:
            print(f"Iter {it}/{args.steps} - loss {float(loss.detach().cpu()):.6f}")
        if it >= args.steps:
            break

    dt = time.time() - t0
    print(f"OK: entrenamiento sintético completado en {dt:.2f}s, última loss={float(loss.detach().cpu()):.6f}")

if __name__ == "__main__":
    main()
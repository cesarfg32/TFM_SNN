# src/training.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn, optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from .utils import set_seeds  # para reproducibilidad global


# ---------------------------------------------------------------------
# Acelerar FP32 en GPUs Ampere/Ada con TF32 (sin cambiar tu código)
# Recomendado para entreno: suele ofrecer un boost notable con
# impacto mínimo en precisión FP32.
# ---------------------------------------------------------------------
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------
# Configuración de entrenamiento
# ---------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 2
    batch_size: int = 8
    lr: float = 1e-3
    amp: bool = True
    seed: Optional[int] = None  # si se indica, fija reproducibilidad


# ---------------------------------------------------------------------
# Utilidad: permutar batch para el modelo SNN (T,B,C,H,W)
# - Los DataLoaders devuelven (B,T,C,H,W); el modelo espera (T,B,C,H,W).
# ---------------------------------------------------------------------
def _permute_if_needed(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 5 and x.shape[0] != x.shape[1]:
        # Asumimos (B,T,C,H,W) -> (T,B,C,H,W)
        return x.permute(1, 0, 2, 3, 4).contiguous()
    return x


# ---------------------------------------------------------------------
# Entrenamiento supervisado de una tarea
# - method: objeto con .penalty() (EWC) o None.
# - Guarda manifest.json con metadatos de ejecución.
# ---------------------------------------------------------------------
def train_supervised(
    model: nn.Module,
    train_loader,
    val_loader,
    loss_fn: nn.Module,
    cfg: TrainConfig,
    out_dir: Path,
    method=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibilidad (opcional)
    if cfg.seed is not None:
        set_seeds(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    # AMP moderna (torch.amp). Solo activa en CUDA.
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(cfg.amp and device_type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    history = {"train_loss": [], "val_loss": []}
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False):
            # Importante:
            # - Primero permutamos (B,T,...) -> (T,B,...) si hace falta
            # - Luego transferimos a GPU con non_blocking=True para solapar
            #   transferencias con cómputo (requiere pin_memory=True en DataLoader)
            x = _permute_if_needed(x).to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with autocast(device_type=device_type, enabled=use_amp):
                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                if method is not None:
                    # Penalización EWC u otros métodos de CL
                    loss = loss + method.penalty()

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

            # if use_amp:
            #     scaler.scale(loss).backward()
            #     scaler.step(opt)
            #     scaler.update()
            # else:
            #     loss.backward()
            #     opt.step()

            running += loss.item()

        train_loss = running / max(1, len(train_loader))

        # Validación (puedes omitirla si no la necesitas)
        model.eval()
        v_running = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = _permute_if_needed(x).to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with autocast(device_type=device_type, enabled=use_amp):
                    y_hat = model(x)
                    v_loss = loss_fn(y_hat, y)
                v_running += v_loss.item()

        val_loss = v_running / max(1, len(val_loader))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

    elapsed = time.time() - t0

    # Manifest con metadatos de ejecución (útil para trazabilidad)
    manifest = {
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "amp": cfg.amp,
        "seed": cfg.seed,
        "elapsed_sec": elapsed,
        "device": str(device),
        "history": history,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return history

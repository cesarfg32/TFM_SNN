# src/training.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn, optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from .utils import set_seeds  # reproducibilidad global

# --- Runtime encode (para loaders 4D) ---
_RUNTIME_ENC = {"mode": None, "T": None, "gain": None, "device": torch.device("cpu")}

def set_runtime_encode(mode: str | None, T: int | None = None, gain: float | None = None, device=None):
    """
    Activa/desactiva codificación temporal en runtime para inputs 4D (B,C,H,W).
    - mode: "rate" | "latency" | "raw" | None
    - T, gain: parámetros del encoder
    - device: torch.device destino (por defecto CUDA si disponible)
    """
    global _RUNTIME_ENC
    if mode is None:
        _RUNTIME_ENC.update({"mode": None, "T": None, "gain": None, "device": torch.device("cpu")})
        return
    _RUNTIME_ENC.update({
        "mode": str(mode),
        "T": int(T) if T is not None else None,
        "gain": float(gain) if gain is not None else None,
        "device": device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    })


def _encode_rate_gpu(x: torch.Tensor, T: int, gain: float) -> torch.Tensor:
    # x: (B,C,H,W) en [0,1]
    p = (x * gain).clamp_(0.0, 1.0)                       # (B,C,H,W)
    pT = p.unsqueeze(0).expand(T, *p.shape).contiguous()  # (T,B,C,H,W)
    rnd = torch.rand_like(pT)
    return (rnd < pT).float()

def _encode_latency_gpu(x: torch.Tensor, T: int) -> torch.Tensor:
    # x: (B,C,H,W) en [0,1]
    B, C, H, W = x.shape
    t_float = (1.0 - x.clamp(0.0, 1.0)) * (T - 1)     # (B,C,H,W)
    t_idx = t_float.floor().to(torch.long)            # (B,C,H,W)
    spikes = torch.zeros((T, B, C, H, W), dtype=torch.float32, device=x.device)
    mask = x > 0.0
    if mask.any():
        b, c, h, w = torch.where(mask)               # 1D idxs alineados
        t = t_idx[b, c, h, w]                        # (N,)
        spikes[t, b, c, h, w] = 1.0
    return spikes

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
    """
    - Si llega 5D como (B,T,C,H,W), permuta a (T,B,C,H,W).
    - Si llega 4D (B,C,H,W) y runtime encode está activo, codifica en el DEVICE indicado
      por set_runtime_encode(...) y devuelve (T,B,C,H,W) directamente en ese device.
    - Si no aplica, devuelve x tal cual.
    """
    # 5D -> (T,B,C,H,W)
    if x.ndim == 5 and x.shape[0] != x.shape[1]:
        return x.permute(1, 0, 2, 3, 4).contiguous()

    # 4D con runtime encode activo
    if x.ndim == 4 and _RUNTIME_ENC["mode"] is not None:
        mode  = _RUNTIME_ENC["mode"].lower()
        T     = int(_RUNTIME_ENC["T"])
        gain  = _RUNTIME_ENC["gain"]
        dev   = _RUNTIME_ENC["device"]

        # ¡Primero mover el 4D pequeño a GPU!
        x = x.to(dev, non_blocking=True)  # (B,C,H,W)
        B, C, H, W = x.shape

        if mode == "rate":
            # Bernoulli en GPU
            p = (x * float(gain)).clamp_(0.0, 1.0)                      # (B,C,H,W)
            pT = p.unsqueeze(0).expand(T, B, C, H, W)                   # (T,B,C,H,W)
            rnd = torch.rand((T, B, C, H, W), device=dev, dtype=p.dtype)
            out = (rnd < pT).to(x.dtype)                                # (T,B,C,H,W)

        elif mode == "latency":
            x1 = x.clamp(0.0, 1.0)
            t_float = (1.0 - x1) * (T - 1)                              # (B,C,H,W)
            t_idx = t_float.floor().to(torch.int64)
            out = torch.zeros((T, B, C, H, W), dtype=x.dtype, device=dev)
            mask = x1 > 0
            if mask.any():
                t_coords = t_idx[mask]
                b, c, h, w = torch.where(mask)
                out[t_coords, b, c, h, w] = 1.0

        elif mode == "raw":
            out = x.unsqueeze(0).expand(T, B, C, H, W).contiguous()     # (T,B,C,H,W)

        else:
            raise ValueError(f"Unsupported runtime encode mode: {mode}")

        return out  # ya está en (T,B,C,H,W) y en la GPU

    # Caso por defecto
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
    use_amp = bool(cfg.amp and torch.cuda.is_available())
    scaler = GradScaler(enabled=use_amp)

    history = {"train_loss": [], "val_loss": []}
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False):
            # A GPU y en el formato que espera el modelo
            x = _permute_if_needed(x).to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            # Forward + pérdida (EWC si procede)
            # Nota: en autocast indicamos 'cuda' explícitamente para PyTorch 2.x
            with autocast("cuda", enabled=use_amp):
                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                if method is not None:
                    loss = loss + method.penalty()

            # Backward + CLIP GRADIENTS + step
            if use_amp:
                scaler.scale(loss).backward()
                # Desescalar antes de hacer clipping
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

            running += loss.item()

        train_loss = running / max(1, len(train_loader))

        # Validación simple
        model.eval()
        v_running = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = _permute_if_needed(x).to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with autocast("cuda", enabled=use_amp):
                    y_hat = model(x)
                    v_loss = loss_fn(y_hat, y)
                v_running += v_loss.item()

        val_loss = v_running / max(1, len(val_loader))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

    elapsed = time.time() - t0

    # Manifest con metadatos de ejecución (útil para tablas de resultados)
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

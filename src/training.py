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

# ---------------------------------------------------------------------
# Runtime encode: permite que el DataLoader entregue 4D (B,C,H,W)
# y aquí hacemos la codificación temporal en GPU (T,B,C,H,W).
# ---------------------------------------------------------------------
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
        "device": device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")),
    })

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

        # mover a device antes de expandir
        x = x.to(dev, non_blocking=True)  # (B,C,H,W)
        B, C, H, W = x.shape

        if mode == "rate":
            p  = (x * float(gain)).clamp_(0.0, 1.0)                     # (B,C,H,W)
            pT = p.unsqueeze(0).expand(T, B, C, H, W)                   # (T,B,C,H,W)
            rnd = torch.rand((T, B, C, H, W), device=dev, dtype=p.dtype)
            out = (rnd < pT).to(x.dtype)

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
            out = x.unsqueeze(0).expand(T, B, C, H, W).contiguous()

        else:
            raise ValueError(f"Unsupported runtime encode mode: {mode}")

        return out  # (T,B,C,H,W) en dev

    # Caso por defecto
    return x

# ---------------------------------------------------------------------
# Configuración de entrenamiento
# ---------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 2
    batch_size: int = 8
    lr: float = 1e-3
    amp: bool = True
    seed: Optional[int] = None  # reproducibilidad opcional
    # Early Stopping (opcional; si no usas, deja None/False)
    es_patience: Optional[int] = None
    es_min_delta: float = 0.0  # mejora mínima en val_loss para resetear paciencia

# ---------------------------------------------------------------------
# Entrenamiento supervisado de una tarea (con EWC opcional y EarlyStopping opcional)
# ---------------------------------------------------------------------
def train_supervised(
    model: nn.Module,
    train_loader,
    val_loader,
    loss_fn: nn.Module,
    cfg: TrainConfig,
    out_dir: Path,
    method=None,  # e.g., EWC con .penalty()
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.seed is not None:
        set_seeds(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    use_amp = bool(cfg.amp and torch.cuda.is_available())
    scaler = GradScaler(enabled=use_amp)

    history = {"train_loss": [], "val_loss": []}
    t0 = time.time()

    best_val = float("inf")
    patience_left = cfg.es_patience if (cfg.es_patience is not None and cfg.es_patience > 0) else None
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False):
            x = _permute_if_needed(x).to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp):
                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                if method is not None:
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

            running += loss.item()

        train_loss = running / max(1, len(train_loader))

        # Validación
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

        # Early stopping (opcional)
        if patience_left is not None:
            improved = (best_val - val_loss) > cfg.es_min_delta
            if improved:
                best_val = val_loss
                patience_left = cfg.es_patience
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_left -= 1
                if patience_left <= 0:
                    # restaura mejor estado y corta
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    break

    elapsed = time.time() - t0

    # Manifest con metadatos
    manifest = {
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "amp": cfg.amp,
        "seed": cfg.seed,
        "elapsed_sec": elapsed,
        "device": str(device),
        "history": history,
        "early_stopping": {
            "used": bool(cfg.es_patience and cfg.es_patience > 0),
            "patience": cfg.es_patience,
            "min_delta": cfg.es_min_delta,
            "best_val": best_val if best_val != float("inf") else None,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return history

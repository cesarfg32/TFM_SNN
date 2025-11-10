# src/nn_io.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Optional
import torch
from torch import nn
from torch.amp import autocast

# --------------------------------------------------------------------------------------
# Runtime encode (para inputs 4D -> 5D) + helpers de forward y alineado de targets
# --------------------------------------------------------------------------------------

_RUNTIME_ENC: Dict[str, object] = {
    "mode": None,   # "rate" | "latency" | "raw" | None
    "T": None,
    "gain": None,
    "device": torch.device("cpu"),
}

def set_encode_runtime(mode: Optional[str], T: Optional[int] = None, gain: Optional[float] = None,
                       device: Optional[torch.device] = None) -> None:
    """Activa/desactiva codificación temporal en runtime para inputs 4D (B,C,H,W)."""
    if mode is None:
        _RUNTIME_ENC.update({"mode": None, "T": None, "gain": None, "device": torch.device("cpu")})
        return
    _RUNTIME_ENC.update({
        "mode": str(mode),
        "T": int(T) if T is not None else None,
        "gain": float(gain) if gain is not None else None,
        "device": device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")),
    })

@torch.no_grad()
def _maybe_runtime_encode(x4: torch.Tensor) -> Optional[torch.Tensor]:
    """Si activo y x es 4D (B,C,H,W), devuelve 5D (T,B,C,H,W); si no, None."""
    enc = _RUNTIME_ENC
    mode, T, gain = enc.get("mode"), enc.get("T"), enc.get("gain")
    if mode is None or T is None or x4.ndim != 4:
        return None

    dev = x4.device
    # Normaliza a [0,1]
    if x4.dtype == torch.uint8:
        x = x4.to(device=dev, dtype=torch.float32) / 255.0
    else:
        x = x4.to(device=dev, dtype=torch.float32)
        if x.max().item() > 1.5:
            x = (x / 255.0).clamp_(0.0, 1.0)
        else:
            x = x.clamp_(0.0, 1.0)

    if mode == "raw":
        return x.unsqueeze(0).repeat(T, 1, 1, 1, 1).contiguous()
    if mode == "rate":
        p = (x * float(gain if gain is not None else 1.0)).clamp_(0.0, 1.0)
        r = torch.rand((T, *x.shape), device=dev)
        return (r < p.unsqueeze(0)).to(torch.float32).contiguous()
    if mode == "latency":
        p = (x * float(gain if gain is not None else 1.0)).clamp_(0.0, 1.0)
        t_fire = torch.round((1.0 - p) * float(max(T - 1, 0))).to(torch.long)  # (B,C,H,W)
        N = t_fire.numel()
        spikes = torch.zeros((T, N), device=dev, dtype=torch.float32)
        idx = torch.arange(N, device=dev, dtype=torch.long)
        spikes[t_fire.view(-1), idx] = 1.0
        return spikes.view(T, *x.shape).contiguous()
    return None

def _align_target_shape(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Alinea y con y_hat: (B,)↔(B,1)."""
    if y_hat.ndim == 2 and y_hat.shape[1] == 1 and y.ndim == 1:
        return y.unsqueeze(1)
    if y_hat.ndim == 1 and y.ndim == 2 and y.shape[1] == 1:
        return y.squeeze(1)
    return y

# --------------------------------------------------------------------------------------
# AMP-safe forward: intenta autocast y si hay mismatch Half/Float en bias, reintenta FP32
# --------------------------------------------------------------------------------------

_WARNED_FALLBACK: Dict[str, bool] = {}  # por fase (train/val/fisher) para no spamear

def _forward_amp_safe(model: nn.Module, x_fwd: torch.Tensor, use_amp: bool, phase: str) -> torch.Tensor:
    """
    Intenta forward con AMP. Si PyTorch lanza el clásico error de:
      "Input type (c10::Half) and bias type (float) should be the same"
    reintentamos el forward en FP32 SOLO para este paso/batch y avisamos una vez por fase.
    """
    try:
        with autocast("cuda", enabled=use_amp):
            return model(x_fwd)
    except RuntimeError as e:
        msg = str(e)
        # Detecta patrón típico de mismatch de dtype en bias
        if ("bias type" in msg or "should be the same" in msg) and ("Half" in msg or "c10::Half" in msg):
            if not _WARNED_FALLBACK.get(phase, False):
                print(f"[AMP-FALLBACK:{phase}] Detectado mismatch de dtype (Half vs Float en bias). Reintentando en FP32 SOLO para este forward.")
                _WARNED_FALLBACK[phase] = True
            with autocast("cuda", enabled=False):
                return model(x_fwd)
        raise  # Si es otro error, lo propagamos

def _forward_with_cached_orientation(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device,
    use_amp: bool, phase_hint: Dict[str, str], phase: str,
) -> torch.Tensor:
    """Forward robusto para 4D/5D, con autocorrección de orientación y runtime-encode opcional."""
    # mover a device
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    # 4D -> 5D (runtime)
    if x.ndim == 4 and _RUNTIME_ENC.get("mode") is not None:
        x_rt = _maybe_runtime_encode(x)
        if x_rt is not None:
            x = x_rt  # (T,B,C,H,W)

    # no secuencial
    if x.ndim != 5:
        return _forward_amp_safe(model, x, use_amp=use_amp, phase=phase)

    B = int(y.shape[0])
    hint = phase_hint.get(phase)
    if hint is None:
        hint = "ok" if x.shape[1] == B else ("permute" if x.shape[0] == B else "ok")
        phase_hint[phase] = hint

    x_fwd = x if hint == "ok" else x.permute(1, 0, 2, 3, 4).contiguous()
    y_hat = _forward_amp_safe(model, x_fwd, use_amp=use_amp, phase=phase)

    # autocorrección si hace falta
    if isinstance(y_hat, torch.Tensor) and y_hat.ndim >= 1 and y_hat.shape[0] != B:
        if hint == "ok":
            x_alt = x.permute(1, 0, 2, 3, 4).contiguous()
            y_try = _forward_amp_safe(model, x_alt, use_amp=use_amp, phase=phase)
            if isinstance(y_try, torch.Tensor) and y_try.ndim >= 1 and y_try.shape[0] == B:
                phase_hint[phase] = "permute"
                y_hat = _forward_amp_safe(model, x_alt, use_amp=use_amp, phase=phase)
        else:
            y_try = _forward_amp_safe(model, x, use_amp=use_amp, phase=phase)
            if isinstance(y_try, torch.Tensor) and y_try.ndim >= 1 and y_try.shape[0] == B:
                phase_hint[phase] = "ok"
                y_hat = _forward_amp_safe(model, x, use_amp=use_amp, phase=phase)
    return y_hat

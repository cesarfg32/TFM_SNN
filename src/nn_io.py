# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, Optional, Union

import torch
from torch import nn

__all__ = [
    "set_encode_runtime",
    "get_encode_runtime",
    "_forward_with_cached_orientation",
    "_align_target_shape",
    # alias utilitarios expuestos
    "move_to_device",
    "to_float32",
]

# ============================================================
#  Estado global + configuración (retro-compat runner)
# ============================================================
# Antes era un bool; ahora guardamos una config flexible pero
# mantenemos verdad/falsedad al evaluar get_encode_runtime().
_ENCODE_RUNTIME_CFG: Dict[str, Any] = {
    "enabled": False,
    "mode": None,
    "T": None,
    "gain": None,
    "device": None,
}


def set_encode_runtime(
    enabled: Optional[bool] = None,
    *,
    mode: Optional[str] = None,
    T: Optional[int] = None,
    gain: Optional[float] = None,
    device: Optional[Union[torch.device, str]] = None,
    **_: Any,
) -> None:
    """
    Retro-compat con el runner:
      - set_encode_runtime(mode=encoder, T=T, gain=gain, device=device)
      - set_encode_runtime(None) -> OFF
      - set_encode_runtime(True/False)
    """
    global _ENCODE_RUNTIME_CFG

    if enabled is None and mode is None:
        # llamada tipo set_encode_runtime(None) -> OFF
        _ENCODE_RUNTIME_CFG.update(
            {"enabled": False, "mode": None, "T": None, "gain": None, "device": None}
        )
        return

    if enabled is None:
        enabled = mode is not None

    _ENCODE_RUNTIME_CFG.update(
        {
            "enabled": bool(enabled),
            "mode": mode,
            "T": T,
            "gain": gain,
            "device": str(device) if device is not None else None,
        }
    )


def get_encode_runtime(key: Optional[str] = None) -> Any:
    """
    - Sin argumentos: devuelve bool(enabled) para compat anterior.
    - Con 'key': devuelve el valor de esa clave en la config.
    """
    if key is None:
        return bool(_ENCODE_RUNTIME_CFG.get("enabled", False))
    return _ENCODE_RUNTIME_CFG.get(key, None)


# ============================================================
#  Utilidades internas / alias públicos
# ============================================================
def _to_device(obj: Any, device: torch.device, non_blocking: bool = True) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device=device, non_blocking=non_blocking)
    if isinstance(obj, (list, tuple)):
        typ = type(obj)
        return typ(_to_device(x, device, non_blocking) for x in obj)
    if isinstance(obj, dict):
        return {k: _to_device(v, device, non_blocking) for k, v in obj.items()}
    return obj


def move_to_device(obj: Any, device: Union[torch.device, str], non_blocking: bool = True) -> Any:
    """
    Alias público de _to_device: mueve tensores (o estructuras de tensores)
    a un device dado. Útil en otros módulos.
    """
    dev = torch.device(device)
    return _to_device(obj, dev, non_blocking=non_blocking)


def to_float32(obj: Any) -> Any:
    """
    Convierte tensores (o estructuras) a float32 de forma recursiva.
    No altera tensores no-flotantes. Útil para estabilizar métricas.
    """
    if torch.is_tensor(obj):
        return obj.float() if obj.dtype.is_floating_point else obj
    if isinstance(obj, (list, tuple)):
        typ = type(obj)
        return typ(to_float32(x) for x in obj)
    if isinstance(obj, dict):
        return {k: to_float32(v) for k, v in obj.items()}
    return obj


def _first_tensor(x: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(x):
        return x
    if isinstance(x, dict):
        # heurística: claves típicas primero
        for key in ("logits", "pred", "y_hat", "output", "out"):
            v = x.get(key, None)  # type: ignore[arg-type]
            if torch.is_tensor(v):
                return v
        for v in x.values():
            t = _first_tensor(v)
            if t is not None:
                return t
        return None
    if isinstance(x, (tuple, list)):
        for v in x:
            t = _first_tensor(v)
            if t is not None:
                return t
        return None
    return None


# -------- Permuta (B,T,...) -> (T,B,...) cuando aplique --------
def _maybe_BT_to_TB(x: torch.Tensor, B_hint: Optional[int]) -> torch.Tensor:
    """
    Normaliza a (T,B,...) SOLO cuando:
      - x.ndim >= 5  típicamente spikes (B,T,C,H,W) / (T,B,C,H,W)
      - x.ndim == 3  secuencias (B,T,F) / (T,B,F)
      - x.ndim == 2  borde (B,T) / (T,B)
    No toca tensores 4D de imagen (B,C,H,W).
    """
    if not torch.is_tensor(x) or B_hint is None:
        return x

    # 5D: (B,T,C,H,W) <-> (T,B,C,H,W)
    if x.ndim >= 5:
        if x.shape[0] == B_hint and x.shape[1] != B_hint:
            return x.permute(1, 0, *range(2, x.ndim)).contiguous()
        return x

    # 3D: (B,T,F) <-> (T,B,F)
    if x.ndim == 3:
        if x.shape[0] == B_hint and x.shape[1] != B_hint:
            return x.permute(1, 0, 2).contiguous()
        return x

    # 2D: (B,T) <-> (T,B)
    if x.ndim == 2:
        if x.shape[0] == B_hint and x.shape[1] != B_hint:
            return x.permute(1, 0).contiguous()
        return x

    # 4D (imágenes): NO permutar (B,C,H,W)
    return x


def _bt_to_tb_structure(x: Any, B_hint: Optional[int]) -> Any:
    if torch.is_tensor(x):
        return _maybe_BT_to_TB(x, B_hint)
    if isinstance(x, (list, tuple)):
        typ = type(x)
        return typ(_bt_to_tb_structure(v, B_hint) for v in x)
    if isinstance(x, dict):
        return {k: _bt_to_tb_structure(v, B_hint) for k, v in x.items()}
    return x


# ============================================================
#  Forward "seguro" con AMP + orientación temporal
# ============================================================
def _forward_with_cached_orientation(
    model: nn.Module,
    x: Any,
    y: Optional[torch.Tensor] = None,
    device: Union[torch.device, str, None] = None,
    use_amp: bool = False,
    phase_hint: Optional[Dict[str, Any]] = None,
    phase: Optional[str] = None,
    *args,
    **kwargs,
) -> torch.Tensor:
    """
    - Mueve x al device.
    - Si detecta (B,T,...) en spikes/secuencias, lo permuta a (T,B,...) (estilo snnTorch).
    - Ejecuta el forward con autocast opcional (sólo en CUDA).
    - Devuelve SIEMPRE un Tensor (primer tensor de la salida).
    """
    # Normaliza device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device(device)

    # Pista de B por 'y'
    B_hint = None
    if torch.is_tensor(y):
        try:
            B_hint = int(y.shape[0])
        except Exception:
            B_hint = None

    # Mueve x
    x_dev = _to_device(x, device)

    # Normaliza orientación temporal si aplica
    x_dev = _bt_to_tb_structure(x_dev, B_hint=B_hint)

    # AMP según device (sólo activo si CUDA)
    dev_type = "cuda" if device.type == "cuda" else "cpu"
    amp_enabled = bool(use_amp and device.type == "cuda")
    with torch.amp.autocast(device_type=dev_type, enabled=amp_enabled):
        out = model(x_dev)

    y_hat = _first_tensor(out)
    if y_hat is None:
        if torch.is_tensor(out):
            y_hat = out
        else:
            raise RuntimeError(
                "[nn_io] El modelo no devolvió ningún tensor reconocible. "
                f"Tipo de salida: {type(out).__name__}"
            )
    return y_hat


# ============================================================
#  Alineación de targets a la forma de y_hat
# ============================================================
def _align_target_shape(y_hat: Any, y: torch.Tensor) -> torch.Tensor:
    """
    Alinea 'y' con la forma de 'y_hat' en casos comunes de regresión de 1D:
    - (B,) <-> (B,1)
    Tolera y_hat como dict/tuple/list; extrae el primer tensor de referencia.
    """
    ref = _first_tensor(y_hat) if not torch.is_tensor(y_hat) else y_hat
    if ref is None or not torch.is_tensor(y):
        return y

    if y.dtype.is_floating_point:
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        if ref.ndim == 2 and ref.shape[1] == 1 and y.ndim == 1:
            return y.unsqueeze(1)
        if ref.ndim == 1 and y.ndim == 2 and y.shape[1] == 1:
            return y.squeeze(1)
    except Exception:
        return y
    return y

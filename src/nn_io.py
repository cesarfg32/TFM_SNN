# src/nn_io.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Union, Iterable

import torch
from torch import nn

__all__ = [
    "set_encode_runtime",
    "get_encode_runtime",
    "_forward_with_cached_orientation",
    "_align_target_shape",
]

# ============================================================
#  Estado global mínimo para compatibilidad
# ============================================================
_ENCODE_RUNTIME: bool = False  # toggled por training/runner según preset


def set_encode_runtime(enabled: bool) -> None:
    """
    Activa/desactiva el modo de codificación en runtime (shim para compatibilidad).
    """
    global _ENCODE_RUNTIME
    _ENCODE_RUNTIME = bool(enabled)


def get_encode_runtime() -> bool:
    """Devuelve el flag global de encode_runtime."""
    return _ENCODE_RUNTIME


# ============================================================
#  Utilidades internas
# ============================================================
def _to_device(obj: Any, device: torch.device, non_blocking: bool = True) -> Any:
    """
    Mueve tensores (o estructuras anidadas de tensores) a 'device'.
    Deja intactos los tipos no-tensor.
    """
    if torch.is_tensor(obj):
        return obj.to(device=device, non_blocking=non_blocking)
    if isinstance(obj, (list, tuple)):
        typ = type(obj)
        return typ(_to_device(x, device, non_blocking) for x in obj)
    if isinstance(obj, dict):
        return {k: _to_device(v, device, non_blocking) for k, v in obj.items()}
    return obj


def _first_tensor(x: Any) -> Optional[torch.Tensor]:
    """
    Extrae el primer Tensor encontrado de x (Tensor|tuple|list|dict), o None si no hay.
    """
    if torch.is_tensor(x):
        return x
    if isinstance(x, dict):
        # heurística: prueba keys más comunes, si no, el primer tensor en values
        for key in ("logits", "pred", "y_hat", "output", "out"):
            if key in x and torch.is_tensor(x[key]):
                return x[key]
        for v in x.values():
            if torch.is_tensor(v):
                return v
            t = _first_tensor(v)
            if t is not None:
                return t
        return None
    if isinstance(x, (tuple, list)):
        for v in x:
            if torch.is_tensor(v):
                return v
            t = _first_tensor(v)
            if t is not None:
                return t
        return None
    return None


# ============================================================
#  Forward "seguro" con AMP + extracción de tensor
# ============================================================
def _forward_with_cached_orientation(
    model: nn.Module,
    x: Any,
    y: Optional[torch.Tensor] = None,
    *,
    device: Union[torch.device, str, None] = None,
    use_amp: bool = False,
    phase_hint: Optional[Dict[str, Any]] = None,
    phase: Optional[str] = None,
) -> torch.Tensor:
    """
    Ejecuta el forward del modelo moviendo x al device, con autocast opcional,
    y devuelve SIEMPRE un Tensor (primer tensor de la salida), apto para losses.

    - Acepta salidas del modelo como Tensor | tuple | list | dict.
    - No asume orientación temporal concreta; no reordena ejes (la data/preset ya lo fija).
    - 'phase_hint' y 'phase' se aceptan por compatibilidad; no se usan aquí.

    Returns:
        torch.Tensor: primer tensor extraído de la salida del modelo.
    """
    # Normaliza device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device(device)

    # Mueve x a device (respetando estructuras)
    x_dev = _to_device(x, device)

    # AMP según device
    dev_type = "cuda" if device.type == "cuda" else "cpu"
    amp_enabled = bool(use_amp and device.type == "cuda")

    model_was_training = model.training
    # No tocamos el modo: que decida el caller (EWC ya pone eval en estimate_fisher)

    with torch.amp.autocast(device_type=dev_type, enabled=amp_enabled):
        out = model(x_dev)

    y_hat = _first_tensor(out)
    if y_hat is None:
        # Último intento: si 'out' es directamente un número/array convertible:
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
    - (B,)  <->  (B,1)

    También tolera que y_hat sea dict/tuple/list; en ese caso se extrae el primer tensor
    para chequear su forma. Si no se puede extraer, devuelve 'y' tal cual.

    Casos soportados:
      * y_hat=(B,1), y=(B,)  --> y.unsqueeze(1)
      * y_hat=(B,),  y=(B,1) --> y.squeeze(1)
      * otros                --> y
    """
    ref = _first_tensor(y_hat) if not torch.is_tensor(y_hat) else y_hat
    if ref is None or not torch.is_tensor(y):
        return y

    # Limpieza de NaNs/Inf por robustez (no cambia la forma)
    if y.dtype.is_floating_point:
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        if ref.ndim == 2 and ref.shape[1] == 1 and y.ndim == 1:
            return y.unsqueeze(1)
        if ref.ndim == 1 and y.ndim == 2 and y.shape[1] == 1:
            return y.squeeze(1)
    except Exception:
        # En caso de algo raro, devolvemos y sin tocar
        return y
    return y

# src/methods/sa_snn.py
# -*- coding: utf-8 -*-
"""SA-SNN (Sparse Selective Activation)
Gating esparso sobre una capa objetivo del modelo para reducir interferencia entre tareas.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

Tensor = torch.Tensor
DeviceLike = Union[torch.device, str]

@dataclass
class SASNNConfig:
    # Dónde enganchar (nombre exacto del submódulo en model.named_modules())
    attach_to: str
    # Grado de esparsidad: entero (Top-K) o fracción (0<k<1 => ratio)
    k: float = 8.0
    # Dinámica de traza: tr <- tr - tr/tau + S
    tau: float = 32.0
    # Umbral adaptativo v(t) ~ vt_scale * EMA(activity) clamped en [th_min, th_max].
    vt_scale: float = 1.33
    th_min: float = 1.0
    th_max: float = 2.0
    # Suavizado del EMA (si None, exp(-1/p))
    p: int = 2_000_000
    ema_beta: Optional[float] = None
    # Dominio del gating / tipo de activación
    assume_binary_spikes: bool = False
    flatten_spatial: bool = False
    # Gestión por tarea
    reset_counters_each_task: bool = False
    # ¿Actualizar también en eval()? por defecto no
    update_on_eval: bool = False

class SASNN:
    """Implementación de SA-SNN como wrapper con forward_hook."""
    name = "sa-snn"

    def __init__(self, model: nn.Module, cfg: SASNNConfig, device: Optional[DeviceLike] = None):
        self.name = "sa-snn"
        self.model = model
        self.cfg = cfg
        self.device = torch.device(device) if device is not None else next(model.parameters()).device

        target = dict(model.named_modules()).get(cfg.attach_to, None)
        if target is None:
            raise ValueError(f"[SA-SNN] No se encontró submódulo '{cfg.attach_to}'.")
        self._target: nn.Module = target
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

        # Estado dinámico (lazy)
        self._neurons_N: Optional[int] = None
        self._tr: Optional[Tensor] = None
        self._ema: Optional[Tensor] = None
        self._counters: Optional[Tensor] = None
        self._t_seen: int = 0
        self._vt_bias: Optional[Tensor] = None

        # EMA beta
        if self.cfg.ema_beta is not None:
            self._ema_beta = float(self.cfg.ema_beta)
        else:
            self._ema_beta = float(torch.exp(torch.tensor(-1.0 / max(1, self.cfg.p))).item())

        # Rango y tau
        self._tau = float(max(1.0, self.cfg.tau))
        self._th_min = float(self.cfg.th_min)
        self._th_max = float(max(self.cfg.th_min, self.cfg.th_max))

        # Engancha hook
        self.attach()

    def __repr__(self) -> str:
        return (
            f"SASNN(name={self.name}, attach_to={self.cfg.attach_to}, "
            f"k={self.cfg.k}, tau={self.cfg.tau}, vt_scale={self.cfg.vt_scale}, "
            f"th=[{self.cfg.th_min},{self.cfg.th_max}], "
            f"flatten_spatial={self.cfg.flatten_spatial}, "
            f"assume_binary_spikes={self.cfg.assume_binary_spikes})"
        )

    def attach(self) -> None:
        if self._hook_handle is not None:
            return
        self._hook_handle = self._target.register_forward_hook(self._forward_hook)
        self.to(self.device)

    def detach(self) -> None:
        try:
            if self._hook_handle is not None:
                self._hook_handle.remove()
        finally:
            self._hook_handle = None
            self._target = None  # type: ignore
            self._tr = None
            self._ema = None
            self._counters = None
            self._vt_bias = None
            self._neurons_N = None

    def to(self, device: DeviceLike) -> "SASNN":
        self.device = torch.device(device)
        for name in ("_tr", "_ema", "_counters", "_vt_bias"):
            buf = getattr(self, name)
            if isinstance(buf, torch.Tensor):
                setattr(self, name, buf.to(self.device, non_blocking=True))
        return self

    # API esperada por el runner
    def before_task(self, model: nn.Module, train_loader, val_loader) -> None:
        if self.cfg.reset_counters_each_task and self._counters is not None:
            self._counters.zero_()

    def after_task(self, model: nn.Module, train_loader, val_loader) -> None:
        pass

    def before_epoch(self, model: nn.Module, epoch: int) -> None: ...
    def after_epoch(self, model: nn.Module, epoch: int) -> None: ...
    def before_batch(self, model: nn.Module, batch) -> None: ...
    def after_batch(self, model: nn.Module, batch, loss) -> None: ...

    def penalty(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32, device=self.device)

    # Hook principal
    @torch.no_grad()
    def _forward_hook(self, module: nn.Module, inputs, out: Tensor) -> Tensor:
        out = out if isinstance(out, torch.Tensor) else _pick_first_tensor(out)
        if out is None:
            return out
        out = out.to(self.device, non_blocking=True)

        if out.dim() < 2:
            out = out.view(out.shape[0], 1)

        B, C = out.shape[:2]
        spatial = out.shape[2:]
        if self.cfg.flatten_spatial:
            HW = int(torch.prod(torch.tensor(spatial)).item()) if spatial else 1
            N = max(1, C * HW)
        else:
            N = max(1, C)

        if self._neurons_N is None:
            self._neurons_N = N
            self._tr = torch.zeros(N, device=self.device)
            self._ema = torch.zeros(N, device=self.device)
            self._counters = torch.zeros(N, device=self.device)
            self._vt_bias = torch.zeros(N, device=self.device)
            self._t_seen = 0

        S = _reduce_activity(out, flatten_spatial=self.cfg.flatten_spatial)  # (N,)
        if self.cfg.assume_binary_spikes:
            S = (S > 0).to(out.dtype)

        is_train_mode = module.training or self.cfg.update_on_eval
        if is_train_mode:
            tr_next = self._tr - (self._tr / self._tau) + S
            tr_next = torch.nan_to_num(tr_next, nan=0.0, posinf=0.0, neginf=0.0)
            self._tr = tr_next

            self._ema.mul_(self._ema_beta).add_(S, alpha=(1.0 - self._ema_beta))
            self._ema[:] = torch.nan_to_num(self._ema, nan=0.0, posinf=0.0, neginf=0.0)

            self._counters.add_(S)
            self._t_seen += 1

        vt_bias = _compute_vt_bias(self._ema, self.cfg.vt_scale, self._th_min, self._th_max)
        self._vt_bias = vt_bias

        score = self._tr - vt_bias
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        kk = _resolve_k(self.cfg.k, N)
        if kk >= N:
            sel_idx = torch.arange(N, device=self.device)
        else:
            sel_idx = torch.topk(score, k=kk, dim=0, largest=True, sorted=False).indices

        mask_neurons = torch.zeros(N, dtype=out.dtype, device=self.device)
        mask_neurons[sel_idx] = 1.0
        masked = _apply_mask(out, mask_neurons, flatten_spatial=self.cfg.flatten_spatial)
        return masked


# ----- Utilidades locales (sin dependencias externas) -----

def _pick_first_tensor(x) -> Optional[torch.Tensor]:
    if isinstance(x, torch.Tensor): return x
    if isinstance(x, (tuple, list)) and x: return x[0] if isinstance(x[0], torch.Tensor) else None
    return None

def _reduce_activity(out: Tensor, *, flatten_spatial: bool) -> Tensor:
    if not flatten_spatial:
        dims = (0,) + tuple(range(2, out.dim()))
        return out.mean(dim=dims)
    B = out.shape[0]
    return out.view(B, -1).mean(dim=0)

def _compute_vt_bias(ema: Tensor, vt_scale: float, th_min: float, th_max: float) -> Tensor:
    vt = vt_scale * ema
    return torch.clamp(vt, min=th_min, max=th_max)

def _resolve_k(k: float, N: int) -> int:
    if k < 1.0:
        kk = max(1, int(round(float(k) * N)))
    else:
        kk = int(round(k))
    return min(max(1, kk), N)

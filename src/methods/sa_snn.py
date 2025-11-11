# src/methods/sa_snn.py
# -*- coding: utf-8 -*-
"""SA-SNN (Sparse Selective Activation)
--------------------------------------
Gating esparso sobre una capa objetivo del modelo para reducir interferencia entre tareas.
- Hook de forward que calcula una traza temporal por neurona y aplica Top-K según score.
- Umbral adaptativo v(t) ~ vt_scale * EMA(activity) clamped en [th_min, th_max].
- k puede ser entero (# neuronas) o fracción (0<k<1 => ratio).
- Opciones: operar por canales o a nivel espacial completo (flatten_spatial).

Interfaz esperada por el runner:
  - .name (str)
  - .before_task(model, train_loader, val_loader)
  - .after_task(model, train_loader, val_loader)
  - .before_epoch(model, epoch) / .after_epoch(model, epoch): opcionales
  - .before_batch(model, batch) / .after_batch(model, batch, loss): opcionales
  - .penalty() -> Tensor escalar (aquí 0.0 en device correcto)
  - .close(), .state_dict(), .load_state_dict()
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, List

import torch
import torch.nn as nn

Tensor = torch.Tensor
DeviceLike = Union[torch.device, str]

__all__ = ["SASNN", "SASNNConfig"]


@dataclass
class SASNNConfig:
    # Si None => auto: última Linear; si no hay, última Conv2d
    attach_to: Optional[str] = None
    # Grado de esparsidad: entero (Top-K) o fracción (0<k<1 => ratio)
    k: float = 8.0
    # Dinámica de la traza (en pasos temporales T): tr <- tr - tr/tau + S
    tau: float = 32.0
    # Escalado del umbral adaptativo (clamp a [th_min, th_max])
    vt_scale: float = 1.33
    th_min: float = 1.0
    th_max: float = 2.0
    # Suavizado del EMA (si None, se calcula como exp(-1/p))
    p: int = 2_000_000
    ema_beta: Optional[float] = None
    # Dominio del gating / tipo de activación
    assume_binary_spikes: bool = False
    flatten_spatial: bool = False
    # Gestión por tarea
    reset_counters_each_task: bool = False
    # ¿Actualizar también en eval()? por defecto no (para no contaminar)
    update_on_eval: bool = False


# ---------- utilidades ----------
def _find_target_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    if not name:
        return None
    lookup: Dict[str, nn.Module] = dict(model.named_modules())
    return lookup.get(name, None)


def _auto_select_target(model: nn.Module) -> Optional[nn.Module]:
    last_linear: Optional[nn.Module] = None
    last_conv: Optional[nn.Module] = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
        elif isinstance(m, nn.Conv2d):
            last_conv = m
    return last_linear or last_conv


# ------------------- implementación -------------------
class SASNN:
    name = "sa-snn"

    def __init__(self, model: nn.Module, cfg: SASNNConfig, device: Optional[DeviceLike] = None):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(device) if device is not None else next(model.parameters()).device

        # Selección de capa objetivo
        if self.cfg.attach_to is not None:
            target = _find_target_by_name(model, self.cfg.attach_to)
            attach_tag = self.cfg.attach_to
        else:
            target = _auto_select_target(model)
            attach_tag = "auto"

        if target is None:
            raise ValueError(
                "[SA-SNN] No se encontró capa objetivo. "
                "Define 'attach_to' con el path de un submódulo (p.ej. 'f6'), "
                "o asegúrate de que el modelo tenga al menos una nn.Linear/nn.Conv2d."
            )

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

        # Seguridad en rangos
        self._tau = float(max(1.0, self.cfg.tau))
        self._th_min = float(self.cfg.th_min)
        self._th_max = float(max(self.cfg.th_min, self.cfg.th_max))

        # Nombre legible
        self.name = (
            f"sa-snn"
            f"_k{self.cfg.k:g}"
            f"_tau{self.cfg.tau:g}"
            f"_vt{self.cfg.vt_scale:g}"
            f"_th[{self.cfg.th_min:g},{self.cfg.th_max:g}]"
            f"_{attach_tag}"
        )

        # Engancha hook
        self.attach()

    # ---------- gestión de target / hook ----------
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

    # ---------- API esperada por el runner ----------
    def before_task(self, model: nn.Module, train_loader, val_loader) -> None:
        if self.cfg.reset_counters_each_task and self._counters is not None:
            self._counters.zero_()

    def after_task(self, model: nn.Module, train_loader, val_loader) -> None:
        pass

    def before_epoch(self, model: nn.Module, epoch: int) -> None:
        pass

    def after_epoch(self, model: nn.Module, epoch: int) -> None:
        pass

    def before_batch(self, model: nn.Module, batch) -> None:
        pass

    def after_batch(self, model: nn.Module, batch, loss) -> None:
        pass

    def penalty(self) -> torch.Tensor:
        # SA-SNN no añade penalización al loss base, pero devolvemos Tensor en el device correcto
        return torch.zeros((), dtype=torch.float32, device=self.device)

    def close(self) -> None:
        self.detach()

    # ---------- Serialización ligera ----------
    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "cfg": self.cfg.__dict__,
            "t_seen": self._t_seen,
            "neurons_N": self._neurons_N,
            "ema_mean": float(self._ema.mean().item()) if isinstance(self._ema, torch.Tensor) else 0.0,
        }

    def load_state_dict(self, state: dict) -> None:
        self._t_seen = int(state.get("t_seen", 0))

    # ---------- Hook principal (¡SIN @torch.no_grad!) ----------
    def _forward_hook(self, module: nn.Module, inputs: Tuple[Tensor, ...], out: Tensor) -> Tensor:
        y = out
        if not torch.is_tensor(y):
            return out

        y = y.to(self.device, non_blocking=True)
        # Asegura al menos (B, C, ...)
        if y.dim() < 2:
            B = y.shape[0]
            y = y.view(B, 1)

        B, C = y.shape[:2]
        spatial = y.shape[2:]
        if self.cfg.flatten_spatial:
            hw = int(torch.prod(torch.tensor(spatial)).item()) if spatial else 1
            N = max(1, C * hw)
        else:
            N = max(1, C)

        # Init perezosa
        if self._neurons_N is None:
            self._init_buffers(N)

        # 1) Actividad S por "neurona"
        # (esto sí forma parte del grafo, pero la máscara no necesita grad)
        S = self._reduce_activity(y, flatten_spatial=self.cfg.flatten_spatial)  # (N,)
        if self.cfg.assume_binary_spikes:
            S = (S > 0).to(y.dtype)

        # 2) Actualiza traza/EMA/contadores (sin grad)
        is_train_mode = module.training or self.cfg.update_on_eval
        if is_train_mode:
            with torch.no_grad():
                tr_next = self._tr - (self._tr / self._tau) + S.detach()
                tr_next = torch.nan_to_num(tr_next, nan=0.0, posinf=0.0, neginf=0.0)
                self._tr = tr_next

                self._ema.mul_(self._ema_beta).add_(S.detach(), alpha=(1.0 - self._ema_beta))
                self._ema[:] = torch.nan_to_num(self._ema, nan=0.0, posinf=0.0, neginf=0.0)

                self._counters.add_(S.detach())
                self._t_seen += 1

        # 3) Umbral adaptativo y máscara (sin grad)
        with torch.no_grad():
            vt_bias = self._compute_vt_bias()
            score = self._tr - vt_bias
            score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
            kk = self._resolve_k(self.cfg.k, N)
            if kk >= N:
                sel_idx = torch.arange(N, device=self.device)
            else:
                sel_idx = torch.topk(score, k=kk, dim=0, largest=True, sorted=False).indices

            mask_neurons = torch.zeros(N, dtype=y.dtype, device=self.device)
            mask_neurons[sel_idx] = 1.0

        # 4) Aplica máscara **con gradiente** (multiplicación fuera de no_grad)
        masked = self._apply_mask(y, mask_neurons, flatten_spatial=self.cfg.flatten_spatial)
        return masked

    # ---------- Utilidades internas ----------
    def _init_buffers(self, N: int) -> None:
        self._neurons_N = N
        self._tr = torch.zeros(N, device=self.device)
        self._ema = torch.zeros(N, device=self.device)
        self._counters = torch.zeros(N, device=self.device)
        self._vt_bias = torch.zeros(N, device=self.device)
        self._t_seen = 0

    @staticmethod
    def _resolve_k(k: float, N: int) -> int:
        if k < 1.0:
            kk = max(1, int(round(float(k) * N)))
        else:
            kk = int(round(k))
        return min(max(1, kk), N)

    def _reduce_activity(self, out: Tensor, flatten_spatial: bool) -> Tensor:
        if not flatten_spatial:
            # (B, C, H, W?) -> media en (B,H,W) => (C,)
            dims = (0,) + tuple(range(2, out.dim()))
            return out.mean(dim=dims)
        # flatten_spatial=True -> (B, C*H*W) -> mean en B
        B = out.shape[0]
        flat = out.view(B, -1)
        return flat.mean(dim=0)

    def _compute_vt_bias(self) -> Tensor:
        vt = self.cfg.vt_scale * self._ema
        vt = torch.clamp(vt, min=self._th_min, max=self._th_max)
        self._vt_bias = vt
        return vt

    def _apply_mask(self, out: Tensor, mask_neurons: Tensor, flatten_spatial: bool) -> Tensor:
        if not flatten_spatial:
            # máscara por canal: (C,) -> (1,C,1,1,...)
            shape = [1, -1] + [1] * (out.dim() - 2)
            m = mask_neurons.view(*shape)
            return out * m
        # máscara a nivel espacial
        B, C = out.shape[:2]
        spatial = out.shape[2:]
        HW = int(torch.prod(torch.tensor(spatial)).item()) if spatial else 1
        N = C * HW
        if N != mask_neurons.numel():
            return out  # fallback seguro
        if HW == 1:
            m = mask_neurons.view(1, C)
            return out * m
        m = mask_neurons.view(1, C, *spatial)
        return out * m

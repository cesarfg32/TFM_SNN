# src/methods/as_snn.py
# -*- coding: utf-8 -*-
"""
AS-SNN (Activity Sparsity + Synaptic Scaling) — versión con penalización por-capa
y opcional synaptic scaling tras cada tarea, sin modificar la arquitectura.

Qué hace:
- Registra:
  (1) un forward_pre_hook en el modelo para reiniciar penalizaciones por batch
      y (opcional) medir actividad de la ENTRADA (modo "input").
  (2) forward_hooks en un conjunto de módulos (capas) objetivo para medir la
      actividad de SUS SALIDAS (modo "modules"), sumar una penalización con
      gradiente y llevar un EMA por capa (para logging y scaling).
- `penalty()` devuelve la suma de penalizaciones del *último* batch (con grafo).
- `after_task()` puede aplicar synaptic scaling por capa: w <- s * w,
  donde s = clip(gamma / alpha_ema, [scale_clip_min, scale_clip_max]).

Compatibilidad:
- Interfaz compatible con api.ContinualMethod / registry.build_method(...).
- No requiere cambios en runner ni en el modelo.
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader


def _resolve_modules_by_name(model: nn.Module, name_substr: str) -> List[tuple[str, nn.Module]]:
    """Devuelve [(full_name, module), ...] cuyos nombres contienen `name_substr` (case-insensitive)."""
    out = []
    low = name_substr.lower()
    for n, m in model.named_modules():
        if low in n.lower():
            out.append((n, m))
    return out


class AS_SNN:
    """
    Regularización de actividad con gradiente (por-capa) + synaptic scaling opcional.

    Args:
        lambda_a           : Peso de la penalización (>= 0).
        gamma_ratio        : Objetivo de actividad en [0,1].
        ema                : Factor EMA para actividad (0,1).
        penalty_mode       : "l1" (|a-γ|) o "l2" ((a-γ)^2). Default "l1".
        measure_at         : "modules" (salidas de capas) | "input" (entrada al modelo) | "both".
                             Si se pasa attach_to -> por defecto "modules"; si no -> "input".
        attach_to          : (opcional) subcadena para seleccionar capas por nombre (p.ej. "f6").
                             Si None y measure_at incluye "modules", se usan heurísticas: Conv2d/Linear.
        do_synaptic_scaling: Si True, aplica escalado w <- s*w al final de cada tarea (NO por defecto).
        scale_clip         : Tuple(min,max) para recortar s (evitar explosiones). Default (0.5, 2.0).
        scale_bias         : Si True, escala también bias. Default False.
        eps                : Pequeño valor para divisiones seguras.
        name_suffix        : Sufijo solo para “naming” en outputs.
    """

    name = "as-snn"

    def __init__(
        self,
        lambda_a: float = 2.5,
        gamma_ratio: float = 0.5,
        ema: float = 0.9,
        penalty_mode: str = "l1",
        measure_at: Optional[str] = None,
        attach_to: Optional[str] = None,
        do_synaptic_scaling: bool = False,
        scale_clip: Tuple[float, float] = (0.5, 2.0),
        scale_bias: bool = False,
        eps: float = 1e-6,
        name_suffix: str = "",
        **kw,  # ignorar kwargs desconocidos
    ) -> None:
        assert 0.0 <= gamma_ratio <= 1.0, "gamma_ratio debe estar en [0,1]"
        assert 0.0 < ema < 1.0, "ema debe estar en (0,1)"
        assert lambda_a >= 0.0, "lambda_a debe ser >= 0"
        assert penalty_mode in ("l1", "l2"), "penalty_mode debe ser 'l1' o 'l2'"
        assert scale_clip[0] > 0 and scale_clip[0] <= scale_clip[1], "scale_clip inválido"

        self.lambda_a: float = float(lambda_a)
        self.gamma_ratio: float = float(gamma_ratio)
        self.ema: float = float(ema)
        self.penalty_mode: str = penalty_mode
        self.attach_to: Optional[str] = attach_to
        self.do_synaptic_scaling: bool = bool(do_synaptic_scaling)
        self.scale_clip: Tuple[float, float] = (float(scale_clip[0]), float(scale_clip[1]))
        self.scale_bias: bool = bool(scale_bias)
        self.eps: float = float(eps)

        # Política por defecto de dónde medir
        if measure_at is None:
            self.measure_at: str = "modules" if (attach_to is not None) else "input"
        else:
            assert measure_at in ("modules", "input", "both")
            self.measure_at = measure_at

        # Nombre legible
        tag = "as-snn"
        tag += f"_gr_{self.gamma_ratio:g}"
        tag += f"_lam_{self.lambda_a:g}"
        if self.attach_to:
            tag += f"_att_{self.attach_to}"
        if self.do_synaptic_scaling:
            tag += "_scale_on"
        if name_suffix:
            tag += f"_{name_suffix}"
        self.name = tag

        # ---- Estado interno ----
        self._device: Optional[torch.device] = None

        # Penalizaciones del batch actual (con grafo)
        self._batch_penalties: List[torch.Tensor] = []

        # EMA global de la entrada (si se usa "input" o "both")
        self._alpha_in_ema: Optional[torch.Tensor] = None
        self._alpha_in_last: Optional[torch.Tensor] = None

        # Stats por capa medida (si se usa "modules" o "both")
        # Estructura: name -> {"alpha_ema": Tensor, "alpha_last": Tensor, "module": nn.Module}
        self._layer_stats: Dict[str, Dict[str, torch.Tensor | nn.Module]] = {}

        # Handles de hooks para limpiar al final
        self._pre_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._fw_handles: List[torch.utils.hooks.RemovableHandle] = []

        # Logging inyectable desde preset
        self.activity_verbose: bool = False
        self.activity_every: int = 100
        self._batch_idx: int = 0

    # -------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------
    @staticmethod
    def _as_float_unit(x: torch.Tensor) -> torch.Tensor:
        """Convierte a float y recorta a [0,1] por robustez."""
        if not x.dtype.is_floating_point:
            x = x.float()
        return torch.clamp(x, 0.0, 1.0)

    def _maybe_on_device(self, t: torch.Tensor) -> None:
        if self._device is None:
            self._device = t.device

    def _ensure_layer_entry(self, name: str, device: torch.device, init_val: torch.Tensor) -> None:
        if name not in self._layer_stats:
            self._layer_stats[name] = {
                "alpha_ema": init_val.detach().clone().to(device),
                "alpha_last": init_val.detach().clone().to(device),
                "module": None,  # se rellena al registrar hooks
            }
        else:
            # mover a device en caso de cambio
            for k in ("alpha_ema", "alpha_last"):
                self._layer_stats[name][k] = self._layer_stats[name][k].to(device)  # type: ignore[assignment]

    def _phi(self, a: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        if self.penalty_mode == "l2":
            return (a - gamma).pow(2)
        return torch.abs(a - gamma)

    # -------------------------------------------------------------
    # Hooks
    # -------------------------------------------------------------
    def _pre_forward_hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
        """Se ejecuta al inicio de cada forward: resetea penalizaciones y (opcional) mide entrada."""
        self._batch_penalties.clear()
        if not inputs:
            return
        x = inputs[0]
        if not isinstance(x, torch.Tensor):
            return

        self._maybe_on_device(x)
        device = x.device

        if self.measure_at in ("input", "both"):
            a_now = self._as_float_unit(x).mean()
            if self._alpha_in_ema is None:
                self._alpha_in_ema = a_now.detach().clone().to(device)
                self._alpha_in_last = a_now.detach().clone().to(device)
            else:
                self._alpha_in_ema.mul_(self.ema).add_(a_now.detach(), alpha=(1.0 - self.ema))  # EMA sin grafo
                self._alpha_in_last.copy_(a_now.detach())

            # Penalización sobre la entrada (CON grafo si se quiere contribuir)
            gamma = torch.tensor(self.gamma_ratio, device=device, dtype=a_now.dtype)
            self._batch_penalties.append(self.lambda_a * self._phi(a_now, gamma))

        # logging ligero
        self._batch_idx += 1
        if self.activity_verbose and (self._batch_idx % max(1, int(self.activity_every))) == 0:
            try:
                a_in_ema = None if self._alpha_in_ema is None else float(self._alpha_in_ema.item())
                print(f"[AS-SNN] input_ema={a_in_ema} γ={self.gamma_ratio:.2f} λ={self.lambda_a:g}")
            except Exception:
                pass

    def _make_forward_hook(self, name: str):
        def _hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output):
            if not isinstance(output, torch.Tensor):
                return
            self._maybe_on_device(output)
            device = output.device

            # Medición por-capa sobre la SALIDA del módulo (CON grafo para penalización)
            a_now = self._as_float_unit(output).mean()  # escalar, con grafo
            self._ensure_layer_entry(name, device, a_now)

            # EMA solo para logging / scaling (DETACH)
            self._layer_stats[name]["alpha_ema"].mul_(self.ema).add_(a_now.detach(), alpha=(1.0 - self.ema))  # type: ignore[index]
            self._layer_stats[name]["alpha_last"].copy_(a_now.detach())  # type: ignore[index]

            gamma = torch.tensor(self.gamma_ratio, device=device, dtype=a_now.dtype)
            self._batch_penalties.append(self.lambda_a * self._phi(a_now, gamma))
        return _hook

    # -------------------------------------------------------------
    # API ContinualMethod
    # -------------------------------------------------------------
    def before_task(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Registra hooks según la configuración."""
        # pre-forward para reset y (opcional) medir entrada
        if self._pre_hook_handle is None:
            self._pre_hook_handle = model.register_forward_pre_hook(self._pre_forward_hook, with_kwargs=False)

        # forward_hooks por-capa si se pidió "modules" o "both"
        if self.measure_at in ("modules", "both") and not self._fw_handles:
            modules_to_hook: List[tuple[str, nn.Module]] = []
            if self.attach_to:
                cand = _resolve_modules_by_name(model, self.attach_to)
                modules_to_hook.extend(cand)
            else:
                # Heurística liviana: conv/linear suelen preceder a neuronas espiking; medimos su salida.
                for n, m in model.named_modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        modules_to_hook.append((n, m))

            # Registrar hooks
            registered = set()
            for name, mod in modules_to_hook:
                if mod in registered:
                    continue
                h = mod.register_forward_hook(self._make_forward_hook(name), with_kwargs=False)
                self._fw_handles.append(h)
                # Guarda referencia del módulo para scaling
                self._ensure_layer_entry(name, next(mod.parameters(), torch.tensor(0., device=self._device or torch.device('cpu'))).device, torch.tensor(0.0, device=self._device or torch.device('cpu')))
                self._layer_stats[name]["module"] = mod  # type: ignore[index]
                registered.add(mod)

    def penalty(self) -> torch.Tensor:
        """Suma de penalizaciones del *último* batch (con grafo)."""
        if len(self._batch_penalties) == 0:
            # crear 0 en el device conocido (o CPU si aún no hay)
            dev = self._device if self._device is not None else torch.device("cpu")
            return torch.zeros((), dtype=torch.float32, device=dev)
        return torch.stack(self._batch_penalties).sum()

    @torch.no_grad()
    def after_task(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Opcional: synaptic scaling por-capa al finalizar la tarea."""
        if not self.do_synaptic_scaling:
            return
        for name, stats in self._layer_stats.items():
            alpha_ema = stats.get("alpha_ema", None)
            mod = stats.get("module", None)
            if alpha_ema is None or mod is None or not hasattr(mod, "weight"):
                continue
            a = float(alpha_ema.item())
            if a <= 0.0:
                continue
            s_nominal = self.gamma_ratio / max(a, self.eps)
            s = float(max(self.scale_clip[0], min(self.scale_clip[1], s_nominal)))

            try:
                mod.weight.mul_(s)  # type: ignore[attr-defined]
                if self.scale_bias and getattr(mod, "bias", None) is not None:
                    mod.bias.mul_(s)   # type: ignore[attr-defined]
            except Exception:
                # Si algún módulo no soporta scaling directo, lo omitimos.
                pass

    def detach(self) -> None:
        if self._pre_hook_handle is not None:
            try: self._pre_hook_handle.remove()
            except Exception: pass
            self._pre_hook_handle = None

        if self._fw_handles:
            for h in self._fw_handles:
                try: h.remove()
                except Exception: pass
            self._fw_handles.clear()

    # -------------------------------------------------------------
    # Introspección (para logging)
    # -------------------------------------------------------------
    def get_activity_state(self) -> dict:
        out = {
            "gamma_ratio": self.gamma_ratio,
            "lambda_a": self.lambda_a,
            "ema": self.ema,
            "penalty_mode": self.penalty_mode,
            "measure_at": self.measure_at,
            "attach_to": self.attach_to,
            "do_synaptic_scaling": self.do_synaptic_scaling,
        }
        # Entrada global
        out["input_alpha_ema"] = (None if self._alpha_in_ema is None else float(self._alpha_in_ema.item()))
        out["input_alpha_last"] = (None if self._alpha_in_last is None else float(self._alpha_in_last.item()))
        # Capas
        layers = {}
        for name, st in self._layer_stats.items():
            layers[name] = {
                "alpha_ema": float(st["alpha_ema"].item()) if isinstance(st.get("alpha_ema"), torch.Tensor) else None,  # type: ignore[index]
                "alpha_last": float(st["alpha_last"].item()) if isinstance(st.get("alpha_last"), torch.Tensor) else None,  # type: ignore[index]
            }
        out["layers"] = layers
        return out

    def __repr__(self) -> str:
        return (f"AS_SNN(name={self.name!r}, lambda_a={self.lambda_a}, "
                f"gamma_ratio={self.gamma_ratio}, ema={self.ema}, "
                f"penalty_mode={self.penalty_mode}, measure_at={self.measure_at}, "
                f"attach_to={self.attach_to}, do_synaptic_scaling={self.do_synaptic_scaling})")

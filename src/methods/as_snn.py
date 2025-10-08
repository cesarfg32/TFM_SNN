# src/methods/as_snn.py
# -*- coding: utf-8 -*-
"""
AS-SNN (Activity Sparsity + Synaptic Scaling) — versión sencilla “piecewise”.

Qué hace esta implementación:
- Mide una “actividad media” α por batch justo antes del forward del modelo.
- Suaviza esa medida con un EMA (alpha_ema).
- Define un objetivo (gamma) como una fracción fija (gamma_ratio) y penaliza
  la desviación |alpha_ema - gamma| con un peso lambda_a.
- Devuelve esa penalización en penalty() para que el loop de entrenamiento la
  sume a la loss supervisada (sin romper autograd).
- (Opcional / futuro) Tras cada tarea, podrías aplicar un escalado sináptico
  por capa hacia la tasa objetivo. Aquí lo dejamos como no-op para no invadir
  tu arquitectura, pero el hueco está indicado.

Importante:
- El cálculo de la actividad se hace en un forward_pre_hook del modelo y
  la penalización se guarda como un escalar (tensor) en el device correcto.
- La penalización se calcula a partir de tensores DETACHED (sin grad) para
  evitar construir grafos adicionales que disparen errores de backward.
- Los buffers internos se mueven automáticamente al device del primer batch
  para evitar mismatches CUDA/CPU.

Limitaciones (consciente y explícita):
- La “actividad” se mide sobre la entrada al modelo (spikes o intensidades).
  Esto NO propaga gradientes hacia los pesos por sí mismo; la regularización
  sumada a la loss actúa como término constante. Para que influya en los
  pesos, la versión completa debería medir actividad en salidas de capas
  espiking internas (y opcionalmente escalar pesos al final de cada tarea).
  Aun así, esta versión te permite instrumentar, registrar y comparar con
  otros métodos (EWC, rehearsal, etc.) sin tocar tu modelo.

Interfaz compatible con src/methods/api.ContinualMethod y registry.py.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


class AS_SNN:
    """
    Método “AS-SNN” (simplificado) con regularización de actividad global.

    Args:
        lambda_a     : Peso de la penalización de actividad (>= 0).
        gamma_ratio  : Objetivo de actividad en [0,1] (p.ej. 0.5 => “moderada”).
        ema          : Factor de suavizado exponencial de la actividad (en (0,1)).
        name_suffix  : Sufijo opcional para el nombre (solo naming en outputs).
    """

    name = "as-snn"

    def __init__(
        self,
        lambda_a: float = 2.5,
        gamma_ratio: float = 0.5,
        ema: float = 0.9,
        name_suffix: str = "",
        **kw,  # se ignoran kwargs desconocidos para ser robustos con presets
    ) -> None:
        assert 0.0 <= gamma_ratio <= 1.0, "gamma_ratio debe estar en [0,1]"
        assert 0.0 < ema < 1.0, "ema debe estar en (0,1)"
        assert lambda_a >= 0.0, "lambda_a debe ser >= 0"

        self.lambda_a: float = float(lambda_a)
        self.gamma_ratio: float = float(gamma_ratio)
        self.ema: float = float(ema)

        # Nombre legible en outputs
        tag = f"{self.name}"
        tag += f"_gr_{self.gamma_ratio:g}" if gamma_ratio is not None else ""
        tag += f"_lam_{self.lambda_a:g}" if lambda_a is not None else ""
        if name_suffix:
            tag += f"_{name_suffix}"
        self.name = tag

        # Buffers internos (se crean perezosamente en el primer batch)
        self._alpha_ema: Optional[torch.Tensor] = None  # EMA(actividad)
        self._last_alpha: Optional[torch.Tensor] = None  # actividad instantánea
        self._penalty_value: Optional[torch.Tensor] = None  # φ(α) * lambda_a
        self._device: Optional[torch.device] = None

        # Handle del hook para poder desregistrarlo si hiciera falta
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

        # ---- Flags de logging inyectables desde preset ----
        self.activity_verbose: bool = False  # imprime estado de actividad
        self.activity_every: int = 100       # frecuencia (batches)
        self._batch_idx: int = 0

    # -------------------------------------------------------------------------
    # Utilidades internas
    # -------------------------------------------------------------------------
    def _maybe_init_buffers(self, device: torch.device) -> None:
        """
        Crea (o mueve) los buffers internos al device del batch actual.
        Evita mismatches CUDA/CPU.
        """
        if self._alpha_ema is None:
            # Primera vez: crea en el device adecuado
            self._alpha_ema = torch.tensor(0.0, device=device)
            self._last_alpha = torch.tensor(0.0, device=device)
            self._penalty_value = torch.tensor(0.0, device=device)
            self._device = device
        else:
            # Si ya existen pero están en otro device, muévelos
            if self._alpha_ema.device != device:
                self._alpha_ema = self._alpha_ema.to(device)
                self._last_alpha = self._last_alpha.to(device)
                self._penalty_value = self._penalty_value.to(device)
                self._device = device

    @staticmethod
    def _as_float_tensor(x: torch.Tensor) -> torch.Tensor:
        """
        Convierte a float32 si no lo es ya, sin crear cópias innecesarias.
        """
        return x if x.dtype.is_floating_point else x.float()

    @staticmethod
    def _estimate_activity(x: torch.Tensor) -> torch.Tensor:
        """
        Estima una “actividad media” α ∈ [0,1] a partir del tensor de entrada.
        - Si x son “spikes” binarios o intensidades [0,1], la media ya está en [0,1].
        - Acepta formas (T,B,C,H,W) o (T,B,H,W) u otras similares; realizamos un mean().
        - No se asume nada de la codificación temporal concreta; es una medida global.

        Devuelve: escalar (tensor 0D) en el mismo device de x, DETACHED.
        """
        x_f = AS_SNN._as_float_tensor(x)
        # Opcional: recorta a [0,1] por robustez ante codificaciones no acotadas
        x_f = torch.clamp(x_f, 0.0, 1.0)
        # Media global
        alpha_now = x_f.mean()
        # NO construimos grafo aquí: detach para que la penalty no requiera backward
        return alpha_now.detach()

    def _phi_piecewise(self, alpha: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        Regularizador “piecewise” simple: φ(α) = |α - γ|.
        - Es simétrico y empuja la actividad hacia γ sin saturar a cero gradiente
          (si lo conectáramos a la red). Aquí lo usamos solo como métrica detached.
        - Devuelve tensor escalar en el mismo device que alpha.
        """
        return torch.abs(alpha - gamma)

    # -------------------------------------------------------------------------
    # Hook: se ejecuta justo antes del forward del modelo
    # -------------------------------------------------------------------------
    def _pre_forward_hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
        """
        Hook de “pre-forward”: recibe el batch (ya movido a device por tu loop).
        Calcula alpha_now, actualiza el EMA y pre-computa la penalización φ(α)*λ.
        Todo con tensores DETACHED para no añadir dependencias en autograd.
        """
        if not inputs:
            return
        x = inputs[0]
        if not isinstance(x, torch.Tensor):
            # Por si algún modelo recibe inputs complejos; ignoramos el resto
            return

        device = x.device
        self._maybe_init_buffers(device)

        # 1) Actividad instantánea (DETACHED)
        alpha_now = self._estimate_activity(x)  # escalar 0D (detached, en device)
        # 2) EMA
        if self._alpha_ema is None:
            # defensa extra (no debería entrar aquí)
            self._alpha_ema = alpha_now.clone()
            self._last_alpha = alpha_now.clone()
        else:
            # alpha_ema = ema * alpha_ema + (1-ema) * alpha_now
            self._alpha_ema.mul_(self.ema).add_(alpha_now, alpha=(1.0 - self.ema))
            self._last_alpha.copy_(alpha_now)

        # 3) Penalización φ(α) con γ fijo = gamma_ratio (tensor en device)
        gamma = torch.tensor(self.gamma_ratio, device=device)
        phi = self._phi_piecewise(self._alpha_ema, gamma)  # escalar 0D
        # Guardamos la penalty ya escalada; DETACHED (no crea grafos)
        self._penalty_value = (self.lambda_a * phi).detach()

        # ---- Logging opcional y barato ----
        self._batch_idx += 1
        if getattr(self, "activity_verbose", False):
            every = max(1, int(getattr(self, "activity_every", 100)))
            if (self._batch_idx % every) == 0:
                try:
                    a_ema = float(self._alpha_ema.item())
                    a_now = float(alpha_now.item())
                    pen = float(self._penalty_value.item()) if self._penalty_value is not None else 0.0
                    print(f"[AS-SNN] α_now={a_now:.4f} | α_ema={a_ema:.4f} | γ={self.gamma_ratio:.2f} | λ_a={self.lambda_a:.3g} | pen={pen:.4g}")
                except Exception:
                    pass

    # -------------------------------------------------------------------------
    # API ContinualMethod
    # -------------------------------------------------------------------------
    def penalty(self) -> torch.Tensor:
        """
        Devuelve el último valor de penalización φ(α)*λ.
        - Si aún no se ha visto ningún batch, devuelve 0 en el device actual
          (si no hay device aún, se crea en CPU).
        - Está DETACHED a propósito: no construye grafos adicionales.
        """
        if self._penalty_value is not None:
            return self._penalty_value
        # fallback (CPU) si nunca pasó ningún batch por el hook
        return torch.zeros((), dtype=torch.float32)

    def before_task(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """
        Registra el hook si no está registrado. No inicializa buffers todavía:
        dejamos que el primer batch defina el device real (CPU o CUDA).
        """
        if self._hook_handle is None:
            # Nota: with_kwargs=False para máxima compatibilidad
            self._hook_handle = model.register_forward_pre_hook(
                self._pre_forward_hook, with_kwargs=False
            )

    def after_task(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """
        Punto de inserción para "Synaptic Scaling" tras cada tarea.
        En esta versión *no* tocamos los pesos por defecto (no-op),
        pero aquí podrías:
          1) Medir actividad por capa con forward hooks,
          2) Calcular factor s_l = gamma / alpha_l,
          3) Escalar pesos de la capa l con w_l <- s_l * w_l (y rebalancear bias).
        """
        # --- NO-OP por defecto ---
        # Si quisieras desregistrar el hook al final de cada tarea:
        # if self._hook_handle is not None:
        #     self._hook_handle.remove()
        #     self._hook_handle = None
        pass

    def detach(self) -> None:
        if self._hook_handle is not None:
            try:
                self._hook_handle.remove()
            except Exception:
                pass
            self._hook_handle = None

    # -------------------------------------------------------------------------
    # Utilidad introspectiva (debug / logging opcional)
    # -------------------------------------------------------------------------
    def get_activity_state(self) -> dict:
        """
        Devuelve un dict con las últimas métricas internas (para logging).
        """
        out = {
            "alpha_ema": (None if self._alpha_ema is None else float(self._alpha_ema.item())),
            "alpha_last": (None if self._last_alpha is None else float(self._last_alpha.item())),
            "penalty": (None if self._penalty_value is None else float(self._penalty_value.item())),
            "gamma_ratio": self.gamma_ratio,
            "lambda_a": self.lambda_a,
            "ema": self.ema,
        }
        return out

    def __repr__(self) -> str:
        return (
            f"AS_SNN(name={self.name!r}, lambda_a={self.lambda_a}, "
            f"gamma_ratio={self.gamma_ratio}, ema={self.ema})"
        )

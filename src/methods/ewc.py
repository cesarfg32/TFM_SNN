# -*- coding: utf-8 -*-
"""EWC (Elastic Weight Consolidation) para mitigar el olvido catastrófico."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from src.methods.base import BaseMethod
from src.nn_io import _forward_with_cached_orientation, _align_target_shape


@dataclass
class EWCConfig:
    # Peso de la penalización
    lambd: float = 0.0
    # Nº de batches para estimar Fisher
    fisher_batches: int = 25
    # Precisión del cálculo de Fisher:
    #  - "fp32" (por defecto): más estable, más lento
    #  - "bf16" / "fp16": rápido, posible infrarrepresentación del grad
    #  - "amp": usa autocast con dtype por defecto o el indicado en fisher_amp_dtype
    fisher_precision: str = "fp32"
    # Si fisher_precision == "amp", se puede forzar dtype: "bf16" | "fp16"
    fisher_amp_dtype: Optional[str] = None
    # Política de permuta/orientación temporal:
    #  - "auto": usa la pista del batch (y) para normalizar (T,B,...) si procede
    #  - "skip": no pasamos 'y' al forward helper (evita permutas si tu modelo ya es coherente)
    permute_policy: str = "auto"


class EWC:
    """Implementación EWC con Fisher diagonal (modelo perezoso; se inyecta antes de consolidar)."""
    name = "ewc"

    def __init__(
        self,
        *,
        model: Optional[nn.Module] = None,
        cfg: EWCConfig,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.cfg = cfg
        self.loss_fn = loss_fn or nn.MSELoss()
        self.device = (
            device
            if device is not None
            else (torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"))
        )
        self._mu: Dict[str, torch.Tensor] = {}
        self._fisher: Dict[str, torch.Tensor] = {}

    def _snapshot_params(self) -> Dict[str, torch.Tensor]:
        assert self.model is not None, "[EWC] model no inicializado"
        return {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def _resolve_amp(self) -> tuple[bool, Optional[torch.dtype]]:
        """Devuelve (usar_autocast, dtype_autocast)."""
        mode = (self.cfg.fisher_precision or "fp32").lower()
        if self.device.type != "cuda":
            return (False, None)
        if mode == "fp32":
            return (False, None)
        if mode == "bf16":
            return (True, torch.bfloat16)
        if mode == "fp16":
            return (True, torch.float16)
        if mode == "amp":
            dt = None
            if isinstance(self.cfg.fisher_amp_dtype, str):
                s = self.cfg.fisher_amp_dtype.lower()
                if s == "bf16":
                    dt = torch.bfloat16
                elif s == "fp16":
                    dt = torch.float16
            return (True, dt)
        # fallback conservador
        return (False, None)

    def estimate_fisher(self, loader) -> int:
        """
        Estima Fisher ≈ E[(∂L/∂θ)^2] usando batches del loader (post-tarea).

        Cambios clave vs. versión anterior:
        - Permite AMP/BF16/FP16 para acelerar (configurable).
        - Evita casts innecesarios: la loss se calcula en el dtype de y_hat.
        - Opción 'permute_policy="skip"' para saltar la permuta (si el modelo ya está OK).
        """
        assert self.model is not None, "[EWC] model no inicializado"
        self.model.to(self.device)
        was_training = self.model.training
        self.model.eval()

        fisher = {
            n: torch.zeros_like(p, device=p.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        # θ* de la tarea que ACABA de terminar
        self._mu = self._snapshot_params()

        n_batches = 0
        cap = int(self.cfg.fisher_batches)
        hint = {"fisher": None}

        use_autocast, amp_dtype = self._resolve_amp()

        # Nota: necesitamos gradientes → no usar inference_mode
        with torch.enable_grad():
            # Autocast externo; el helper interno se llama con use_amp=False para no anidar
            ctx = torch.amp.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=bool(use_autocast),
            ) if self.device.type == "cuda" else torch.no_grad()  # placeholder, no se usa en CPU

            # Usamos el contexto sólo si está habilitado
            if self.device.type == "cuda" and bool(use_autocast):
                cm = ctx
            else:
                # contexto vacío
                class _NullCtx:
                    def __enter__(self): return None
                    def __exit__(self, *args): return False
                cm = _NullCtx()

            with cm:
                for x, y in loader:
                    y = y.to(self.device, non_blocking=True)
                    # Si queremos evitar permutas, no pasamos 'y' como pista
                    y_for_forward = y if (self.cfg.permute_policy.lower() != "skip") else None

                    # Forward SIN AMP interno (ya estamos en autocast externo si aplica)
                    y_hat = _forward_with_cached_orientation(
                        self.model, x, y_for_forward,
                        self.device, use_amp=False,
                        phase_hint=hint, phase="fisher"
                    )

                    # Alineación del target a dtype/shape de y_hat sin forzar FP32
                    target = _align_target_shape(y_hat, y).to(
                        device=y_hat.device, dtype=y_hat.dtype, non_blocking=True
                    )

                    # Pérdida en el dtype actual (fp32/bf16/fp16 según caso)
                    loss = self.loss_fn(y_hat, target).mean()

                    self.model.zero_grad(set_to_none=True)
                    loss.backward()

                    for n, p in self.model.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            # Acumulamos en el dispositivo del propio parámetro
                            fisher[n] += p.grad.detach().pow(2)

                    n_batches += 1
                    if n_batches >= cap:
                        break

        if n_batches > 0:
            for n in fisher:
                fisher[n] /= float(n_batches)

        self._fisher = fisher
        if was_training:
            self.model.train()
        return n_batches

    def penalty(self) -> torch.Tensor:
        """λ * Σ F_i (θ_i - θ*_i)^2."""
        if not self._fisher:
            return torch.tensor(0.0, device=self.device)
        assert self.model is not None, "[EWC] model no inicializado"
        reg = 0.0
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self._fisher:
                reg = reg + (self._fisher[n] * (p - self._mu[n]).pow(2)).sum()
        return float(self.cfg.lambd) * reg


class EWCMethod(BaseMethod):
    """Wrapper homogéneo: recibe device/loss_fn aquí; consolida en after_task."""
    name = "ewc"

    def __init__(
        self,
        *,
        lambd: float,
        fisher_batches: int = 100,
        # Nuevos parámetros de rendimiento/precisión
        fisher_precision: str = "fp32",
        fisher_amp_dtype: Optional[str] = None,  # "bf16" | "fp16" | None
        permute_policy: str = "auto",            # "auto" | "skip"
        device: Optional[torch.device] = None,
        loss_fn: Optional[nn.Module] = None,
    ):
        super().__init__(device=device, loss_fn=loss_fn)
        self.impl = EWC(
            model=None,
            cfg=EWCConfig(
                lambd=float(lambd),
                fisher_batches=int(fisher_batches),
                fisher_precision=str(fisher_precision),
                fisher_amp_dtype=fisher_amp_dtype,
                permute_policy=str(permute_policy),
            ),
            loss_fn=loss_fn,
            device=self.device,
        )
        self.name = f"ewc_lam_{lambd:.0e}"
        # Flags inyectables desde cfg['logging']['ewc']
        self.inner_verbose = getattr(self, "inner_verbose", False)
        self.inner_every = getattr(self, "inner_every", 50)
        # fisher_verbose se inyecta dinámicamente

    def penalty(self) -> torch.Tensor:
        return self.impl.penalty()

    def before_task(self, model: nn.Module, train_loader, val_loader) -> None:
        # Sólo alinea model/device; NO consolidar aquí
        self.impl.model = model
        try:
            dev_model = next(model.parameters()).device
            if self.impl.device != dev_model:
                self.impl.device = dev_model
        except StopIteration:
            pass

    def after_task(self, model: nn.Module, train_loader, val_loader) -> None:
        # Consolidación post-entrenamiento (como en master antiguo / paper)
        used = self.impl.estimate_fisher(train_loader)
        if bool(getattr(self, "fisher_verbose", False)):
            try:
                if self.impl._fisher:
                    abs_sum = float(sum(f.abs().sum().item() for f in self.impl._fisher.values()))
                    abs_max = float(max(f.abs().max().item() for f in self.impl._fisher.values()))
                    print(f"[EWC] Fisher listo: batches_usados={used} | sum={abs_sum:.3e} | max={abs_max:.3e}")
                else:
                    print(f"[EWC] Fisher listo: batches_usados={used} | (vacío)")
            except Exception:
                pass

    def tunable(self) -> dict:
        try:
            val = float(self.impl.cfg.lambd)
        except Exception:
            val = 0.0
        return {"param": "lambd", "value": val, "strategy": "ratio", "target_ratio": 1.0}

# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Tuple, Dict, List, Any, Union
import torch
from torch import nn
from torch.utils.data import DataLoader
from .base import BaseMethod

def _resolve_modules_by_name(model: nn.Module, name_substr: str) -> List[tuple[str, nn.Module]]:
    out: List[tuple[str, nn.Module]] = []
    low = name_substr.lower()
    for n, m in model.named_modules():
        if low in n.lower():
            out.append((n, m))
    return out

def _norm_scale_clip(sc: Any) -> Tuple[float, float]:
    if sc is None: return (0.5, 2.0)
    if isinstance(sc, (list, tuple)):
        if len(sc) == 2: lo, hi = sc; return (float(lo), float(hi))
        if len(sc) == 1: v = float(sc[0]); return (v, v)
    try:
        v = float(sc); return (v, v)
    except Exception:
        return (0.5, 2.0)

def _first_tensor(x) -> Optional[torch.Tensor]:
    if isinstance(x, torch.Tensor): return x
    if isinstance(x, (tuple, list)) and x:
        return x[0] if isinstance(x[0], torch.Tensor) else None
    return None

class AS_SNN(BaseMethod):
    """
    Regularización de actividad con gradiente (por-capa) + synaptic scaling opcional.
    """
    name = "as-snn"

    def __init__(
        self,
        *,
        lambda_a: float = 2.5,
        gamma_ratio: float = 0.5,
        ema: float = 0.9,
        penalty_mode: str = "l1",
        measure_at: Optional[str] = None,
        attach_to: Optional[str] = None,
        do_synaptic_scaling: bool = False,
        scale_clip: Union[float, Tuple[float, float], List[float]] = (0.5, 2.0),
        scale_bias: bool = False,
        eps: float = 1e-6,
        name_suffix: str = "",
        activity_verbose: bool = False,
        activity_every: int = 100,
        device: Optional[torch.device] = None,
        loss_fn: Optional[nn.Module] = None,
        **kw,
    ) -> None:
        super().__init__(device=device, loss_fn=loss_fn)
        assert 0.0 <= gamma_ratio <= 1.0
        assert 0.0 < ema < 1.0
        assert lambda_a >= 0.0
        assert penalty_mode in ("l1", "l2")

        self.lambda_a = float(lambda_a)
        self.gamma_ratio = float(gamma_ratio)
        self.ema = float(ema)
        self.penalty_mode = penalty_mode
        self.attach_to = attach_to
        self.do_synaptic_scaling = bool(do_synaptic_scaling)
        self.scale_clip = _norm_scale_clip(scale_clip)
        assert self.scale_clip[0] > 0 and self.scale_clip[0] <= self.scale_clip[1]
        self.scale_bias = bool(scale_bias)
        self.eps = float(eps)

        if measure_at is None:
            self.measure_at = "modules" if (attach_to is not None) else "input"
        else:
            assert measure_at in ("modules", "input", "both")
            self.measure_at = measure_at

        tag = "as-snn"
        tag += f"_gr_{self.gamma_ratio:g}"
        tag += f"_lam_{self.lambda_a:g}"
        if self.attach_to: tag += f"_att_{self.attach_to}"
        if self.do_synaptic_scaling: tag += "_scale_on"
        if name_suffix: tag += f"_{name_suffix}"
        self.name = tag

        self._device: Optional[torch.device] = None
        self._batch_penalties: List[torch.Tensor] = []
        self._alpha_in_ema: Optional[torch.Tensor] = None
        self._alpha_in_last: Optional[torch.Tensor] = None
        self._layer_stats: Dict[str, Dict[str, torch.Tensor | nn.Module]] = {}
        self._pre_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._fw_handles: List[torch.utils.hooks.RemovableHandle] = []
        self.activity_verbose = bool(activity_verbose)
        self.activity_every = int(activity_every)
        self._batch_idx = 0

    # helpers + hooks (idéntico a tu versión anterior) ...
    @staticmethod
    def _as_float_unit(x: torch.Tensor) -> torch.Tensor:
        if not x.dtype.is_floating_point: x = x.float()
        return torch.clamp(x, 0.0, 1.0)

    def _maybe_on_device(self, t: torch.Tensor) -> None:
        if self._device is None: self._device = t.device

    def _ensure_layer_entry(self, name: str, device: torch.device, init_val: torch.Tensor) -> None:
        if name not in self._layer_stats:
            self._layer_stats[name] = {
                "alpha_ema": init_val.detach().clone().to(device),
                "alpha_last": init_val.detach().clone().to(device),
                "module": None,
            }
        else:
            for k in ("alpha_ema", "alpha_last"):
                self._layer_stats[name][k] = self._layer_stats[name][k].to(device)  # type: ignore[index]

    def _phi(self, a: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return (a - gamma).pow(2) if self.penalty_mode == "l2" else torch.abs(a - gamma)

    def _pre_forward_hook(self, module: nn.Module, inputs):
        self._batch_penalties.clear()
        x = None
        if isinstance(inputs, tuple) and inputs:
            x = _first_tensor(inputs[0]) if not isinstance(inputs[0], torch.Tensor) else inputs[0]
        elif isinstance(inputs, torch.Tensor):
            x = inputs
        if x is None or not isinstance(x, torch.Tensor): return
        self._maybe_on_device(x); device = x.device
        if self.measure_at in ("input", "both"):
            a_now = self._as_float_unit(x).mean()
            if self._alpha_in_ema is None:
                self._alpha_in_ema = a_now.detach().clone().to(device)
                self._alpha_in_last = a_now.detach().clone().to(device)
            else:
                self._alpha_in_ema.mul_(self.ema).add_(a_now.detach(), alpha=(1.0 - self.ema))
                self._alpha_in_last.copy_(a_now.detach())
            gamma = torch.tensor(self.gamma_ratio, device=device, dtype=a_now.dtype)
            self._batch_penalties.append(self.lambda_a * self._phi(a_now, gamma))
        self._batch_idx += 1
        if self.activity_verbose and (self._batch_idx % max(1, self.activity_every)) == 0:
            try:
                a_in_ema = None if self._alpha_in_ema is None else float(self._alpha_in_ema.item())
                print(f"[AS-SNN] input_ema={a_in_ema} γ={self.gamma_ratio:.2f} λ={self.lambda_a:g}")
            except Exception:
                pass

    def _make_forward_hook(self, name: str):
        def _hook(module: nn.Module, inputs, output):
            y = _first_tensor(output) if not isinstance(output, torch.Tensor) else output
            if y is None: return
            self._maybe_on_device(y); device = y.device
            a_now = self._as_float_unit(y).mean()
            self._ensure_layer_entry(name, device, a_now)
            self._layer_stats[name]["alpha_ema"].mul_(self.ema).add_(a_now.detach(), alpha=(1.0 - self.ema))  # type: ignore[index]
            self._layer_stats[name]["alpha_last"].copy_(a_now.detach())  # type: ignore[index]
            gamma = torch.tensor(self.gamma_ratio, device=device, dtype=a_now.dtype)
            self._batch_penalties.append(self.lambda_a * self._phi(a_now, gamma))
        return _hook

    # API BaseMethod
    def before_task(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
        if self._pre_hook_handle is None:
            self._pre_hook_handle = model.register_forward_pre_hook(self._pre_forward_hook, with_kwargs=False)
        if self.measure_at in ("modules", "both") and not self._fw_handles:
            modules_to_hook: List[tuple[str, nn.Module]] = []
            if self.attach_to:
                modules_to_hook.extend(_resolve_modules_by_name(model, self.attach_to))
            else:
                for n, m in model.named_modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        modules_to_hook.append((n, m))
            registered = set()
            for name, mod in modules_to_hook:
                if mod in registered: continue
                h = mod.register_forward_hook(self._make_forward_hook(name), with_kwargs=False)
                self._fw_handles.append(h)
                p0 = next(mod.parameters(), None)
                dev = p0.device if p0 is not None else (self._device or torch.device("cpu"))
                self._ensure_layer_entry(name, dev, torch.zeros((), device=dev))
                self._layer_stats[name]["module"] = mod  # type: ignore[index]
                registered.add(mod)

    def penalty(self) -> torch.Tensor:
        if not self._batch_penalties:
            dev = self._device if self._device is not None else torch.device("cpu")
            return torch.zeros((), dtype=torch.float32, device=dev)
        return torch.stack(self._batch_penalties).sum()

    @torch.no_grad()
    def after_task(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
        if not self.do_synaptic_scaling: return
        lo, hi = self.scale_clip
        for name, stats in self._layer_stats.items():
            alpha_ema = stats.get("alpha_ema", None)
            mod = stats.get("module", None)
            if alpha_ema is None or mod is None or not hasattr(mod, "weight"): continue
            a = float(alpha_ema.item())
            if a <= 0.0: continue
            s_nominal = self.gamma_ratio / max(a, self.eps)
            s = float(max(lo, min(hi, s_nominal)))
            try:
                mod.weight.mul_(s)  # type: ignore[attr-defined]
                if self.scale_bias and getattr(mod, "bias", None) is not None:
                    mod.bias.mul_(s)   # type: ignore[attr-defined]
            except Exception:
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

    # introspección (igual que tu versión)
    def get_activity_state(self) -> dict:
        out = {
            "gamma_ratio": self.gamma_ratio,
            "lambda_a": self.lambda_a,
            "ema": self.ema,
            "penalty_mode": self.penalty_mode,
            "measure_at": self.measure_at,
            "attach_to": self.attach_to,
            "do_synaptic_scaling": self.do_synaptic_scaling,
            "scale_clip": self.scale_clip,
        }
        out["input_alpha_ema"] = None if self._alpha_in_ema is None else float(self._alpha_in_ema.item())
        out["input_alpha_last"] = None if self._alpha_in_last is None else float(self._alpha_in_last.item())
        layers = {}
        for name, st in self._layer_stats.items():
            layers[name] = {
                "alpha_ema": float(st["alpha_ema"].item()) if isinstance(st.get("alpha_ema"), torch.Tensor) else None,  # type: ignore[index]
                "alpha_last": float(st["alpha_last"].item()) if isinstance(st.get("alpha_last"), torch.Tensor) else None,  # type: ignore[index]
            }
        out["layers"] = layers
        return out

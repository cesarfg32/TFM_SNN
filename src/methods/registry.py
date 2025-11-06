# src/methods/registry.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, List, Dict, Any
from torch import nn
import torch

from .api import ContinualMethod
from .naive import Naive
from .ewc import EWCMethod
from .rehearsal import RehearsalMethod, RehearsalConfig
from .composite import CompositeMethod

# Nuevos
from .sa_snn import SASNN, SASNNConfig
from .as_snn import AS_SNN
from .sca_snn import SCA_SNN
from .colanet import CoLaNET

# --------------------------------------------------------------------------------------
# Filtros de kwargs por método (evita pasar lam/fisher a SA/AS/SCA, etc.)
# --------------------------------------------------------------------------------------

EWC_KEYS = {"lam", "ewc_lam", "fisher_batches"}

ALLOWED_KEYS: Dict[str, set[str]] = {
    "sa-snn": {
        "attach_to", "k", "tau", "th_min", "th_max", "p", "vt_scale",
        "flatten_spatial", "assume_binary_spikes", "reset_counters_each_task"
    },
    "as-snn": {
        "measure_at", "attach_to", "gamma_ratio", "lambda_a", "ema",
        "do_synaptic_scaling", "scale_clip", "scale_bias", "penalty_mode"
    },
    "sca-snn": {
        "attach_to", "flatten_spatial", "num_bins", "anchor_batches",
        "bin_lo", "bin_hi", "max_per_bin", "beta", "bias",
        "soft_mask_temp", "habit_decay", "verbose", "log_every"
    },
    "rehearsal": {"buffer_size", "replay_ratio"},
    "naive": set(),
    "colanet": {"attach_to", "flatten_spatial"},  # amplía si tu implementación necesita más
}

# Parámetros “genéricos” que pueden colarse desde el preset y nunca deben ir a métodos
GENERIC_TO_DROP = {
    "T", "epochs", "batch_size", "lr", "es_patience", "es_min_delta",
    "compile", "compile_mode", "amp", "encoder", "gain"
}

def _filter_kwargs_for(method_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Conserva solo las claves permitidas del método y quita genéricos."""
    method = method_name.lower().strip()
    allowed = ALLOWED_KEYS.get(method, set())
    return {
        k: v for k, v in kwargs.items()
        if (k in allowed) and (k not in GENERIC_TO_DROP)
    }

def _ewc_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Extrae solo las claves propias de EWC y normaliza 'lam'."""
    out = {k: v for k, v in kwargs.items() if k in EWC_KEYS}
    if "lam" not in out and "ewc_lam" in out:
        out["lam"] = out.pop("ewc_lam")
    return out


# --------------------------------------------------------------------------------------
# build_method: construcción de métodos simples y composites con filtrado estricto
# --------------------------------------------------------------------------------------
def build_method(
    name: str,
    model: nn.Module,
    *,
    loss_fn: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    **method_kwargs,
) -> ContinualMethod:
    """
    Construye el método de aprendizaje continuo a partir del nombre.
    Soporta combinaciones tipo "<main>+ewc" o "ewc+<main>" con filtrado de kwargs.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    lname = name.lower().strip()

    # Limpia genéricos a nivel global (antes solo se filtraba "T")
    for k in list(method_kwargs.keys()):
        if k in GENERIC_TO_DROP:
            method_kwargs.pop(k, None)

    def _attach_name_and_hooks(obj: ContinualMethod, nm: str) -> ContinualMethod:
        # Garantiza .name
        if not hasattr(obj, "name"):
            try:
                setattr(obj, "name", nm)
            except Exception:
                pass
        # Hooks no-op por si faltan (para evitar AttributeError)
        for hook in (
            "before_task", "after_task", "before_epoch", "after_epoch",
            "before_batch", "after_batch", "close", "state_dict",
            "load_state_dict", "penalty"
        ):
            if not hasattr(obj, hook):
                if hook == "penalty":
                    setattr(obj, hook, (lambda self=obj: 0.0))
                elif hook == "state_dict":
                    setattr(obj, hook, (lambda self=obj: {}))
                elif hook == "load_state_dict":
                    setattr(obj, hook, (lambda state, self=obj: None))
                else:
                    setattr(obj, hook, (lambda *a, **kw: None))
        return obj

    def _build_base(main: str) -> ContinualMethod:
        m = main.lower().strip()

        if m == "naive":
            return _attach_name_and_hooks(Naive(), "naive")

        if m == "ewc":
            assert loss_fn is not None, "EWC requiere 'loss_fn'"
            ek = _ewc_kwargs(method_kwargs)
            lam = ek.get("lam", None)
            assert lam is not None, "EWC requiere 'lam' (o 'ewc_lam')"
            fisher_batches = int(ek.get("fisher_batches", 100))
            return _attach_name_and_hooks(
                EWCMethod(model, float(lam), fisher_batches, loss_fn, device), "ewc"
            )

        if m == "rehearsal":
            rk = _filter_kwargs_for("rehearsal", method_kwargs)
            cfg = RehearsalConfig(
                buffer_size=int(rk.get("buffer_size", 10_000)),
                replay_ratio=float(rk.get("replay_ratio", 0.2)),
            )
            return _attach_name_and_hooks(RehearsalMethod(cfg), "rehearsal")

        # --- SNN methods filtrando kwargs ---
        if m == "sa-snn":
            sk = _filter_kwargs_for("sa-snn", method_kwargs)
            cfg = SASNNConfig(**sk)
            obj = SASNN(model=model, cfg=cfg, device=device)
            return _attach_name_and_hooks(obj, "sa-snn")

        if m == "as-snn":
            ak = _filter_kwargs_for("as-snn", method_kwargs)
            return _attach_name_and_hooks(AS_SNN(**ak), "as-snn")

        if m == "sca-snn":
            ck = _filter_kwargs_for("sca-snn", method_kwargs)
            return _attach_name_and_hooks(SCA_SNN(**ck), "sca-snn")

        if m == "colanet":
            ck = _filter_kwargs_for("colanet", method_kwargs)
            return _attach_name_and_hooks(CoLaNET(**ck), "colanet")

        raise ValueError(f"Método desconocido: {main}")

    # --- Composite "<main>+ewc" y/o "ewc+<main>" (soporta varias piezas) ---
    if "+" in lname:
        parts: List[str] = [p.strip() for p in lname.split("+") if p.strip()]
        bases: List[ContinualMethod] = []

        for p in parts:
            pl = p.lower()
            if pl == "ewc":
                # EWC SOLO recibe sus claves
                ek = _ewc_kwargs(method_kwargs)
                lam = ek.get("lam", None)
                assert lam is not None, "Composite +EWC requiere 'lam' (o 'ewc_lam')"
                assert loss_fn is not None, "Composite +EWC requiere 'loss_fn'"
                fisher_batches = int(ek.get("fisher_batches", 100))
                bases.append(_attach_name_and_hooks(
                    EWCMethod(model, float(lam), fisher_batches, loss_fn, device), "ewc"
                ))
            elif pl == "sa-snn":
                sub = _filter_kwargs_for("sa-snn", method_kwargs)
                cfg = SASNNConfig(**sub)
                bases.append(_attach_name_and_hooks(SASNN(model=model, cfg=cfg, device=device), "sa-snn"))
            elif pl == "as-snn":
                sub = _filter_kwargs_for("as-snn", method_kwargs)
                bases.append(_attach_name_and_hooks(AS_SNN(**sub), "as-snn"))
            elif pl == "sca-snn":
                sub = _filter_kwargs_for("sca-snn", method_kwargs)
                bases.append(_attach_name_and_hooks(SCA_SNN(**sub), "sca-snn"))
            elif pl == "rehearsal":
                sub = _filter_kwargs_for("rehearsal", method_kwargs)
                cfg = RehearsalConfig(
                    buffer_size=int(sub.get("buffer_size", 10_000)),
                    replay_ratio=float(sub.get("replay_ratio", 0.2)),
                )
                bases.append(_attach_name_and_hooks(RehearsalMethod(cfg), "rehearsal"))
            elif pl == "naive":
                bases.append(_attach_name_and_hooks(Naive(), "naive"))
            elif pl == "colanet":
                sub = _filter_kwargs_for("colanet", method_kwargs)
                bases.append(_attach_name_and_hooks(CoLaNET(**sub), "colanet"))
            else:
                raise ValueError(f"Método desconocido dentro de composite: {p}")

        comp = CompositeMethod(bases)
        if not hasattr(comp, "name"):
            comp_name = "+".join(getattr(b, "name", b.__class__.__name__.lower()) for b in bases)
            _attach_name_and_hooks(comp, comp_name)
        return comp

    # Método simple
    return _build_base(lname)

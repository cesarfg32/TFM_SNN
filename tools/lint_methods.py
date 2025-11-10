#!/usr/bin/env python3
# tools/lint_methods.py
# -*- coding: utf-8 -*-
from __future__ import annotations

# --- Asegura que la raíz del repo está en sys.path ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from src.methods.registry import build_method

# ---- Modelo dummy con un submódulo llamado 'f6' ----
class FakePilotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Flatten())
        # Submódulo con el nombre exacto que esperan los métodos
        self.f6 = nn.Linear(4, 1)

    def forward(self, x):
        # Soporta (B,4) o cualquier cosa que se aplane a 4 por batch
        x = self.stem(x)
        if x.shape[-1] != 4:
            # adapta a 4 características para no fallar en el forward
            x = nn.functional.adaptive_avg_pool1d(x.unsqueeze(1), 4).squeeze(1)
        return self.f6(x)

def _fake_model():
    return FakePilotNet()

# Kwargs mínimos por método para “compilar” sin romper semántica
METHOD_DEFAULTS = {
    "ewc": dict(lam=3e6, fisher_batches=5),
    "sa-snn": dict(
        attach_to="f6", k=6, tau=28,
        th_min=1.0, th_max=2.0, p=2_000_000,
        vt_scale=1.2, flatten_spatial=False,
        assume_binary_spikes=False, reset_counters_each_task=False
    ),
    "as-snn": dict(
        attach_to="f6", measure_at="f6",
        gamma_ratio=0.33, lambda_a=1.0, ema=0.95,
        do_synaptic_scaling=True, scale_clip=2.0, scale_bias=0.0,
        penalty_mode="l2"   # <- válido: 'l1' o 'l2'
    ),
    "sca-snn": dict(
        attach_to="f6", flatten_spatial=False, num_bins=60,
        anchor_batches=8, bin_lo=0.0, bin_hi=1.0, max_per_bin=24,
        beta=0.5, bias=0.0, soft_mask_temp=0.75, habit_decay=0.0,
        verbose=False, log_every=50
    ),
    "rehearsal": dict(buffer_size=256, replay_ratio=0.2),
    "naive": dict(),
    "colanet": dict(attach_to="f6", flatten_spatial=False),
}

def _merge_kwargs(parts):
    """Fusiona kwargs por partes en composites (p.ej., 'as-snn+ewc')."""
    merged = {}
    for p in parts:
        d = METHOD_DEFAULTS.get(p, {})
        for k, v in d.items():
            if k not in merged:
                merged[k] = v
    return merged

def _check(name: str) -> bool:
    ok = True
    parts = [s.strip().lower() for s in name.split("+")]
    kwargs = _merge_kwargs(parts)

    try:
        m = build_method(
            name, _fake_model(), loss_fn=nn.MSELoss(), device=torch.device("cpu"),
            **kwargs
        )
    except Exception as e:
        print(f"[FAIL] build_method('{name}') lanzó excepción: {e}")
        return False

    # Contrato mínimo
    for attr in ("before_task", "after_task", "penalty", "name"):
        if not hasattr(m, attr):
            print(f"[FAIL] {name}: falta atributo '{attr}'")
            ok = False

    # penalty debe ser escalar (float o tensor escalar)
    try:
        p = m.penalty()
        if isinstance(p, torch.Tensor):
            _ = float(p.detach().cpu().item())
        else:
            _ = float(p)
    except Exception as e:
        print(f"[FAIL] {name}: penalty no es escalar numérico. ({type(p)}) err={e}")
        ok = False

    if ok:
        print(f"[OK] {name}")
    return ok

if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or [
        "naive",
        "ewc",
        "rehearsal",
        "sa-snn",
        "as-snn",
        "sca-snn",
        "colanet",
        "as-snn+ewc",
        "sca-snn+ewc",
    ]
    any_fail = False
    for n in names:
        any_fail |= (not _check(n))
    sys.exit(1 if any_fail else 0)

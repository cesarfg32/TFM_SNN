# src/selftest.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, traceback
from typing import Dict, Any
import torch

def _p(msg: str): print(msg, flush=True)

def quickcheck(preset: str = "fast") -> int:
    """
    Comprueba: load_preset -> build_components -> primer batch -> forward -> penalty().
    Devuelve 0 si OK, >0 si algún chequeo falla.
    """
    from src.config import load_preset
    from src.utils_components import build_components_for
    from src.methods.registry import build_method
    from src.eval import eval_loader
    from torch import nn

    try:
        cfg: Dict[str, Any] = load_preset(preset)
        _p(f"[CFG] preset='{preset}' cargado.")
    except Exception:
        traceback.print_exc(); return 1

    try:
        mk_loader, mk_model, tfm = build_components_for(cfg)
        _p("[OK] Componentes construidos (loader+model+tfm).")
    except Exception:
        traceback.print_exc(); return 2

    data = cfg["data"]; cont = cfg["continual"]
    encoder, T, gain, seed = data["encoder"], int(data["T"]), float(data["gain"]), int(data.get("seed", 42))

    # 1) Carga un batch y forward
    try:
        tr, va, te = mk_loader(task={"name": "circuito1"}, batch_size=8, encoder=encoder, T=T, gain=gain, tfm=tfm, seed=seed)
        xb, yb = next(iter(tr))
        model = mk_model(tfm).eval().to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.inference_mode():
            # Reutilizamos el forward unificado del training/eval para asegurar orientaciones
            from src.nn_io import _forward_with_cached_orientation
            y_hat = _forward_with_cached_orientation(model=model, x=xb, y=yb, device=next(model.parameters()).device,
                                                    use_amp=bool(data.get("amp", True) and torch.cuda.is_available()),
                                                    phase_hint={"selftest": None}, phase="selftest")
        _p(f"[OK] Forward de humo: y_hat shape={tuple(y_hat.shape)}")
    except Exception:
        traceback.print_exc(); return 3

    # 2) Eval de humo (MSE/MAE de 1-2 batches)
    try:
        loss_fn = nn.MSELoss()
        mse, mae = eval_loader(va, model, loss_fn)  # eval_loader infiere device/use_amp
        _p(f"[OK] Eval humo: mse≈{mse:.4f} | mae≈{mae:.4f}")
    except Exception:
        traceback.print_exc(); return 4

    # 3) Método EWC de humo (penalty escalar)
    try:
        method = build_method(cont.get("method", "ewc"), model, loss_fn=loss_fn, device=next(model.parameters()).device, **(cont.get("params", {}) or {}))
        pen = method.penalty()
        if isinstance(pen, torch.Tensor): pen = float(pen.detach().item())
        _p(f"[OK] penalty() escalar ({cont.get('method')}): {pen:.4g}")
    except Exception:
        traceback.print_exc(); return 5

    _p("[SELFTEST] OK")
    return 0

if __name__ == "__main__":
    pr = sys.argv[1] if len(sys.argv) > 1 else "fast"
    sys.exit(quickcheck(pr))

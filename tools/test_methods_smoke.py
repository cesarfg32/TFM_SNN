# tools/test_methods_smoke.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import traceback, sys
from torch import nn
import torch

def _run_one(method_name: str, params: dict, preset: str="fast") -> bool:
    from src.config import load_preset
    from src.utils_components import build_components_for
    from src.methods.registry import build_method
    from src.eval import eval_loader

    cfg = load_preset(preset)
    mk_loader, mk_model, tfm = build_components_for(cfg)
    tr, va, te = mk_loader(task={"name":"circuito1"}, batch_size=8,
                           encoder=cfg["data"]["encoder"], T=cfg["data"]["T"],
                           gain=cfg["data"]["gain"], tfm=tfm, seed=cfg["data"]["seed"])
    model = mk_model(tfm).to("cuda" if torch.cuda.is_available() else "cpu")
    loss = nn.MSELoss()

    method = build_method(method_name, model, loss_fn=loss, device=next(model.parameters()).device, **(params or {}))

    # 1 forward de humo de train y 1 de val
    xb, yb = next(iter(tr))
    from src.nn_io import _forward_with_cached_orientation, _align_target_shape
    y_hat = _forward_with_cached_orientation(model=model, x=xb, y=yb,
        device=next(model.parameters()).device,
        use_amp=bool(cfg["optim"]["amp"] and torch.cuda.is_available()),
        phase_hint={"train": None}, phase="train")
    yb2 = _align_target_shape(y_hat, yb).to(device=y_hat.device, dtype=y_hat.dtype)
    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        loss_val = loss(y_hat, yb2) + (method.penalty() if hasattr(method, "penalty") else 0.0)
    float(loss_val.detach().item())  # debe ser convertible a float

    mse, mae = eval_loader(va, model, loss)  # smoke
    print(f"[{method_name}] OK | val mse≈{mse:.4f} mae≈{mae:.4f}")
    return True

if __name__ == "__main__":
    cases = [
        ("ewc", {"lam": 3e6, "fisher_batches": 10}),
        ("sca-snn", {"attach_to":"f6","num_bins":50,"beta":0.55,"soft_mask_temp":0.3,"anchor_batches":16}),
        # añade aquí as-snn / sa-snn con tus params válidos
    ]
    ok = True
    for name, params in cases:
        try:
            ok &= _run_one(name, params)
        except Exception:
            print(f"[{name}] FAIL")
            traceback.print_exc()
            ok = False
    sys.exit(0 if ok else 1)

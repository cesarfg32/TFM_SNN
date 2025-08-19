# src/runner.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import sys, json, torch
from typing import Optional

from src.utils import load_preset, set_seeds
from src.training import TrainConfig, train_supervised, set_runtime_encode
from src.methods.registry import build_method_with_optional_ewc
from src.eval import eval_loader

ROOT = Path.cwd().parents[0] if (Path.cwd().name == "notebooks") else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def run_continual(
    task_list: list[dict],          # [{"name": "...", "paths": {"train":..., "val":..., "test":...}}, ...]
    make_loader_fn,                 # callable(task, batch_size, encoder, T, gain, tfm, seed, **dl_kwargs) -> (tr,va,te)
    make_model_fn,                  # callable(tfm) -> nn.Module
    tfm,                            # ImageTransform
    preset: str,                    # "fast" | "std" | "accurate"
    method: str,                    # "ewc" | "naive"
    lam: float | None,              # λ si EWC
    seed: int,
    encoder: str,                   # "rate" | "latency" | "raw"
    fisher_batches_by_preset: Optional[dict[str,int]] = None,
    epochs_override: Optional[int] = None,
    runtime_encode: bool = True,    # activa set_runtime_encode para loaders 4D
    out_root: Path | str | None = None,
    verbose: bool = True,
):
    cfg = load_preset(ROOT/"configs"/"presets.yaml", preset)
    T      = int(cfg["T"])
    gain   = float(cfg["gain"])
    lr     = float(cfg["lr"])
    epochs = int(epochs_override if epochs_override is not None else cfg["epochs"])
    bs     = int(cfg["batch_size"])
    use_amp= bool(cfg["amp"])

    fb = 100
    if fisher_batches_by_preset and preset in fisher_batches_by_preset:
        fb = int(fisher_batches_by_preset[preset])

    set_seeds(seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = torch.nn.MSELoss()
    model   = make_model_fn(tfm)
    # Construye método compuesto: baseline (naive) + EWC opcional
    ewc_lam = (float(lam) if method == "ewc" else None)
    method_obj = build_method_with_optional_ewc(
        main_method="naive", model=model, loss_fn=loss_fn, device=device,
        fisher_batches=fb, ewc_lam=ewc_lam
    )

    out_tag = f"continual_{preset}_{method}" + (f"_lam_{lam:.0e}" if method=='ewc' else "") + f"_{encoder}_seed_{seed}"
    out_dir = (Path(out_root) if out_root else Path("outputs")) / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tcfg    = TrainConfig(epochs=epochs, batch_size=bs, lr=lr, amp=use_amp, seed=seed,
                          es_patience=(int(cfg["es_patience"]) if "es_patience" in cfg else None),
                          es_min_delta=float(cfg.get("es_min_delta", 0.0)))

    results = {}
    seen = []

    for i, t in enumerate(task_list, start=1):
        name = t["name"]
        if verbose:
            print(f"\n--- Tarea {i}/{len(task_list)}: {name} | preset={preset} | method={method} | "
                  f"λ={lam if lam is not None else '-'} | B={bs} T={T} AMP={use_amp} | enc={encoder} ---")

        # Encoder para el LOADER:
        # - si vamos a runtime encode ⇒ pedimos 4D (image) al loader
        # - si NO ⇒ pedimos 5D (rate/latency/raw) directamente
        # src/runner.py (dentro de run_continual, en el bucle)
        loader_encoder = "image" if (runtime_encode and encoder in {"rate","latency","raw"}) else encoder
        tr, va, te = make_loader_fn(
            task=t, batch_size=bs, encoder=loader_encoder, T=T, gain=gain, tfm=tfm, seed=seed,
        )

        encoder_for_loader = ("image" if (runtime_encode and encoder in {"rate","latency","raw"}) else encoder)
        tr, va, te = make_loader_fn(
            task=t, batch_size=bs, encoder=encoder_for_loader, T=T, gain=gain, tfm=tfm, seed=seed,
        )

        # Inspección del batch
        xb, yb = next(iter(tr))
        if verbose:
            print(f"  loader batch shape: {tuple(xb.shape)} | y: {tuple(yb.shape)}")

        # Codificación en GPU si el loader es 4D ("image")
        used_rt = False
        if runtime_encode and xb.ndim == 4:
            set_runtime_encode(mode=encoder, T=T, gain=gain, device=device)
            used_rt = True
            if verbose:
                print("  runtime encode: ON (GPU)")

        # Hook antes de la tarea (no-op para naive/EWC)
        method_obj.before_task(model, tr, va)

        # Entrena (penalty() se aplica dentro si el método lo implementa)
        _ = train_supervised(
            model, tr, va, loss_fn, tcfg,
            out_dir / f"task_{i}_{name}",
            method=method_obj
        )

        # Hook después de la tarea (para EWC: estima Fisher en val)
        method_obj.after_task(model, tr, va)

        # Eval tarea actual
        te_mae, te_mse = eval_loader(te, model, device)
        results[name] = {"test_mae": te_mae, "test_mse": te_mse}
        seen.append((name, te))

        # Reevaluación (olvido)
        for pname, p_loader in seen[:-1]:
            p_mae, p_mse = eval_loader(p_loader, model, device)
            results[pname][f"after_{name}_mae"] = p_mae
            results[pname][f"after_{name}_mse"] = p_mse

        if used_rt:
            set_runtime_encode(None)
            if verbose:
                print("  runtime encode: OFF")

    (out_dir/"continual_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return out_dir, results

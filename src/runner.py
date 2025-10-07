# src/runner.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import sys, json, torch
from typing import Dict, Any

from src.training import TrainConfig, train_supervised, set_encode_runtime
from src.methods.registry import build_method
from src.eval import eval_loader
from src.telemetry import carbon_tracker_ctx, log_telemetry_event, system_snapshot
import os, time as _time

# Asegura raíz en sys.path (igual que tenías)
ROOT = Path.cwd().parents[0] if (Path.cwd().name == "notebooks") else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def run_continual(
    task_list: list[dict],
    make_loader_fn,
    make_model_fn,
    tfm,
    cfg: Dict[str, Any],           # ← preset YA cargado y merged
    preset_name: str,              # ← "fast" | "std" | "accurate" (solo para naming)
    out_root: Path | str | None = None,
    verbose: bool = True,
):
    # -----------------------------
    # 1) Leer config del preset
    # -----------------------------
    data = cfg["data"]; optim = cfg["optim"]; cont = cfg["continual"]; naming = cfg.get("naming", {})

    encoder  = str(data["encoder"])
    T        = int(data["T"])
    gain     = float(data["gain"])
    seed     = int(data.get("seed", 42))

    epochs   = int(optim["epochs"])
    bs       = int(optim["batch_size"])
    use_amp  = bool(optim["amp"] and torch.cuda.is_available())
    lr       = float(optim["lr"])

    # -----------------------------
    # 2) Kwargs comunes del DataLoader (coherentes con preset)
    # -----------------------------
    dl_kwargs = dict(
        num_workers         = int(data["num_workers"]),
        pin_memory          = bool(data["pin_memory"]),
        persistent_workers  = bool(data["persistent_workers"]),
        prefetch_factor     = data["prefetch_factor"],
    )

    # augment (si existe)
    from src.datasets import AugmentConfig
    if data.get("aug_train"):
        dl_kwargs["aug_train"] = AugmentConfig(**data["aug_train"])

    # balanceo online (si se activa en el preset)
    if data.get("balance_online", False):
        dl_kwargs["balance_train"] = True
        # Hereda de prep.bins si balance_bins es None
        bb = data.get("balance_bins", None)
        if bb is None:
            bb = int(cfg.get("prep", {}).get("bins", 21))
        dl_kwargs["balance_bins"] = int(bb)
        dl_kwargs["balance_smooth_eps"] = float(data.get("balance_smooth_eps", 1e-3))

    # -----------------------------
    # 3) Modelo y pérdidas
    # -----------------------------
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = torch.nn.MSELoss()
    model   = make_model_fn(tfm)

    def _model_label(model, tfm) -> str:
        cls = getattr(model, "__class__", type(model)).__name__
        h, w = getattr(tfm, "h", "?"), getattr(tfm, "w", "?")
        ch   = "rgb" if not getattr(tfm, "to_gray", True) else "gray"
        return f"{cls}_{h}x{w}_{ch}"

    model_lbl = _model_label(model, tfm)

    # -----------------------------
    # 4) Construir el método continual
    #    (inyectamos T si el método lo usa)
    # -----------------------------
    method = cont["method"].lower()
    method_kwargs = dict(cont.get("params", {}))

    # ▲ INYECTA T por defecto (AS-SNN lo usa para normalizar actividad).
    #   No afecta a EWC/Rehearsal porque su rama en registry no pasa **kwargs.
    method_kwargs.setdefault("T", T)

    method_obj = build_method(method, model, loss_fn=loss_fn, device=device, **method_kwargs)

    # === NUEVO (genérico): inyecta TODO lo que haya en logging.<método> como atributos ===
    log_root    = cfg.get("logging", {}) or {}
    log_section = log_root.get(method.lower(), {})

    # fallback opcional si el nombre es compuesto (p.ej., "rehearsal+ewc"):
    if not log_section and "+" in method:
        log_section = log_root.get(method.split("+")[0].lower(), {})

    # aplica todas las claves (p.ej., inner_verbose, inner_every, fisher_verbose, etc.)
    for k, v in (log_section or {}).items():
        try:
            setattr(method_obj, k, v)
        except Exception:
            pass

    tag = method_obj.name
    if ("ewc" in tag) and ("lam" in method_kwargs):
        tag = f"{tag}_lam_{float(method_kwargs['lam']):.0e}"
    tag_extra = naming.get("tag", "")
    if tag_extra:
        tag = f"{tag}_{tag_extra}"

    out_tag = f"continual_{preset_name}_{tag}_{encoder}_model-{model_lbl}_seed_{seed}"
    out_dir = (Path(out_root) if out_root else Path("outputs")) / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 5) Entrenamiento por tareas
    # -----------------------------
    # Early Stopping desde el preset (si está configurado)
    es_pat = optim.get("es_patience", None)
    es_md  = optim.get("es_min_delta", None)
    es_pat = int(es_pat) if es_pat is not None else None
    es_md  = float(es_md) if es_md is not None else 0.0

    tcfg = TrainConfig(
        epochs=epochs,
        batch_size=bs,
        lr=lr,
        amp=use_amp,
        seed=seed,
        es_patience=es_pat,
        es_min_delta=es_md,
    )
    results, seen = {}, []

    for i, t in enumerate(task_list, start=1):
        name = t["name"]
        if verbose:
            print(
                f"\n--- Tarea {i}/{len(task_list)}: {name} | preset={preset_name} | method={method_obj.name} "
                f"| B={bs} T={T} AMP={use_amp} | enc={encoder} ---"
            )

        # DataLoaders (H5 ó CSV+runtime encode, según factory)
        tr, va, te = make_loader_fn(
            task=t, batch_size=bs, encoder=encoder, T=T, gain=gain, tfm=tfm, seed=seed, **dl_kwargs
        )

        # Si el loader entrega 4D (C,H,W), activar runtime encode en GPU
        xb_sample, _ = next(iter(tr))
        used_rt = False
        if xb_sample.ndim == 4:
            set_encode_runtime(mode=encoder, T=T, gain=gain, device=device)
            used_rt = True
            if verbose: print("  runtime encode: ON (GPU)")

        # Si el método (o un composite) propone envolver el loader (p.ej. Rehearsal), respétalo
        if hasattr(method_obj, "prepare_train_loader"):
            tr = method_obj.prepare_train_loader(tr)

        # Hooks del método antes/después de cada tarea
        method_obj.before_task(model, tr, va)
        _ = train_supervised(model, tr, va, loss_fn, tcfg, out_dir / f"task_{i}_{name}", method=method_obj)
        method_obj.after_task(model, tr, va)

        # Eval en test de la tarea actual
        te_mae, te_mse = eval_loader(te, model, device)
        results[name] = {"test_mae": te_mae, "test_mse": te_mse}
        seen.append((name, te))

        # Eval de olvido: re-evalúa test loaders de tareas previas
        for pname, p_loader in seen[:-1]:
            p_mae, p_mse = eval_loader(p_loader, model, device)
            results[pname][f"after_{name}_mae"] = p_mae
            results[pname][f"after_{name}_mse"] = p_mse

        if used_rt:
            set_encode_runtime(None)
            if verbose: print("  runtime encode: OFF")

    # Guardar resultados
    (out_dir / "continual_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ▲ Limpia ganchos si el método los registró (AS-SNN expone detach()).
    if hasattr(method_obj, "detach"):
        try:
            method_obj.detach()
        except Exception:
            pass

    return out_dir, results

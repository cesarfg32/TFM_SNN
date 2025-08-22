from __future__ import annotations
from pathlib import Path
import sys, json, torch
from typing import Optional, Dict, Any

from src.utils import load_preset, set_seeds
from src.training import TrainConfig, train_supervised, set_runtime_encode
from src.methods.registry import build_method
from src.eval import eval_loader

ROOT = Path.cwd().parents[0] if (Path.cwd().name == "notebooks") else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def run_continual(
    task_list: list[dict],
    make_loader_fn,
    make_model_fn,
    tfm,
    preset: str,
    method: str,                 # "naive" | "ewc" | "rehearsal" | "rehearsal+ewc" | ...
    seed: int,
    encoder: str,
    epochs_override: Optional[int] = None,
    runtime_encode: bool = True,
    out_root: Path | str | None = None,
    verbose: bool = True,
    *,
    method_kwargs: Optional[Dict[str, Any]] = None,  # <- único sitio para hiperparámetros
):
    cfg = load_preset(ROOT / "configs" / "presets.yaml", preset)
    T, gain, lr = int(cfg["T"]), float(cfg["gain"]), float(cfg["lr"])
    epochs, bs, use_amp = int(epochs_override or cfg["epochs"]), int(cfg["batch_size"]), bool(cfg["amp"])

    set_seeds(seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = torch.nn.MSELoss()
    model   = make_model_fn(tfm)

    def _model_label(model, tfm) -> str:
        # p. ej. "PilotNetSNN_66x200_gray" o "PilotNetANN_66x200_gray"
        cls = getattr(model, "__class__", type(model)).__name__
        h = getattr(tfm, "h", "?")
        w = getattr(tfm, "w", "?")
        ch = "rgb" if not getattr(tfm, "to_gray", True) else "gray"
        return f"{cls}_{h}x{w}_{ch}"

    model_lbl = _model_label(model, tfm)

    method_l = method.lower()
    method_kwargs = (method_kwargs or {}).copy()

    # Un único builder: soporta puro o composite "+ewc"
    method_obj = build_method(
        method_l, model,
        loss_fn=loss_fn, device=device,
        **method_kwargs,
    )
    tag = method_obj.name  # "naive" | "ewc" | "rehearsal" | "rehearsal+ewc"

    # Si el método incluye EWC y pasaste 'lam', añádelo al tag
    if ("ewc" in tag) and ("lam" in method_kwargs):
        tag = f"{tag}_lam_{float(method_kwargs['lam']):.0e}"


    out_tag = f"continual_{preset}_{tag}_{encoder}_model-{model_lbl}_seed_{seed}"
    out_dir = (Path(out_root) if out_root else Path("outputs")) / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tcfg = TrainConfig(epochs=epochs, batch_size=bs, lr=lr, amp=use_amp, seed=seed)
    results, seen = {}, []

    for i, t in enumerate(task_list, start=1):
        name = t["name"]
        if verbose:
            print(f"\n--- Tarea {i}/{len(task_list)}: {name} | preset={preset} | method={method_obj.name} "
                  f"| B={bs} T={T} AMP={use_amp} | enc={encoder} ---")

        tr, va, te = make_loader_fn(task=t, batch_size=bs, encoder=encoder, T=T, gain=gain, tfm=tfm, seed=seed)

        # Detecta si el loader devuelve 4D para activar runtime encode ANTES de que el método lo envuelva
        xb_sample, _ = next(iter(tr))
        used_rt = False
        if runtime_encode and xb_sample.ndim == 4:
            set_runtime_encode(mode=encoder, T=T, gain=gain, device=device)
            used_rt = True
            if verbose: print("  runtime encode: ON (GPU)")

        # Deja al método envolver el train_loader (rehearsal, etc.)
        if hasattr(method_obj, "prepare_train_loader"):
            tr = method_obj.prepare_train_loader(tr)

        method_obj.before_task(model, tr, va)
        _ = train_supervised(
            model, tr, va, loss_fn, tcfg,
            out_dir / f"task_{i}_{name}",
            method=method_obj
        )
        method_obj.after_task(model, tr, va)

        te_mae, te_mse = eval_loader(te, model, device)
        results[name] = {"test_mae": te_mae, "test_mse": te_mse}
        seen.append((name, te))

        for pname, p_loader in seen[:-1]:
            p_mae, p_mse = eval_loader(p_loader, model, device)
            results[pname][f"after_{name}_mae"] = p_mae
            results[pname][f"after_{name}_mse"] = p_mse

        if used_rt:
            set_runtime_encode(None)
            if verbose: print("  runtime encode: OFF")

    (out_dir / "continual_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return out_dir, results

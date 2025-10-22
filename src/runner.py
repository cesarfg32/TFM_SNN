# src/runner.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import sys, json, torch, math
from typing import Dict, Any, Tuple, List

from src.training import TrainConfig, train_supervised, set_encode_runtime
from src.methods.registry import build_method
from src.eval import eval_loader

# Telemetría
from src.telemetry import (
    carbon_tracker_ctx,
    log_telemetry_event,
    system_snapshot,
    read_emissions_kg,
)
from contextlib import nullcontext
import os, time as _time, logging

# NEW: para detectar versión de codecarbon y loguearla
try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
except Exception:  # safety en entornos antiguos
    _pkg_version = None
    class PackageNotFoundError(Exception): ...
# ---------------------------------------------------

# Asegura raíz en sys.path (igual que tenías)
ROOT = Path.cwd().parents[0] if (Path.cwd().name == "notebooks") else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# -----------------------------
# utilidades locales para métricas/archivos
# -----------------------------
def _model_label(model, tfm) -> str:
    cls = getattr(model, "__class__", type(model)).__name__
    h, w = getattr(tfm, "h", "?"), getattr(tfm, "w", "?")
    ch = "rgb" if not getattr(tfm, "to_gray", True) else "gray"
    return f"{cls}_{h}x{w}_{ch}"

def _safe_write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if isinstance(payload, (dict, list)):
            json.dump(payload, f, indent=2, ensure_ascii=False)
        else:
            f.write(str(payload))

def _build_eval_matrix(results: Dict[str, Dict[str, float]]) -> Tuple[List[str], List[List[float]]]:
    """
    Devuelve (task_names, M) donde M[i][j] = MAE de la tarea i evaluada al terminar la tarea j.
    Diagonal M[i][i] = test_mae tras entrenar la tarea i.
    Última columna M[i][-1] = MAE final en la tarea i tras la última tarea (útil para olvido).
    """
    names = list(results.keys())
    n = len(names)
    mat = [[math.nan for _ in range(n)] for _ in range(n)]
    for i, ti in enumerate(names):
        # valor al completar su propia tarea
        mat[i][i] = float(results[ti].get("test_mae", math.nan))
        # valores "after_*"
        for j, tj in enumerate(names):
            if j < i:
                continue  # no existe evaluación "futura" antes de entrenar esa tarea
            if j == i:
                continue
            key = f"after_{tj}_mae"
            if key in results[ti]:
                mat[i][j] = float(results[ti][key])
    return names, mat

def _compute_forgetting(names: List[str], mat: List[List[float]]) -> Dict[str, Dict[str, float]]:
    """
    Olvido para MAE (menor es mejor):
      F_abs(i) = M[i][last] - min_{k<=last} M[i][k]    (aumento de error respecto al mejor histórico)
      F_rel(i) = F_abs(i) / min_{k<=last} M[i][k]
    """
    n = len(names)
    out: Dict[str, Dict[str, float]] = {}
    for i in range(n):
        row = [v for v in mat[i] if not math.isnan(v)]
        if not row:
            out[names[i]] = {"forget_abs": math.nan, "forget_rel": math.nan}
            continue
        best = min(row)          # mejor MAE alcanzado en cualquier punto
        final = row[-1]          # MAE tras última tarea
        f_abs = float(final - best)
        f_rel = float(f_abs / best) if best > 0 else math.nan
        out[names[i]] = {"forget_abs": f_abs, "forget_rel": f_rel, "best_mae": best, "final_mae": final}
    return out

def _torch_cuda_mem_peak_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    try:
        return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    except Exception:
        return None

def _torch_num_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def run_continual(
    task_list: list[dict],
    make_loader_fn,
    make_model_fn,
    tfm,
    cfg: Dict[str, Any],             # ← preset YA cargado y merged
    preset_name: str,               # ← "fast" | "std" | "accurate" (solo para naming)
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
    model_lbl = _model_label(model, tfm)

    # -----------------------------
    # 4) Construir el método continual (inyecta T si procede)
    # -----------------------------
    method = cont["method"].lower()
    method_kwargs = dict(cont.get("params", {}))
    method_kwargs.setdefault("T", T)  # útil para SA-SNN

    method_obj = build_method(method, model, loss_fn=loss_fn, device=device, **method_kwargs)

    # === Logging por preset: logging.<método> ===
    log_root = (cfg.get("logging", {}) or {})
    def _apply_log_section(obj, key: str):
        sect = log_root.get(key, {}) or {}
        for k, v in sect.items():
            try:
                setattr(obj, k, v)
            except Exception:
                pass

    # aplica al método principal
    _apply_log_section(method_obj, method.lower())
    # si es composite y expone submétodos, intenta aplicar por nombre base de cada uno
    if hasattr(method_obj, "methods"):
        try:
            for sub in getattr(method_obj, "methods", []):
                base = str(getattr(sub, "name", "")).lower().split("+")[0].split("_")[0]
                if base:
                    _apply_log_section(sub, base)
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
    # Telemetría (activada por preset: logging.telemetry)
    # -----------------------------
    tele_cfg   = (cfg.get("logging", {}) or {}).get("telemetry", {}) or {}
    use_cc     = bool(tele_cfg.get("codecarbon", False))
    cc_offline = bool(tele_cfg.get("offline", True))
    cc_country = tele_cfg.get("country_iso_code") or os.getenv("CODECARBON_COUNTRY_ISO_CODE", None)
    cc_period  = int(tele_cfg.get("measure_power_secs", 15))
    cc_loglvl  = str(tele_cfg.get("log_level", "warning")).lower()

    try:
        logging.getLogger("codecarbon").setLevel(getattr(logging, cc_loglvl.upper(), logging.WARNING))
    except Exception:
        pass

    # NEW (1/2): log de configuración efectiva de CodeCarbon, incluida la versión instalada
    cc_ver = None
    if _pkg_version is not None:
        try:
            cc_ver = _pkg_version("codecarbon")
        except PackageNotFoundError:
            cc_ver = None
        except Exception:
            cc_ver = None

    log_telemetry_event(out_dir, {
        "event": "run_start",
        "preset": preset_name, "method": method_obj.name,
        "encoder": encoder, "T": T, "batch_size": bs, "amp": use_amp,
        "out_dir": str(out_dir),
        **system_snapshot(),
        "params_total": _torch_num_params(model),
        "lr": lr, "epochs": epochs,
    })

    log_telemetry_event(out_dir, {
        "event": "cc_config",
        "codecarbon_enabled": use_cc,
        "codecarbon_version": cc_ver,
        "cc_offline": cc_offline,
        "cc_country": cc_country,
        "cc_period": cc_period,
        "cc_loglvl": cc_loglvl,
    })
    # ---------------------------------------------------

    t_start = _time.time()
    cc_context = carbon_tracker_ctx(
        out_dir,
        project_name=out_tag,
        offline=cc_offline,
        country_iso_code=cc_country,
        measure_power_secs=cc_period,
        log_level=cc_loglvl,
    ) if use_cc else nullcontext()

    # -----------------------------
    # Estructuras para reportes
    # -----------------------------
    results: Dict[str, Dict[str, float]] = {}
    seen: List[Tuple[str, Any]] = []
    perf_rows: List[Dict[str, Any]] = []  # tiempos/mem por tarea

    # NEW (2/2): capturar el tracker y emissions explícitamente tras el with
    tracker_ref = None
    emissions_kg = None
    # ---------------------------------------------------

    with cc_context as _cc:
        # Mantén una referencia al tracker para leer final_emissions tras el with
        tracker_ref = _cc

        # -----------------------------
        # 5) Entrenamiento por tareas
        # -----------------------------
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

            # Si el método (o composite) propone envolver el loader (p.ej. Rehearsal), respétalo
            if hasattr(method_obj, "prepare_train_loader"):
                tr = method_obj.prepare_train_loader(tr)

            # Hooks del método antes/después de cada tarea
            method_obj.before_task(model, tr, va)

            # Métricas de eficiencia por tarea
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t0 = _time.time()
            _ = train_supervised(model, tr, va, loss_fn, tcfg, out_dir / f"task_{i}_{name}", method=method_obj)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = _time.time()
            mem_peak = _torch_cuda_mem_peak_mb()

            perf_rows.append({
                "task_idx": i, "task_name": name,
                "train_time_sec": t1 - t0,
                "cuda_mem_peak_mb": mem_peak,
                "batch_size": bs, "amp": use_amp, "encoder": encoder, "T": T,
            })

            # Evita calcular Fisher en la última tarea: ahorra MUCHO tiempo
            if i < len(task_list) and hasattr(method_obj, "after_task"):
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

            # Guardar estado opcional del método (si lo expone) para trazabilidad
            try:
                snap = None
                if hasattr(method_obj, "get_state"):
                    snap = method_obj.get_state()  # SA-SNN
                elif hasattr(method_obj, "get_activity_state"):
                    snap = method_obj.get_activity_state()  # AS-SNN
                if snap is not None:
                    _safe_write(out_dir / f"method_state_task_{i}_{name}.json", snap)
            except Exception:
                pass

            if used_rt:
                set_encode_runtime(None)
                if verbose: print("  runtime encode: OFF")

        # Asegura flush de kernels antes de cerrar el tracker
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # -----------------------------
    # Persistencia: resultados + matrices + eficiencia
    # -----------------------------
    # 1) resultados crudos por tarea
    _safe_write(out_dir / "continual_results.json", results)

    # 2) matriz de evaluación (MAE) y olvido (abs/rel) —> CSV + JSON
    names, mat = _build_eval_matrix(results)
    eval_mat = {
        "tasks": names,
        "mae_matrix": mat,  # fila i: tarea i; col j: MAE tras tarea j
    }
    _safe_write(out_dir / "eval_matrix.json", eval_mat)
    # CSV legible
    csv_lines = []
    header = ["task"] + [f"after_{n}" for n in names]
    csv_lines.append(",".join(header))
    for i, ti in enumerate(names):
        row = [ti] + [("" if math.isnan(v) else f"{v:.6f}") for v in mat[i]]
        csv_lines.append(",".join(row))
    (out_dir / "eval_matrix.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    # Olvido por tarea
    forgetting = _compute_forgetting(names, mat)
    _safe_write(out_dir / "forgetting.json", forgetting)

    # 3) tiempos/mem por tarea
    _safe_write(out_dir / "per_task_perf.json", perf_rows)

    # 4) resumen de eficiencia del run
    elapsed = _time.time() - t_start

    # NEW (2/2): recoge final_emissions del tracker si no lo tienes aún
    if emissions_kg is None and tracker_ref is not None:
        try:
            emissions_kg = getattr(tracker_ref, "final_emissions", None)
        except Exception:
            emissions_kg = None

    # intenta leer emissions.csv por robustez (versiones CC antiguas / fallos de stop())
    if emissions_kg is None:
        emissions_kg = read_emissions_kg(out_dir)

    summary = {
        "elapsed_sec": elapsed,
        "emissions_kg": emissions_kg,
        "params_total": _torch_num_params(model),
        "preset": preset_name,
        "method": method_obj.name,
        "encoder": encoder,
        "T": T,
        "batch_size": bs,
        "amp": use_amp,
        "seed": seed,
        "model": model_lbl,
    }
    _safe_write(out_dir / "efficiency_summary.json", summary)

    # ▲ Limpia ganchos si el método los registró (AS/SA-SNN exponen detach()).
    if hasattr(method_obj, "detach"):
        try:
            method_obj.detach()
        except Exception:
            pass

    log_telemetry_event(out_dir, {
        "event": "run_end",
        "elapsed_sec": elapsed,
        "emissions_kg": emissions_kg,
    })

    return out_dir, results

# src/runner.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import sys, json, torch, math, os, time as _time, logging, platform
from typing import Dict, Any, Tuple, List, Optional

# NEW: multiprocessing tweaks para evitar /dev/shm bus errors
import torch.multiprocessing as mp
try:
    mp.set_sharing_strategy("file_system")
except Exception:
    pass

def _get_mp_ctx():
    # En Linux/WSL suele ir mejor 'forkserver' que 'fork' con PyTorch + CUDA.
    for name in ("forkserver", "spawn"):
        try:
            return mp.get_context(name)
        except Exception:
            continue
    return None

from torch.utils.data import DataLoader
from src.training import TrainConfig, train_supervised, set_encode_runtime
from src.methods.registry import build_method
from src.eval import eval_loader
from src.telemetry import (
    carbon_tracker_ctx,
    log_telemetry_event,
    system_snapshot,
    read_emissions_kg,
)
from contextlib import nullcontext

# ---------------------------------------------------
# Asegura raíz en sys.path
ROOT = Path.cwd().parents[0] if (Path.cwd().name == "notebooks") else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# -----------------------------
# utilidades locales
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
    names = list(results.keys())
    n = len(names)
    mat = [[math.nan for _ in range(n)] for _ in range(n)]
    for i, ti in enumerate(names):
        mat[i][i] = float(results[ti].get("test_mae", math.nan))
        for j, tj in enumerate(names):
            if j <= i:
                continue
            key = f"after_{tj}_mae"
            if key in results[ti]:
                mat[i][j] = float(results[ti][key])
    return names, mat

def _compute_forgetting(names: List[str], mat: List[List[float]]) -> Dict[str, Dict[str, float]]:
    n = len(names)
    out: Dict[str, Dict[str, float]] = {}
    for i in range(n):
        row = [v for v in mat[i] if not math.isnan(v)]
        if not row:
            out[names[i]] = {"forget_abs": math.nan, "forget_rel": math.nan}
            continue
        best = min(row)
        final = row[-1]
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

# --- CSV history: ahora incluye columnas extra si existen ---
def _to_csv_lines_from_history(history: dict) -> str:
    cols = ["epoch", "train_loss", "val_loss", "val_mae", "val_mse", "train_mae", "train_mse"]
    L = max([
        len(history.get("train_loss", [])),
        len(history.get("val_loss",   [])),
        len(history.get("val_mae",    [])),
        len(history.get("val_mse",    [])),
        len(history.get("train_mae",  [])),
        len(history.get("train_mse",  [])),
        0
    ])
    lines = [",".join(cols)]
    for e in range(L):
        row = [str(e+1)]
        for k in cols[1:]:
            seq = history.get(k, [])
            v = seq[e] if e < len(seq) else None
            row.append("" if v is None else f"{float(v):.8f}")
        lines.append(",".join(row))
    return "\n".join(lines)

def _pick_c2_name(task_names: List[str]) -> str | None:
    import re
    for n in task_names:
        if re.search(r"(c2|circuito2|track2|tarea2|task2)", n, re.I):
            return n
    return task_names[1] if len(task_names) >= 2 else (task_names[-1] if task_names else None)

# -----------------------------
# NUEVO: clonar loader con num_workers=0 para anclas (SCA)
# -----------------------------
def _to_single_worker_loader(loader: DataLoader) -> DataLoader:
    """Clona un DataLoader para uso *solo* en cálculo de anclas (SCA)."""
    try:
        return DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=getattr(loader, "sampler", None),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=1,
            drop_last=False,
            collate_fn=loader.collate_fn,
            timeout=0,
        )
    except Exception:
        return loader

# -----------------------------
# NUEVO: manifest por tarea (para plots robustos)
# -----------------------------
def _write_task_manifest(task_dir: Path, meta: Dict[str, Any], history: Dict[str, Any]) -> None:
    payload = {
        "meta": meta,
        "history": {
            "train_loss": history.get("train_loss", []),
            "val_loss":   history.get("val_loss",   []),
            "val_mae":    history.get("val_mae",    []),
            "val_mse":    history.get("val_mse",    []),
            "train_mae":  history.get("train_mae",  []),
            "train_mse":  history.get("train_mse",  []),
        },
    }
    _safe_write(Path(task_dir) / "manifest.json", payload)

# ================================================
def run_continual(
    task_list: list[dict],
    make_loader_fn,
    make_model_fn,
    tfm,
    cfg: Dict[str, Any],
    preset_name: str,
    out_root: Path | str | None = None,
    verbose: bool = True,
):
    data = cfg["data"]; optim = cfg["optim"]; cont = cfg["continual"]; naming = cfg.get("naming", {})

    encoder  = str(data["encoder"])
    T        = int(data["T"])
    gain     = float(data["gain"])
    seed     = int(data.get("seed", 42))

    use_offline = bool(data.get("use_offline_spikes", False))

    epochs   = int(optim["epochs"])
    bs       = int(optim["batch_size"])
    use_amp  = bool(optim["amp"] and torch.cuda.is_available())
    lr       = float(optim["lr"])

    # ------------- DataLoader kwargs (mitigaciones shm) -------------
    dl_kwargs = dict(
        num_workers         = int(data["num_workers"]),
        pin_memory          = bool(data["pin_memory"]),
        persistent_workers  = bool(data["persistent_workers"]),
        prefetch_factor     = data["prefetch_factor"],
        drop_last           = True,
        pin_memory_device   = data.get("pin_memory_device", "cuda"),
        timeout             = 0,
    )
    mp_ctx = _get_mp_ctx()
    if mp_ctx is not None and dl_kwargs["num_workers"] > 0:
        dl_kwargs["multiprocessing_context"] = mp_ctx

    from src.datasets import AugmentConfig
    if data.get("aug_train"):
        dl_kwargs["aug_train"] = AugmentConfig(**data["aug_train"])
    if data.get("balance_online", False):
        dl_kwargs["balance_train"] = True
        bb = data.get("balance_bins", None)
        if bb is None:
            bb = int(cfg.get("prep", {}).get("bins", 21))
        dl_kwargs["balance_bins"] = int(bb)
        dl_kwargs["balance_smooth_eps"] = float(data.get("balance_smooth_eps", 1e-3))

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = torch.nn.MSELoss()
    model   = make_model_fn(tfm)
    model_lbl = _model_label(model, tfm)

    method = cont["method"].lower()
    method_kwargs = dict(cont.get("params", {}))
    method_kwargs.setdefault("T", T)
    method_obj = build_method(method, model, loss_fn=loss_fn, device=device, **method_kwargs)

    log_root = (cfg.get("logging", {}) or {})
    def _apply_log_section(obj, key: str):
        sect = log_root.get(key, {}) or {}
        for k, v in sect.items():
            try:
                setattr(obj, k, v)
            except Exception:
                pass
    _apply_log_section(method_obj, method.lower())
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

    # -------- Telemetría --------
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
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
    try:
        cc_ver = _pkg_version("codecarbon")
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
    t_start = _time.time()
    cc_context = carbon_tracker_ctx(
        out_dir,
        project_name=out_tag,
        offline=cc_offline,
        country_iso_code=cc_country,
        measure_power_secs=cc_period,
        log_level=cc_loglvl,
    ) if use_cc else nullcontext()

    results: Dict[str, Dict[str, float]] = {}
    seen: List[Tuple[str, Any]] = []
    perf_rows: List[Dict[str, Any]] = []
    tracker_ref = None
    emissions_kg = None

    with cc_context as _cc:
        tracker_ref = _cc

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

            tr, va, te = make_loader_fn(
                task=t, batch_size=bs, encoder=encoder, T=T, gain=gain, tfm=tfm, seed=seed, **dl_kwargs
            )

            # Muestra para decidir encode runtime (solo si NO hay offline spikes)
            xb_sample, _ = next(iter(tr))
            used_rt = False
            if (not use_offline) and xb_sample.ndim == 4:
                set_encode_runtime(mode=encoder, T=T, gain=gain, device=device)
                used_rt = True
                if verbose: print("  runtime encode: ON (GPU)")

            # --- SOLO SCA: loader "anclas" con num_workers=0 para evitar /dev/shm ---
            is_sca = "sca-snn" in method_obj.name.lower() or "sca_snn" in method_obj.name.lower()
            tr_anchor = _to_single_worker_loader(tr) if is_sca else tr

            if hasattr(method_obj, "prepare_train_loader"):
                tr = method_obj.prepare_train_loader(tr)

            # --- NUEVO: evento de inicio de tarea (telemetría ligera)
            log_telemetry_event(out_dir, {"event": "task_start", "task_idx": i, "task_name": name})

            # Antes de la tarea con loader anclas "single-worker" SOLO si SCA
            method_obj.before_task(model, tr_anchor, va)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t0 = _time.time()

            try:
                history = train_supervised(
                    model, tr, va, loss_fn, tcfg, out_dir / f"task_{i}_{name}", method=method_obj
                )
            except (OSError, RuntimeError) as e:
                msg = str(e).lower()
                # Fallback solo si detectamos 'bus error' típico de /dev/shm en WSL
                if "bus error" in msg or "unexpected bus error" in msg:
                    print("[WARN] Bus error en DataLoader durante training. Reintento con num_workers=0…")
                    tr_safe = _to_single_worker_loader(tr)
                    history = train_supervised(
                        model, tr_safe, va, loss_fn, tcfg, out_dir / f"task_{i}_{name}", method=method_obj
                    )
                else:
                    raise

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = _time.time()
            mem_peak = _torch_cuda_mem_peak_mb()

            # CSV de curvas (historia por epoch)
            try:
                csv_curve = _to_csv_lines_from_history(history or {})
                (out_dir / f"task_{i}_{name}" / "loss_curves.csv").write_text(csv_curve, encoding="utf-8")
            except Exception:
                pass

            # Manifest por tarea (meta + history) para plots robustos
            try:
                meta = {
                    "task_idx": i,
                    "task_name": name,
                    "preset": preset_name,
                    "method": method_obj.name,
                    "encoder": encoder,
                    "T": T,
                    "gain": gain,
                    "batch_size": bs,
                    "epochs": epochs,
                    "lr": lr,
                    "amp": use_amp,
                    "seed": seed,
                    "model": _model_label(model, tfm),
                }
                _write_task_manifest(out_dir / f"task_{i}_{name}", meta, history or {})
            except Exception:
                pass

            n_epochs_done = len((history or {}).get("val_loss", []))
            perf_rows.append({
                "task_idx": i, "task_name": name,
                "train_time_sec": t1 - t0,
                "cuda_mem_peak_mb": mem_peak,
                "batch_size": bs, "amp": use_amp, "encoder": encoder, "T": T,
                "epochs_done": n_epochs_done,
            })

            # Después de la tarea: idem (solo SCA)
            if i < len(task_list) and hasattr(method_obj, "after_task"):
                method_obj.after_task(model, tr_anchor, va)

            te_mae, te_mse = eval_loader(te, model, device)
            results[name] = {"test_mae": te_mae, "test_mse": te_mse}
            seen.append((name, te))
            for pname, p_loader in seen[:-1]:
                p_mae, p_mse = eval_loader(p_loader, model, device)
                results[pname][f"after_{name}_mae"] = p_mae
                results[pname][f"after_{name}_mse"] = p_mse

            try:
                snap = None
                if hasattr(method_obj, "get_state"):
                    snap = method_obj.get_state()
                elif hasattr(method_obj, "get_activity_state"):
                    snap = method_obj.get_activity_state()
                if snap is not None:
                    _safe_write(out_dir / f"method_state_task_{i}_{name}.json", snap)
            except Exception:
                pass

            if used_rt:
                set_encode_runtime(None)
                if verbose: print("  runtime encode: OFF")

            # --- NUEVO: evento de fin de tarea (telemetría ligera)
            log_telemetry_event(out_dir, {"event": "task_end", "task_idx": i, "task_name": name})

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    _safe_write(out_dir / "continual_results.json", results)

    names, mat = _build_eval_matrix(results)
    eval_mat = {"tasks": names, "mae_matrix": mat}
    _safe_write(out_dir / "eval_matrix.json", eval_mat)
    csv_lines = []
    header = ["task"] + [f"after_{n}" for n in names]
    csv_lines.append(",".join(header))
    for i, ti in enumerate(names):
        row = [ti] + [("" if math.isnan(v) else f"{v:.6f}") for v in mat[i]]
        csv_lines.append(",".join(row))
    (out_dir / "eval_matrix.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    forgetting = _compute_forgetting(names, mat)
    _safe_write(out_dir / "forgetting.json", forgetting)

    # NUEVO: volcar rendimiento por tarea (json + csv)
    if len(perf_rows) > 0:
        _safe_write(out_dir / "per_task_perf.json", perf_rows)
        # CSV
        perf_csv_h = ["task_idx","task_name","train_time_sec","cuda_mem_peak_mb","batch_size","amp","encoder","T","epochs_done"]
        lines = [",".join(perf_csv_h)]
        for r in perf_rows:
            lines.append(",".join([str(r.get(k,"")) for k in perf_csv_h]))
        (out_dir / "per_task_perf.csv").write_text("\n".join(lines), encoding="utf-8")

    elapsed = _time.time() - t_start
    emissions_kg = read_emissions_kg(out_dir) if emissions_kg is None else emissions_kg

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
        "model": _model_label(model, tfm),
    }
    _safe_write(out_dir / "efficiency_summary.json", summary)

    try:
        _safe_write(out_dir / "method_params.json", method_kwargs)
    except Exception:
        pass

    c2_name = _pick_c2_name(names) or (names[-1] if names else None)
    c2_final_mae = None
    if c2_name is not None:
        row_idx = names.index(c2_name)
        row_vals = [v for v in mat[row_idx] if not math.isnan(v)]
        if row_vals:
            c2_final_mae = float(row_vals[-1])

    avg_forget_rel = None
    try:
        vals = [v.get("forget_rel") for v in forgetting.values() if isinstance(v, dict) and "forget_rel" in v]
        vals = [float(x) for x in vals if (x is not None and not math.isnan(x))]
        if vals:
            avg_forget_rel = sum(vals)/len(vals)
    except Exception:
        pass

    run_row = {
        "exp": out_dir.name,
        "preset": preset_name,
        "method": method_obj.name,
        "encoder": encoder,
        "model": _model_label(model, tfm),
        "seed": seed,
        "c2_final_mae": c2_final_mae,
        "avg_forget_rel": avg_forget_rel,
        "emissions_kg": emissions_kg,
        "elapsed_sec": elapsed,
    }
    _safe_write(out_dir / "run_row.json", run_row)
    (out_dir / "run_row.csv").write_text(
        "exp,preset,method,encoder,model,seed,c2_final_mae,avg_forget_rel,emissions_kg,elapsed_sec\n"
        + ",".join([
            str(run_row["exp"]),
            str(run_row["preset"]),
            str(run_row["method"]),
            str(run_row["encoder"]),
            str(run_row["model"]),
            str(run_row["seed"]),
            "" if run_row["c2_final_mae"]   is None else f"{run_row['c2_final_mae']:.8f}",
            "" if run_row["avg_forget_rel"] is None else f"{run_row['avg_forget_rel']:.8f}",
            "" if run_row["emissions_kg"]   is None else f"{float(run_row['emissions_kg']):.6f}",
            f"{float(run_row['elapsed_sec']):.3f}" if run_row["elapsed_sec"] is not None else "",
        ]),
        encoding="utf-8"
    )

    if hasattr(method_obj, "detach"):
        try:
            method_obj.detach()
        except Exception:
            pass

    log_telemetry_event(out_dir, {"event": "run_end", "elapsed_sec": elapsed, "emissions_kg": emissions_kg})
    return out_dir, results

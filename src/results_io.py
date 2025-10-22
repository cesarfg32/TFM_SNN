# -*- coding: utf-8 -*-
"""
results_io.py
--------------
Utilidades para leer resultados de runs continual_* y construir una tabla
consolidada robusta a runs antiguos (sin eval_matrix.csv/json, sin CodeCarbon, etc).

Salida clave: build_results_table(outputs_root) -> pd.DataFrame
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import json, math, re
import pandas as pd
import numpy as np

# =========================
# Helpers de E/S robustos
# =========================
def _safe_json_load(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _lower_map(cols) -> Dict[str, str]:
    return {str(c).lower().strip(): str(c) for c in cols}

# -------------------------
# CodeCarbon / Telemetry
# -------------------------
def read_codecarbon_summary(exp_dir: Path) -> dict:
    """
    Devuelve resumen de CodeCarbon si existe emissions.csv:
      {'emissions_kg','duration_s','energy_kwh','cpu_kwh','gpu_kwh','ram_kwh','kg_per_hour'}
    Robusto a distintas versiones/encabezados.
    """
    csv_path = exp_dir / "emissions.csv"
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return {}
        cmap = _lower_map(df.columns)
        last = df.iloc[-1]

        emissions_kg = None
        for k in ("emissions", "emissions_kg", "emission", "co2e"):
            if k in cmap:
                emissions_kg = float(last[cmap[k]])
                break

        duration_s = None
        for k in ("duration_s", "duration", "total_duration_s"):
            if k in cmap:
                duration_s = float(last[cmap[k]])
                break

        energy_kwh = None
        if "energy_consumed" in cmap:
            energy_kwh = float(last[cmap["energy_consumed"]])
        else:
            cpu_kwh = float(last[cmap["cpu_energy"]]) if "cpu_energy" in cmap else np.nan
            gpu_kwh = float(last[cmap["gpu_energy"]]) if "gpu_energy" in cmap else np.nan
            ram_kwh = float(last[cmap["ram_energy"]]) if "ram_energy" in cmap else np.nan
            parts = [v for v in (cpu_kwh, gpu_kwh, ram_kwh) if pd.notna(v)]
            energy_kwh = float(np.sum(parts)) if parts else None

        cpu_kwh = float(last[cmap["cpu_energy"]]) if "cpu_energy" in cmap else None
        gpu_kwh = float(last[cmap["gpu_energy"]]) if "gpu_energy" in cmap else None
        ram_kwh = float(last[cmap["ram_energy"]]) if "ram_energy" in cmap else None

        out = {
            "emissions_kg": emissions_kg,
            "duration_s": duration_s,
            "energy_kwh": energy_kwh,
            "cpu_kwh": cpu_kwh,
            "gpu_kwh": gpu_kwh,
            "ram_kwh": ram_kwh,
        }
        if out.get("emissions_kg") is not None and out.get("duration_s"):
            dur_h = max(1e-9, out["duration_s"] / 3600.0)
            out["kg_per_hour"] = out["emissions_kg"] / dur_h
        return out
    except Exception:
        return {}

def read_telemetry_last(exp_dir: Path) -> dict:
    """
    Lee el último evento de telemetry.jsonl si existe.
    Devuelve {'telemetry_elapsed_sec', 'telemetry_emissions_kg'} si están.
    """
    f = exp_dir / "telemetry.jsonl"
    if not f.exists():
        return {}
    try:
        last = {}
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                last = json.loads(line)
        keep = {}
        if "elapsed_sec" in last:
            keep["telemetry_elapsed_sec"] = float(last["elapsed_sec"])
        if "emissions_kg" in last and last["emissions_kg"] is not None:
            keep["telemetry_emissions_kg"] = float(last["emissions_kg"])
        return keep
    except Exception:
        return {}

# ==========================================
# Reconstrucción de eval_matrix si no existe
# ==========================================
def rebuild_eval_matrix_from_results(run_dir: Path) -> Tuple[List[str], List[List[float]]]:
    res = _safe_json_load(run_dir / "continual_results.json")
    if not res:
        return [], []
    tasks = list(res.keys())  # orden de inserción = orden de tareas
    n = len(tasks)
    M = [[math.nan for _ in range(n)] for _ in range(n)]
    for i, ti in enumerate(tasks):
        M[i][i] = float(res[ti].get("test_mae", math.nan))
        for j, tj in enumerate(tasks):
            if j <= i:
                continue
            key = f"after_{tj}_mae"
            if key in res[ti]:
                try:
                    M[i][j] = float(res[ti][key])
                except Exception:
                    M[i][j] = math.nan
    return tasks, M

def ensure_eval_matrix_files(run_dir: Path) -> Tuple[List[str], List[List[float]]]:
    run_dir = Path(run_dir)
    csvp = run_dir / "eval_matrix.csv"
    jsp = run_dir / "eval_matrix.json"

    # 1) JSON
    if jsp.exists():
        data = _safe_json_load(jsp)
        tasks = data.get("tasks") or []
        M = data.get("mae_matrix") or []
        if tasks and M:
            return tasks, M

    # 2) CSV
    if csvp.exists():
        try:
            df = pd.read_csv(csvp)
            if "task" not in df.columns:
                raise ValueError("eval_matrix.csv sin columna 'task'")
            tasks = df["task"].tolist()
            cols = [c for c in df.columns if c != "task"]
            M = (
                df[cols]
                .replace("", np.nan)
                .apply(pd.to_numeric, errors="coerce")
                .values
                .tolist()
            )
            return tasks, M
        except Exception:
            pass

    # 3) Reconstruir desde continual_results.json
    tasks, M = rebuild_eval_matrix_from_results(run_dir)
    if tasks and M:
        try:
            jsp.write_text(
                json.dumps({"tasks": tasks, "mae_matrix": M}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
        try:
            header = ["task"] + [f"after_{t}" for t in tasks]
            rows = []
            for i, ti in enumerate(tasks):
                vals = []
                for j in range(len(tasks)):
                    v = M[i][j] if j < len(M[i]) else math.nan
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        vals.append("")
                    else:
                        try:
                            vals.append(f"{float(v):.6f}")
                        except Exception:
                            vals.append("")
                rows.append([ti] + vals)
            df = pd.DataFrame(rows, columns=header)
            df.to_csv(csvp, index=False)
        except Exception:
            pass
    return tasks, M

# ===============================
# Derivadas a partir de la matriz
# ===============================
def _row_best_final_forgetting(row: List[float]) -> Tuple[float, float, float, float]:
    vals = [v for v in row if isinstance(v, (int, float)) and not math.isnan(v)]
    if not vals:
        return (math.nan, math.nan, math.nan, math.nan)
    best = min(vals)
    final = row[-1]
    if final is None or math.isnan(final) or best is None or math.isnan(best):
        return (best, final, math.nan, math.nan)
    f_abs = final - best
    f_rel = (f_abs / best) if best > 0 else math.nan
    return (best, final, f_abs, f_rel)

# ===========================
# Fallback: parse de run_dir
# ===========================
_RUN_RE = re.compile(
    r"""
    ^continual_
    (?P<preset>[^_]+)_
    (?P<method>.+?)_
    (?P<encoder>[^_]+)
    _model\-(?P<model>[^_]+)
    .*?seed_(?P<seed>\d+)
    """,
    re.X | re.IGNORECASE,
)

def parse_from_run_name(run_name: str) -> dict:
    """
    Intenta extraer preset, method, encoder, model, seed del nombre de carpeta.
    Ejemplo: continual_std_sa-snn_k10_tau10_th1-2_p2000000_rate_model-PilotNetSNN_66x200_gray_seed_42
    """
    m = _RUN_RE.match(run_name)
    if not m:
        return {}
    out = m.groupdict()
    # normaliza tipos
    if "seed" in out:
        try:
            out["seed"] = int(out["seed"])
        except Exception:
            pass
    return out

def _read_first_task_manifest(run_dir: Path) -> dict:
    """
    Lee el primer task_*/(manifest|metrics).json y devuelve varios campos útiles:
    batch_size, lr, amp, seed (si estuvieran).
    """
    for td in sorted(run_dir.glob("task_*")):
        for cand in ("manifest.json", "metrics.json"):
            p = td / cand
            if p.exists():
                data = _safe_json_load(p)
                hist = data.get("history") or {}
                return {
                    "batch_size": data.get("batch_size"),
                    "lr": data.get("lr"),
                    "amp": data.get("amp"),
                    "seed": data.get("seed"),
                    "epochs": data.get("epochs"),
                    "has_history": bool(hist.get("train_loss") or hist.get("val_loss")),
                }
    return {}

# ===========================
# Tabla principal de resultados
# ===========================
def build_results_table(outputs_root: Path | str) -> pd.DataFrame:
    """
    Escanea outputs/continual_* y construye un DataFrame con:
      - metadatos del run (preset, method, encoder, modelo, T, B, amp, seed, params)
      - resumen de eficiencia (elapsed_sec, emisiones, kWh, kg/h, etc.)
      - métricas por tarea (final_mae, best_mae, forgetting abs/rel)
      - medias de olvido (avg_forget_abs, avg_forget_rel)
    Es robusto a runs antiguos sin eval_matrix ni CodeCarbon.
    """
    outputs_root = Path(outputs_root)
    rows: List[Dict[str, Any]] = []

    for run_dir in sorted(outputs_root.glob("continual_*")):
        row: Dict[str, Any] = {"run_dir": run_dir.name}

        # 1) Eficiencia (nuevo runner)
        eff = _safe_json_load(run_dir / "efficiency_summary.json")
        if eff:
            row.update({
                "preset": eff.get("preset"),
                "method": eff.get("method"),
                "encoder": eff.get("encoder"),
                "model": eff.get("model"),
                "seed": eff.get("seed"),
                "T": eff.get("T"),
                "B": eff.get("batch_size"),
                "amp": eff.get("amp"),
                "params": eff.get("params_total"),
                "elapsed_sec": eff.get("elapsed_sec"),
            })
            if eff.get("emissions_kg") is not None:
                row["emissions_kg"] = eff.get("emissions_kg")

        # 2) Fallback: parse desde el nombre si faltan campos clave
        if any(k not in row or pd.isna(row.get(k)) for k in ("preset","method","encoder","model","seed")):
            row.update({k: v for k, v in parse_from_run_name(run_dir.name).items() if v is not None})

        # 3) CodeCarbon + Telemetry (si existen)
        cc = read_codecarbon_summary(run_dir)
        tel = read_telemetry_last(run_dir)
        row.update(cc)
        row.update(tel)

        # 4) Completar B/amp/seed desde el primer task manifest si faltan
        if any(k not in row or row.get(k) is None for k in ("B","amp","seed")):
            mf = _read_first_task_manifest(run_dir)
            if "batch_size" in mf and row.get("B") is None:
                row["B"] = mf["batch_size"]
            if "amp" in mf and row.get("amp") is None:
                row["amp"] = mf["amp"]
            # si seed faltaba y el manifest la tiene
            if "seed" in mf and (row.get("seed") is None or (isinstance(row.get("seed"), float) and math.isnan(row["seed"]))):
                row["seed"] = mf["seed"]

        # 5) Rellena kg_per_hour si faltaba: usar duration_s o elapsed_sec
        if ("kg_per_hour" not in row or row.get("kg_per_hour") is None) and (row.get("emissions_kg") is not None):
            dur_s = None
            if row.get("duration_s"):  # de CodeCarbon
                dur_s = float(row["duration_s"])
            elif row.get("elapsed_sec"):  # del runner
                dur_s = float(row["elapsed_sec"])
            if dur_s and dur_s > 0:
                row["kg_per_hour"] = float(row["emissions_kg"]) / max(1e-9, (dur_s/3600.0))

        # 6) Asegura eval_matrix.* (o reconstruye desde continual_results.json)
        tasks, M = ensure_eval_matrix_files(run_dir)

        # 7) Derivadas por tarea
        forget_abs_vals, forget_rel_vals = [], []
        if tasks and M:
            final_idx = len(tasks) - 1
            for i, ti in enumerate(tasks):
                best, final, f_abs, f_rel = _row_best_final_forgetting(M[i])
                if (math.isnan(final) or final is None) and final_idx < len(M[i]):
                    try:
                        final = float(M[i][final_idx])
                    except Exception:
                        final = math.nan
                row[f"{ti}_best_mae"]   = best
                row[f"{ti}_final_mae"]  = final
                row[f"{ti}_forget_abs"] = f_abs
                row[f"{ti}_forget_rel"] = f_rel
                if not math.isnan(f_abs):
                    forget_abs_vals.append(f_abs)
                if not math.isnan(f_rel):
                    forget_rel_vals.append(f_rel)

        # 8) Agregados de olvido
        row["avg_forget_abs"] = float(np.mean(forget_abs_vals)) if forget_abs_vals else np.nan
        row["avg_forget_rel"] = float(np.mean(forget_rel_vals)) if forget_rel_vals else np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    # 9) Orden de columnas (y evitar duplicados)
    first = [
        "run_dir", "preset", "method", "encoder", "model", "seed", "T", "B", "amp",
        "params", "elapsed_sec", "duration_s", "emissions_kg", "energy_kwh",
        "kg_per_hour", "avg_forget_abs", "avg_forget_rel"
    ]
    other = [c for c in df.columns if c not in first]
    df = df.reindex(columns=[*first, *other])
    # elimina columnas duplicadas si las hubiera
    df = df.loc[:, ~df.columns.duplicated()]

    # 10) Orden “humano”
    sort_cols = [c for c in ["preset", "method", "encoder", "seed"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last", ignore_index=True)

    return df

# src/results_io.py
# -*- coding: utf-8 -*-
"""
Lectura y consolidación de resultados de runs 'continual_*'.

Incluye:
- parse_exp_name(name): preset/method/λ/encoder/model/seed (+ lambda_num, method_base)
- ensure_eval_matrix_files: garantiza eval_matrix.{json,csv} o la reconstruye
- build_results_table(outputs_root): tabla completa por run (eficiencia + métricas)
- build_runs_df(outputs_root): vista simplificada (compat con notebooks modernos)
- aggregate_and_show(df, outputs_root): agregados y CSV resumen
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import json, math, re
import pandas as pd
import numpy as np

# ----------------------- Nombres y regex comunes -----------------------
ALLOWED_ENC = r"(rate|latency|raw|image)"
_PAT = re.compile(
    rf"^continual_"
    rf"(?P<preset>[^_]+)_"               # preset
    rf"(?P<tag>.+)_"                     # tag (method y opcional lam_*)
    rf"(?P<enc>{ALLOWED_ENC})"           # encoder
    rf"(?:_model\-(?P<model>.+?))?"      # modelo (opcional)
    rf"(?:_seed_(?P<seed>\d+))?$",       # seed (opcional)
    re.IGNORECASE
)

def parse_exp_name(name: str) -> Dict[str, Any]:
    """
    Parsea nombres tipo:
      continual_<preset>_<tag>_<encoder>[_model-<model>][_seed_<seed>]
    Devuelve: preset, method, method_base, lambda, lambda_num, encoder, model, seed
    """
    meta = {
        "preset": None, "method": None, "method_base": None,
        "lambda": None, "lambda_num": None,
        "encoder": None, "seed": None, "model": None
    }
    m = _PAT.match(name)
    if not m:
        return meta
    preset = m.group("preset"); tag = m.group("tag"); enc = m.group("enc")
    seed = m.group("seed"); model = m.group("model")

    lam = None
    mlam = re.search(r"_lam_([^_]+)", tag)
    if mlam:
        lam = mlam.group(1)
        method = tag.replace(f"_lam_{lam}", "")
    else:
        method = tag

    # base = antes del primer '+', útil para agrupar as-snn+ewc -> as-snn
    method_base = method.split("+", 1)[0] if method else None
    lam_num = None
    if lam is not None:
        try:
            lam_num = float(lam)
        except Exception:
            lam_num = None

    meta.update({
        "preset": preset,
        "method": method,
        "method_base": method_base,
        "lambda": lam,
        "lambda_num": lam_num,
        "encoder": enc,
        "seed": int(seed) if seed is not None else None,
        "model": model
    })
    return meta


# =========================
# Helpers de E/S robustos
# =========================
def _safe_json_load(p: Path) -> dict:
    if not Path(p).exists():
        return {}
    try:
        with Path(p).open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def safe_read_json(path: Path | str) -> dict:
    """Lee JSON y devuelve {} si no existe o falla (API pública para notebooks)."""
    return _safe_json_load(Path(path))

def _lower_map(cols) -> Dict[str, str]:
    return {str(c).lower().strip(): str(c) for c in cols}


# -------------------------
# CodeCarbon / Telemetry
# -------------------------
def read_codecarbon_summary(exp_dir: Path) -> dict:
    """
    Devuelve resumen de CodeCarbon si existe emissions.csv:
      {'emissions_kg','duration_s','energy_kwh','cpu_kwh','gpu_kwh','ram_kwh','kg_per_hour'}
    """
    csv_path = Path(exp_dir) / "emissions.csv"
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

        if "energy_consumed" in cmap:  # versiones nuevas
            energy_kwh = float(last[cmap["energy_consumed"]])
            cpu_kwh = gpu_kwh = ram_kwh = None
        else:  # versiones clásicas
            cpu_kwh = float(last[cmap["cpu_energy"]]) if "cpu_energy" in cmap else np.nan
            gpu_kwh = float(last[cmap["gpu_energy"]]) if "gpu_energy" in cmap else np.nan
            ram_kwh = float(last[cmap["ram_energy"]]) if "ram_energy" in cmap else np.nan
            parts = [v for v in (cpu_kwh, gpu_kwh, ram_kwh) if pd.notna(v)]
            energy_kwh = float(np.sum(parts)) if parts else None

        out = {
            "emissions_kg": emissions_kg,
            "duration_s": duration_s,
            "energy_kwh": energy_kwh,
            "cpu_kwh": None if (cpu_kwh is None or pd.isna(cpu_kwh)) else cpu_kwh,
            "gpu_kwh": None if (gpu_kwh is None or pd.isna(gpu_kwh)) else gpu_kwh,
            "ram_kwh": None if (ram_kwh is None or pd.isna(ram_kwh)) else ram_kwh,
        }
        if out.get("emissions_kg") is not None and out.get("duration_s"):
            dur_h = max(1e-9, out["duration_s"] / 3600.0)
            out["kg_per_hour"] = out["emissions_kg"] / dur_h
        return out
    except Exception:
        return {}

def read_telemetry_last(exp_dir: Path) -> dict:
    """Lee el último evento de telemetry.jsonl si existe."""
    f = Path(exp_dir) / "telemetry.jsonl"
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
    res = _safe_json_load(Path(run_dir) / "continual_results.json")
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

def _invalid_last_col_all_nan(tasks: List[str], M: List[List[float]]) -> bool:
    try:
        A = np.array(M, dtype=float)
        if A.ndim != 2 or A.shape[0] != len(tasks) or A.shape[1] < 1:
            return True
        last_col = A[:, -1]
        return not np.isfinite(last_col).any()
    except Exception:
        return True

def ensure_eval_matrix_files(run_dir: Path) -> Tuple[List[str], List[List[float]]]:
    run_dir = Path(run_dir)
    csvp = run_dir / "eval_matrix.csv"
    jsp  = run_dir / "eval_matrix.json"

    # 1) JSON
    if jsp.exists():
        data = _safe_json_load(jsp)
        tasks = data.get("tasks") or []
        M = data.get("mae_matrix") or []
        if tasks and M and not _invalid_last_col_all_nan(tasks, M):
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
            if tasks and M and not _invalid_last_col_all_nan(tasks, M):
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
    if final is None or (isinstance(final, float) and math.isnan(final)) or best is None or (isinstance(best, float) and math.isnan(best)):
        return (best, final, math.nan, math.nan)
    f_abs = final - best
    f_rel = (f_abs / best) if best > 0 else math.nan
    return (best, final, f_abs, f_rel)


# ===========================
# Parse desde el nombre (fallback)
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
    m = _RUN_RE.match(run_name)
    if not m:
        return {}
    out = m.groupdict()
    if "seed" in out:
        try:
            out["seed"] = int(out["seed"])
        except Exception:
            pass
    return out


# ==================================
# Fallback per_task_perf (SCA & otros)
# ==================================
def _read_per_task_perf(run_dir: Path) -> Dict[str, float]:
    run_dir = Path(run_dir)
    out: Dict[str, float] = {}

    pj = run_dir / "per_task_perf.json"
    if pj.exists():
        try:
            P = json.loads(pj.read_text(encoding="utf-8"))
            if isinstance(P, list):
                items = P
            elif isinstance(P, dict) and "rows" in P:
                items = P["rows"]
            else:
                items = []
            for it in items:
                tname = str(it.get("task_name") or it.get("name") or it.get("task") or "")
                if not tname:
                    continue
                for k in ("final_mae", "test_mae", "val_mae", "best_val_mae"):
                    v = it.get(k)
                    if v is not None:
                        try:
                            out[tname] = float(v)
                            break
                        except Exception:
                            pass
        except Exception:
            pass

    if out:
        return out

    pc = run_dir / "per_task_perf.csv"
    if pc.exists():
        try:
            P = pd.read_csv(pc)
            tcol = None
            for cand in ("task_name","name","task"):
                if cand in P.columns:
                    tcol = cand
                    break
            if tcol is not None:
                for _, r in P.iterrows():
                    tname = str(r.get(tcol) or "")
                    if not tname:
                        continue
                    for k in ("final_mae","test_mae","val_mae"):
                        if k in P.columns and pd.notna(r.get(k)):
                            try:
                                out[tname] = float(r[k])
                                break
                            except Exception:
                                pass
        except Exception:
            pass
    return out


# ===========================
# Tabla principal de resultados
# ===========================
def build_results_table(outputs_root: Path | str) -> pd.DataFrame:
    """
    Escanea outputs/continual_* y construye un DataFrame con:
      - metadatos del run (preset, method, method_base, encoder, modelo, T, B, amp, seed, params)
      - resumen de eficiencia (elapsed_sec, emisiones, kWh, kg/h, etc.)
      - métricas por tarea (final_mae, best_mae, forgetting abs/rel)
      - medias de olvido (avg_forget_abs, avg_forget_rel)
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
                "method_base": (eff.get("method") or "").split("+", 1)[0] if eff.get("method") else None,
                "encoder": eff.get("encoder"),
                "model": eff.get("model"),
                "seed": eff.get("seed"),
                "T": eff.get("T"),
                "B": eff.get("batch_size") if eff.get("batch_size") is not None else eff.get("B"),
                "amp": eff.get("amp"),
                "params": eff.get("params_total"),
                "elapsed_sec": eff.get("elapsed_sec"),
            })
            if eff.get("emissions_kg") is not None:
                row["emissions_kg"] = eff.get("emissions_kg")

        # 2) Fallback: parse desde el nombre si faltan campos clave
        if any(k not in row or pd.isna(row.get(k)) for k in ("preset","method","encoder","model","seed")):
            parsed = parse_exp_name(run_dir.name)
            for k in ("preset","method","method_base","encoder","model","seed","lambda","lambda_num"):
                v = parsed.get(k)
                if v is not None:
                    row[k] = v

        # 3) CodeCarbon + Telemetry (si existen)
        row.update(read_codecarbon_summary(run_dir))
        row.update(read_telemetry_last(run_dir))

        # 4) Completar B/amp/seed desde el primer manifest disponible (si faltan)
        if any(k not in row or row.get(k) is None for k in ("B","amp","seed")):
            mf = {}
            for td in sorted(run_dir.glob("task_*")):
                cand = td / "manifest.json"
                if cand.exists():
                    mf = _safe_json_load(cand)
                    break
            if mf:
                if "batch_size" in mf and row.get("B") is None:
                    row["B"] = mf["batch_size"]
                if "amp" in mf and row.get("amp") is None:
                    row["amp"] = mf["amp"]
                if "seed" in mf and (row.get("seed") is None or (isinstance(row.get("seed"), float) and math.isnan(row["seed"]))):
                    row["seed"] = mf["seed"]

        # 4.bis) Normaliza alias batch_size
        if "B" in row and row["B"] is not None:
            try:
                row["B"] = int(row["B"])
            except Exception:
                pass
            row["batch_size"] = row["B"]
        elif "batch_size" in row and row["batch_size"] is not None:
            row["B"] = row["batch_size"]

        # 5) kg_per_hour si faltaba
        if ("kg_per_hour" not in row or row.get("kg_per_hour") is None) and (row.get("emissions_kg") is not None):
            dur_s = None
            if row.get("duration_s"):         # CodeCarbon
                dur_s = float(row["duration_s"])
            elif row.get("elapsed_sec"):      # runner
                dur_s = float(row["elapsed_sec"])
            if dur_s and dur_s > 0:
                row["kg_per_hour"] = float(row["emissions_kg"]) / max(1e-9, (dur_s/3600.0))

        # 6) Asegura eval_matrix.* (o reconstruye)
        tasks, M = ensure_eval_matrix_files(run_dir)

        # 7) Derivadas por tarea (desde eval_matrix)
        forget_abs_vals, forget_rel_vals = [], []
        if tasks and M:
            final_idx = len(tasks) - 1
            for i, ti in enumerate(tasks):
                try:
                    row_vals = [float(v) if v is not None else math.nan for v in M[i]]
                except Exception:
                    row_vals = M[i]
                best, final, f_abs, f_rel = _row_best_final_forgetting(row_vals)
                if (final is None or (isinstance(final, float) and math.isnan(final))) and final_idx < len(row_vals):
                    try:
                        v = float(row_vals[final_idx])
                        if not (isinstance(v, float) and math.isnan(v)):
                            final = v
                    except Exception:
                        pass

                row[f"{ti}_best_mae"]   = best
                row[f"{ti}_final_mae"]  = final
                row[f"{ti}_forget_abs"] = f_abs
                row[f"{ti}_forget_rel"] = f_rel
                if f_abs is not None and not (isinstance(f_abs, float) and math.isnan(f_abs)):
                    forget_abs_vals.append(f_abs)
                if f_rel is not None and not (isinstance(f_rel, float) and math.isnan(f_rel)):
                    forget_rel_vals.append(f_rel)

        # 7.bis) Fallback: per_task_perf si faltan finales
        if tasks:
            per_task = _read_per_task_perf(run_dir)
            if per_task:
                for ti in tasks:
                    col = f"{ti}_final_mae"
                    if col not in row or row[col] is None or (isinstance(row[col], float) and math.isnan(row[col])):
                        if ti in per_task:
                            row[col] = per_task[ti]

        # 8) Agregados de olvido
        row["avg_forget_abs"] = float(np.mean(forget_abs_vals)) if forget_abs_vals else np.nan
        row["avg_forget_rel"] = float(np.mean(forget_rel_vals)) if forget_rel_vals else np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    # 9) Orden de columnas
    first = [
        "run_dir", "preset", "method", "method_base", "encoder", "model", "seed",
        "T", "B", "batch_size", "amp", "params", "elapsed_sec", "duration_s",
        "emissions_kg", "energy_kwh", "kg_per_hour", "avg_forget_abs", "avg_forget_rel",
        "lambda", "lambda_num"
    ]
    other = [c for c in df.columns if c not in first]
    df = df.reindex(columns=[*first, *other])
    df = df.loc[:, ~df.columns.duplicated()]

    # 10) Orden “humano”
    sort_cols = [c for c in ["preset", "method_base", "method", "encoder", "seed"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last", ignore_index=True)
    return df


# -----------------------
# Vista simplificada + agregados
# -----------------------
def _find_first_last_tasks(res: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    task_names = list(res.keys())
    if len(task_names) < 2:
        return None, None
    # heurística: la última tarea NO tiene claves after_*
    def is_last(d: Dict[str, Any]) -> bool:
        return not any(k.startswith("after_") for k in d.keys())
    last_task = None; first_task = None
    for tn in task_names:
        if is_last(res[tn]): last_task = tn
        else: first_task = tn
    if first_task is None or last_task is None:
        task_names_sorted = sorted(task_names)
        first_task = task_names_sorted[0]; last_task = task_names_sorted[-1]
    return first_task, last_task

def extract_metrics(res: Dict[str, Any]) -> Dict[str, Any]:
    c1, c2 = _find_first_last_tasks(res)
    if not c1 or not c2:
        return {
            "c1": None, "c2": None,
            "c1_mae": float("nan"),
            "c1_after_c2_mae": float("nan"),
            "forget_rel_%": float("nan"),
            "c2_mae": float("nan"),
        }
    c1_mae = float(res[c1].get("test_mae", float("nan")))
    c2_mae = float(res[c2].get("test_mae", float("nan")))
    c1_after_c2_mae = float(res[c1].get(f"after_{c2}_mae", float("nan")))
    forget_rel = ((c1_after_c2_mae - c1_mae) / c1_mae * 100.0) if c1_mae == c1_mae else float("nan")
    return {
        "c1": c1, "c2": c2,
        "c1_mae": c1_mae,
        "c1_after_c2_mae": c1_after_c2_mae,
        "forget_rel_%": forget_rel,
        "c2_mae": c2_mae,
    }

def build_runs_df(outputs_root: Path | str) -> pd.DataFrame:
    """
    Vista ligera: una fila por run con c1/c2 y %olvido (útil para notebooks).
    """
    rows: List[Dict[str, Any]] = []
    for exp_dir in sorted(Path(outputs_root).glob("continual_*")):
        name = exp_dir.name
        meta = parse_exp_name(name)
        if meta["preset"] is None:
            continue
        results_path = Path(exp_dir) / "continual_results.json"
        if not results_path.exists():
            continue
        res = _safe_json_load(results_path)
        if not res:
            continue

        m = extract_metrics(res)
        rows.append({
            "exp": name,
            "preset": meta["preset"],
            "method": meta["method"],
            "method_base": meta["method_base"],
            "lambda": meta["lambda"],
            "lambda_num": meta["lambda_num"],
            "encoder": meta["encoder"],
            "model": meta["model"],
            "seed": meta["seed"],
            "c1_name": m["c1"], "c2_name": m["c2"],
            "c1_mae": m["c1_mae"],
            "c1_after_c2_mae": m["c1_after_c2_mae"],
            "c1_forgetting_mae_abs": (m["c1_after_c2_mae"] - m["c1_mae"]) if pd.notna(m["c1_mae"]) else float("nan"),
            "c1_forgetting_mae_rel_%": m["forget_rel_%"],
            "c2_mae": m["c2_mae"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["lambda_num"] = pd.to_numeric(df["lambda_num"], errors="coerce")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    df = df.sort_values(
        by=["preset", "method_base", "method", "encoder", "model", "lambda_num", "seed"],
        na_position="last", ignore_index=True
    )
    return df

def aggregate_and_show(df: pd.DataFrame, outputs_root: Path | str) -> pd.DataFrame:
    """
    Agrega, guarda CSV y devuelve el DataFrame agregado (muestra una vista si hay IPython).
    """
    if df.empty:
        print("No hay filas (¿no existen JSONs o solo hubo 1 tarea por run?).")
        return df

    cols_metrics = ["c1_mae", "c1_after_c2_mae", "c1_forgetting_mae_abs", "c1_forgetting_mae_rel_%", "c2_mae"]
    gdf = df.copy()
    if "lambda_num" not in gdf.columns:
        gdf["lambda_num"] = pd.to_numeric(gdf.get("lambda", None), errors="coerce")
    agg = (
        gdf
        .groupby(["preset", "method_base", "method", "encoder", "lambda", "lambda_num"], dropna=False)[cols_metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = ["_".join(filter(None, map(str, col))).rstrip("_")
                   for col in agg.columns.to_flat_index()]
    agg = agg.sort_values(by=["preset", "method_base", "method", "encoder", "lambda_num"],
                          na_position="last", ignore_index=True)

    # Persistir y mostrar (si hay IPython)
    summary_dir = Path(outputs_root) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    out_csv = summary_dir / "continual_summary_agg.csv"
    agg.to_csv(out_csv, index=False)
    print("Guardado:", out_csv)

    try:
        from IPython.display import display  # type: ignore
        def fmt(x, prec=4): return "" if pd.isna(x) else f"{x:.{prec}f}"
        show = agg.copy()
        count_cols = [c for c in show.columns if c.endswith("_count")]
        if count_cols:
            show["count"] = show[count_cols[0]].astype("Int64")
            show = show.drop(columns=count_cols)
        for c in [c for c in show.columns if c.endswith("_mean") or c.endswith("_std")]:
            show[c] = show[c].map(lambda v: fmt(v, 4))
        cols = [
            "preset", "method_base", "method", "encoder", "lambda",
            "c1_mae_mean", "c1_forgetting_mae_rel_%_mean", "c2_mae_mean",
            "c1_mae_std",  "c1_forgetting_mae_rel_%_std",  "c2_mae_std",
            "count"
        ]
        cols = [c for c in cols if c in show.columns]
        show = show[cols].rename(columns={
            "preset": "preset", "method_base": "base", "method": "método", "encoder": "codificador", "lambda": "λ",
            "c1_mae_mean": "MAE Tarea1 (media)",
            "c1_forgetting_mae_rel_%_mean": "Olvido T1 (%) (media)",
            "c2_mae_mean": "MAE Tarea2 (media)",
            "c1_mae_std": "MAE Tarea1 (σ)",
            "c1_forgetting_mae_rel_%_std": "Olvido T1 (%) (σ)",
            "c2_mae_std": "MAE Tarea2 (σ)",
            "count": "n (semillas)"
        })
        display(show)
    except Exception:
        pass

    return agg

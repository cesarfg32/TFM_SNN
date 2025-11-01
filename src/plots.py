# src/plots.py
# -*- coding: utf-8 -*-
"""plots.py
--------
Gráficas comparativas a partir de la tabla consolidada (build_results_table)
y curvas por tarea (lee task_*/manifest.json o, en su defecto, loss_curves.csv)."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, Any, List, Tuple, Optional

# ----------------------- utils básicos -----------------------
def _ensure_outdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _detect_tasks_from_df(df: pd.DataFrame) -> list[str]:
    """Detecta nombres de tareas viendo columnas *_final_mae."""
    tasks = []
    for c in df.columns:
        if c.endswith("_final_mae"):
            t = c[:-len("_final_mae")]
            tasks.append(t)
    return sorted(list(set(tasks)))

def _barplot(series: pd.Series, title: str, ylabel: str, outpath: Path):
    plt.figure(figsize=(max(8, 0.35 * max(1, len(series))), 4))
    plt.bar(series.index, series.values)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath.with_suffix(".png"), dpi=160)
    plt.savefig(outpath.with_suffix(".svg"))
    plt.show()

def _scatter(x, y, labels, title: str, xlabel: str, ylabel: str, outpath: Path):
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y)
    if labels is not None:
        for xi, yi, lab in zip(x, y, labels):
            if pd.isna(xi) or pd.isna(yi):
                continue
            try:
                plt.annotate(str(lab), (xi, yi), fontsize=8, alpha=0.7)
            except Exception:
                pass
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath.with_suffix(".png"), dpi=160)
    plt.savefig(outpath.with_suffix(".svg"))
    plt.show()

# ----------------- Tablas comparativas globales -----------------
def canonical_method(s: str) -> str:
    if not isinstance(s, str):
        return "unknown"
    import re
    t = s.lower()
    if ("rehearsal" in t) and ("+ewc" in t or "_ewc" in t):
        return "rehearsal+ewc"
    if "sca-snn" in t: return "sca-snn"
    if re.search(r"\bsa[-_]snn\b", t): return "sa-snn"
    if re.search(r"\bas[-_]snn\b", t): return "as-snn"
    if "colanet" in t: return "colanet"
    if re.search(r"\bewc\b", t) or "ewc_lam" in t: return "ewc"
    if "rehearsal" in t: return "rehearsal"
    if "naive" in t or "finetune" in t or "fine-tune" in t: return "naive"
    return t.split("_")[0]

def export_leaderboards(df: pd.DataFrame, outdir: Path | str, preset: Optional[str]="accurate", topN:int=6) -> Dict[str, Path]:
    """Genera CSVs de:
       - topN por score compuesto (MAE 0.5, olvido 0.4, emisiones 0.1)
       - ganadores por método_base
       - agregados por método_base (media y std)
       - top por variante de método (columna 'method' completa)
    """
    outdir = _ensure_outdir(Path(outdir))
    d = df.copy()
    if preset is not None and "preset" in d.columns:
        d = d[d["preset"] == preset].copy()
    # seleccionar columna de MAE de la última tarea
    final_cols = [c for c in d.columns if c.endswith("_final_mae")]
    assert final_cols, "No encuentro columnas *_final_mae."
    # orden por nombre natural (…_1, …_2, …)
    import re
    def _key(c):
        base = c.replace("_final_mae","")
        m = re.search(r"(\d+)$", base)
        idx = int(m.group(1)) if m else 0
        base = re.sub(r"\d+$","", base)
        return (base, idx)
    final_cols_sorted = sorted(final_cols, key=_key)
    MAE_COL = final_cols_sorted[-1]

    # coerciones
    for c in [MAE_COL, "emissions_kg", "avg_forget_rel"]:
        if c not in d.columns: d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # score compuesto
    def _norm01(vals: np.ndarray) -> np.ndarray:
        vals = vals.astype(float)
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.ones_like(vals)*0.5
        return (vals - lo) / (hi - lo)
    dn = d.copy()
    dn["_mae_n"]    = _norm01(dn[MAE_COL].values)
    dn["_forget_n"] = _norm01(dn["avg_forget_rel"].values)
    dn["_emiss_n"]  = _norm01(dn["emissions_kg"].values)
    w_mae, w_forget, w_emiss = 0.5, 0.4, 0.1
    dn["score"] = w_mae*dn["_mae_n"] + w_forget*dn["_forget_n"] + w_emiss*dn["_emiss_n"]

    # topN global
    topN_df = dn.sort_values("score", ascending=True).head(topN)
    p1 = outdir / "leaderboard_topN.csv"
    topN_df.to_csv(p1, index=False)

    # ganadores por método_base
    d["method_base"] = d["method"].astype(str).apply(canonical_method) if "method" in d.columns else "unknown"
    winners = (d.sort_values(["method_base", MAE_COL, "avg_forget_rel", "emissions_kg"])
                 .drop_duplicates(subset=["method_base"], keep="first")
                 .sort_values([MAE_COL, "avg_forget_rel", "emissions_kg"]))
    p2 = outdir / "winners_per_method.csv"
    winners.to_csv(p2, index=False)

    # agregados por método_base
    agg = (d.groupby("method_base", as_index=False)
             .agg({MAE_COL:["mean","std"], "avg_forget_rel":["mean","std"], "emissions_kg":["mean","std"], "elapsed_sec":["mean","std"]}))
    # aplanar columnas multiindex
    agg.columns = ["_".join([c for c in col if c]).rstrip("_") for col in agg.columns.values]
    p3 = outdir / "aggregates_per_method.csv"
    agg.to_csv(p3, index=False)

    # top por variante (columna 'method' tal cual)
    if "method" in d.columns:
        var_top = (d.sort_values(["method", MAE_COL, "avg_forget_rel", "emissions_kg"])
                    .drop_duplicates(subset=["method"], keep="first")
                    .sort_values([MAE_COL, "avg_forget_rel", "emissions_kg"]))
        p4 = outdir / "winners_per_method_variant.csv"
        var_top.to_csv(p4, index=False)
    else:
        p4 = outdir / "winners_per_method_variant.csv"
        pd.DataFrame().to_csv(p4, index=False)

    return {"topN": p1, "winners": p2, "aggregates": p3, "winners_variant": p4}

def plot_across_runs(df: pd.DataFrame, outdir: Path | str) -> Path:
    """Barras por run de *_final_mae (una por tarea), olvido relativo, emisiones, trade-off, y 'best per task'."""
    outdir = _ensure_outdir(Path(outdir))
    if df.empty:
        (outdir / "EMPTY.txt").write_text("No hay datos en el DataFrame.", encoding="utf-8")
        return outdir
    if "run_dir" in df.columns:
        df = df.copy().sort_values("run_dir", ignore_index=True)

    tasks = _detect_tasks_from_df(df)
    for t in tasks:
        col = f"{t}_final_mae"
        if col in df.columns:
            s = df.set_index("run_dir")[col] if "run_dir" in df.columns else df[col]
            s = pd.to_numeric(s, errors="coerce")
            if s.notna().any():
                _barplot(s.fillna(np.nan), f"MAE final - {t}", "MAE (test)", outdir / f"final_mae__{t}")

    for t in tasks:
        col = f"{t}_forget_rel"
        if col in df.columns:
            s = df.set_index("run_dir")[col] if "run_dir" in df.columns else df[col]
            s = pd.to_numeric(s, errors="coerce") * 100.0
            if s.notna().any():
                _barplot(s.fillna(np.nan), f"Olvido relativo - {t}", "% (relativo)", outdir / f"forget_rel_pct__{t}")

    if "emissions_kg" in df.columns and df["emissions_kg"].notna().any():
        s = df.set_index("run_dir")["emissions_kg"] if "run_dir" in df.columns else df["emissions_kg"]
        s = pd.to_numeric(s, errors="coerce")
        _barplot(s.fillna(np.nan), "Emisiones por experimento", "kg CO₂e", outdir / "emissions_per_run")

    final_cols = [c for c in df.columns if c.endswith("_final_mae")]
    first_task_col = final_cols[0] if final_cols else None
    if first_task_col and "emissions_kg" in df.columns:
        x = pd.to_numeric(df["emissions_kg"], errors="coerce")
        y = pd.to_numeric(df[first_task_col], errors="coerce")
        labels = df["preset"] if "preset" in df.columns else None
        _scatter(x, y, labels, "Trade-off: rendimiento vs emisiones",
                 "Emisiones (kg CO₂e)", f"MAE ({first_task_col[:-len('_final_mae')]})",
                 outdir / "tradeoff_emissions_vs_mae")

    best_rows = []
    for col in final_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            idx = series.idxmin(skipna=True)
            try:
                best_rows.append({
                    "task": col.replace("_final_mae", ""),
                    "best_run_dir": df.loc[idx, "run_dir"] if "run_dir" in df.columns else str(idx),
                    "best_value": float(series.loc[idx]),
                })
            except Exception:
                pass
    if best_rows:
        pd.DataFrame(best_rows).to_csv(outdir / "best_run_per_task.csv", index=False)
    return outdir

# ----------------- Histories (manifest.json o CSV) -----------------
def _safe_json_load(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _read_loss_csv(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
        out = {}
        for col in ["train_loss","val_loss","val_mae","val_mse","train_mae","train_mse"]:
            if col in df.columns:
                out[col] = df[col].astype(float).replace({np.nan:None}).tolist()
        return out
    except Exception:
        return {}

def _task_idx_from_name(name: str) -> Optional[int]:
    import re
    m = re.match(r"task[_-]?(\d+)", name.strip().lower())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _read_task_histories(run_dir: Path) -> List[Dict[str, Any]]:
    """Lee por tarea:
       1) task_*/manifest.json (nuevo runner)
       2) si no está, task_*/loss_curves.csv (histórico)
    """
    out = []
    for td in sorted(run_dir.glob("task_*")):
        man = _safe_json_load(td / "manifest.json")
        if man:
            hist = man.get("history") or {}
            meta = man.get("meta") or {}
            out.append({
                "task_dir": td.name,
                "task_idx": meta.get("task_idx", _task_idx_from_name(td.name)),
                "train_loss": hist.get("train_loss", []),
                "val_loss":   hist.get("val_loss",   []),
                "val_mae":    hist.get("val_mae",    []),
                "val_mse":    hist.get("val_mse",    []),
                "epochs": meta.get("epochs"),
                "batch_size": meta.get("batch_size"),
                "lr": meta.get("lr"),
                "amp": meta.get("amp"),
                "seed": meta.get("seed"),
            })
            continue
        # fallback: CSV
        hist = _read_loss_csv(td / "loss_curves.csv")
        out.append({
            "task_dir": td.name,
            "task_idx": _task_idx_from_name(td.name),
            "train_loss": hist.get("train_loss", []),
            "val_loss":   hist.get("val_loss",   []),
            "val_mae":    hist.get("val_mae",    []),
            "val_mse":    hist.get("val_mse",    []),
        })
    out.sort(key=lambda r: (9999 if r["task_idx"] is None else r["task_idx"], r["task_dir"]))
    return out

def _maybe_smooth(y: List[float], win: int | None) -> List[float]:
    if not y or not isinstance(win, int) or win is None or win < 2:
        return y
    a = np.asarray(y, dtype=float)
    k = min(win, len(a))
    if k < 2:
        return y
    kern = np.ones(k) / k
    z = np.convolve(a, kern, mode="same")
    return z.tolist()

# ----------------- MAE vs epochs (por run) -----------------
def plot_mae_curves_for_run(
    run_dir: Path | str,
    out_base: Path | str,
    smooth_window: int | None = None,
    title_prefix: str = "MAE (validación) vs Epoch",
) -> Path:
    """Una figura por tarea + grid (usa 'val_mae' si está; si no, caerá a val_loss etiquetado como MSE)."""
    run_dir = Path(run_dir)
    outdir = Path(out_base) / run_dir.name
    outdir.mkdir(parents=True, exist_ok=True)

    rows = _read_task_histories(run_dir)
    if not rows:
        (outdir / "NO_HISTORY.txt").write_text("Sin history en este run.", encoding="utf-8")
        return outdir

    # 1) Una figura por tarea
    for r in rows:
        mae = r.get("val_mae") or []
        use_mae = bool(mae and any([x is not None for x in mae]))
        y = [float(x) for x in mae] if use_mae else [float(x) for x in (r.get("val_loss") or [])]
        y_s = _maybe_smooth(y, smooth_window)
        if not y_s:
            continue
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(y_s)+1), y_s, label=("Val MAE" if use_mae else "Val MSE"), alpha=0.95)
        plt.xlabel("Epoch")
        plt.ylabel("MAE" if use_mae else "MSE (loss)")
        ttl = f"{title_prefix if use_mae else 'MSE (val) vs Epoch'} — {r['task_dir']}"
        plt.title(ttl)
        plt.legend()
        plt.tight_layout()
        fname = f"{'val_mae' if use_mae else 'val_mse'}__{r['task_dir']}"
        plt.savefig((outdir / fname).with_suffix(".png"), dpi=160)
        plt.savefig((outdir / fname).with_suffix(".svg"))
        plt.show()

    # 2) Grid de todas las tareas
    import math as _m
    K = len(rows)
    cols = 3 if K >= 6 else 2 if K >= 3 else 1
    rows_n = _m.ceil(K / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(6 * cols, 3.5 * rows_n), squeeze=False)
    ax_it = iter(axes.flatten())
    for r in rows:
        ax = next(ax_it, None)
        if ax is None:
            break
        mae = r.get("val_mae") or []
        use_mae = bool(mae and any([x is not None for x in mae]))
        y = [float(x) for x in mae] if use_mae else [float(x) for x in (r.get("val_loss") or [])]
        y_s = _maybe_smooth(y, smooth_window)
        if y_s:
            ax.plot(range(1, len(y_s)+1), y_s, label=("Val MAE" if use_mae else "Val MSE"))
        ax.set_title(r["task_dir"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE" if use_mae else "MSE")
        ax.legend()
    for ax in ax_it:
        ax.axis("off")
    fig.suptitle(f"{run_dir.name}", y=1.01)
    fig.tight_layout()
    fname = f"val_metrics__grid"
    fig.savefig((outdir / fname).with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig((outdir / fname).with_suffix(".svg"), bbox_inches="tight")
    plt.show()
    return outdir

# ----------------- Overlay entre runs -----------------
def overlay_val_mae_across_runs(
    run_dirs: List[Path | str],
    task_name_substr: str = "circuito2",
    out_path: Optional[Path | str] = None,
    smooth_window: int | None = None,
):
    plt.figure(figsize=(7,5))
    plotted = 0
    for rd in run_dirs:
        rd = Path(rd)
        task_dirs = [p for p in rd.iterdir() if p.is_dir() and task_name_substr in p.name]
        for td in task_dirs:
            man = _safe_json_load(td / "manifest.json")
            if man:
                y = man.get("history", {}).get("val_mae", []) or []
            else:
                y = _read_loss_csv(td / "loss_curves.csv").get("val_mae", []) or []
            if not y:
                continue
            y_s = _maybe_smooth([float(x) for x in y], smooth_window)
            label = rd.name[:90]
            plt.plot(range(1, len(y_s)+1), y_s, label=label, alpha=0.9)
            plotted += 1
    plt.xlabel("Epoch")
    plt.ylabel("MAE (validación)")
    plt.title(f"Validación MAE por epoch — tarea contiene '{task_name_substr}'")
    plt.grid(True, linestyle=":", alpha=0.5)
    if plotted:
        plt.legend(fontsize=8)
    if out_path:
        out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    else:
        plt.show()

# ----------------- Batch: curvas para todos los runs -----------------
def plot_loss_curves_all_runs(
    outputs_root: Path | str,
    out_base: Path | str,
    smooth_window: int | None = None,
) -> Path:
    outputs_root = Path(outputs_root)
    base = Path(out_base) / "plots_val_metrics"
    base.mkdir(parents=True, exist_ok=True)
    for rd in sorted(outputs_root.glob("continual_*")):
        try:
            plot_mae_curves_for_run(rd, base, smooth_window=smooth_window)
        except Exception:
            pass
    return base

# ===================== NUEVO: Heatmap e Intensidad energética =====================

def plot_eval_matrix_heatmap(run_dir: Path | str, out_base: Path | str) -> Path:
    """Heatmap de la matriz de MAE por tareas (eval_matrix.json)."""
    run_dir = Path(run_dir)
    outdir = Path(out_base) / run_dir.name
    outdir.mkdir(parents=True, exist_ok=True)

    jf = run_dir / "eval_matrix.json"
    if not jf.exists():
        (outdir / "NO_EVAL_MATRIX.txt").write_text("Falta eval_matrix.json", encoding="utf-8")
        return outdir

    data = _safe_json_load(jf)
    tasks = data.get("tasks", [])
    mat = data.get("mae_matrix", [])
    if not tasks or not mat:
        (outdir / "NO_EVAL_MATRIX.txt").write_text("Contenido vacío", encoding="utf-8")
        return outdir

    A = np.array(mat, dtype=float)
    fig, ax = plt.subplots(figsize=(1.2*len(tasks), 1.0*len(tasks)))
    im = ax.imshow(A, interpolation="nearest", aspect="auto")
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.set_yticks(range(len(tasks))); ax.set_yticklabels(tasks)
    ax.set_xlabel("Después de aprender tarea j"); ax.set_ylabel("MAE en tarea i")
    cbar = plt.colorbar(im); cbar.set_label("MAE")
    plt.title(f"Interferencia entre tareas — {run_dir.name}")
    plt.tight_layout()
    fig.savefig(outdir / "heatmap_eval_matrix.png", dpi=160)
    fig.savefig(outdir / "heatmap_eval_matrix.svg")
    plt.close(fig)
    return outdir


def plot_energy_by_task(run_dir: Path | str, out_base: Path | str) -> Path:
    """Reparte emisiones totales (kg CO₂e) por tarea proporcionalmente al train_time_sec."""
    run_dir = Path(run_dir)
    outdir = Path(out_base) / run_dir.name
    outdir.mkdir(parents=True, exist_ok=True)

    perf_p = run_dir / "per_task_perf.json"
    em_p   = run_dir / "emissions.csv"
    if not perf_p.exists() or not em_p.exists():
        (outdir / "NO_ENERGY_BY_TASK.txt").write_text("Falta per_task_perf.json o emissions.csv", encoding="utf-8")
        return outdir

    perf = pd.read_json(perf_p)

    # Lee emisiones totales de forma robusta
    em = pd.read_csv(em_p)
    total_kg = None
    cols = [c for c in em.columns]
    low = [c.lower() for c in cols]
    if "emissions" in low:
        # fila a fila (suele ser incremento por periodo)
        total_kg = float(em[cols[low.index("emissions")]].fillna(0).sum())
    elif "emissions_kg" in low:
        # a veces es acumulado -> coger última
        col = cols[low.index("emissions_kg")]
        total_kg = float(em[col].dropna().iloc[-1]) if em[col].notna().any() else None
    elif "co2e" in low:
        col = cols[low.index("co2e")]
        total_kg = float(em[col].dropna().iloc[-1]) if em[col].notna().any() else None

    if total_kg is None:
        (outdir / "NO_ENERGY_BY_TASK.txt").write_text("No pude estimar kg CO₂e totales", encoding="utf-8")
        return outdir

    # proporción por tiempo de entrenamiento
    t = perf["train_time_sec"].clip(lower=0.0)
    if t.sum() <= 0:
        (outdir / "NO_ENERGY_BY_TASK.txt").write_text("train_time_sec no válido", encoding="utf-8")
        return outdir
    perf = perf.assign(emissions_kg = total_kg * (t / t.sum()))

    # Gráfica
    labels = [f"{int(i)}:{n}" for i,n in zip(perf["task_idx"], perf["task_name"])]
    plt.figure(figsize=(max(6, 0.5*len(labels)), 4))
    plt.bar(labels, perf["emissions_kg"].values)
    plt.xticks(rotation=45, ha="right"); plt.ylabel("kg CO₂e"); plt.title("Emisiones por tarea (aprox.)")
    plt.tight_layout()
    plt.savefig(outdir / "energy_by_task.png", dpi=160)
    plt.savefig(outdir / "energy_by_task.svg")
    plt.close()

    perf.to_csv(outdir / "energy_by_task.csv", index=False)
    return outdir

# -*- coding: utf-8 -*-
"""
plots.py
--------
Gráficas comparativas a partir de la tabla consolidada (build_results_table).

Uso:
  from src.plots import plot_across_runs
  outdir = plot_across_runs(df, Path("outputs/summary/plots"))
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def plot_across_runs(df: pd.DataFrame, outdir: Path | str) -> Path:
    """
    Genera:
      - Barras por run de *_final_mae (una figura por tarea)
      - Barras por run de *_forget_rel (si existen) en %
      - Barras de emisiones_kg (si existen)
      - Scatter trade-off: emisiones_kg vs primer *_final_mae disponible
      - CSV con best run por tarea (mínimo *_final_mae)
    """
    outdir = _ensure_outdir(Path(outdir))

    if df.empty:
        (outdir / "EMPTY.txt").write_text("No hay datos en el DataFrame.", encoding="utf-8")
        return outdir

    # Orden por run_dir para estabilidad visual
    if "run_dir" in df.columns:
        df = df.copy().sort_values("run_dir", ignore_index=True)

    # 1) Final MAE por tarea
    tasks = _detect_tasks_from_df(df)
    for t in tasks:
        col = f"{t}_final_mae"
        if col not in df.columns:
            continue
        s = df.set_index("run_dir")[col] if "run_dir" in df.columns else df[col]
        s = pd.to_numeric(s, errors="coerce")
        if s.notna().any():
            _barplot(s.fillna(np.nan), f"MAE final - {t}", "MAE (test)", outdir / f"final_mae__{t}")

    # 2) Olvido relativo por tarea (%)
    for t in tasks:
        col = f"{t}_forget_rel"
        if col not in df.columns:
            continue
        s = df.set_index("run_dir")[col] if "run_dir" in df.columns else df[col]
        s = pd.to_numeric(s, errors="coerce") * 100.0
        if s.notna().any():
            _barplot(s.fillna(np.nan), f"Olvido relativo - {t}", "% (relativo)", outdir / f"forget_rel_pct__{t}")

    # 3) Emisiones totales por run
    if "emissions_kg" in df.columns and df["emissions_kg"].notna().any():
        s = df.set_index("run_dir")["emissions_kg"] if "run_dir" in df.columns else df["emissions_kg"]
        s = pd.to_numeric(s, errors="coerce")
        _barplot(s.fillna(np.nan), "Emisiones por experimento", "kg CO₂e", outdir / "emissions_per_run")

    # 4) Trade-off: emisiones vs primer *_final_mae válido
    final_cols = [c for c in df.columns if c.endswith("_final_mae")]
    first_task_col = final_cols[0] if final_cols else None
    if first_task_col and "emissions_kg" in df.columns:
        x = pd.to_numeric(df["emissions_kg"], errors="coerce")
        y = pd.to_numeric(df[first_task_col], errors="coerce")
        labels = df["preset"] if "preset" in df.columns else None
        _scatter(x, y, labels, "Trade-off: rendimiento vs emisiones",
                 "Emisiones (kg CO₂e)", f"MAE ({first_task_col[:-len('_final_mae')]})",
                 outdir / "tradeoff_emissions_vs_mae")

    # 5) Best run por tarea (mínimo final)
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

# === NUEVO: curvas de pérdida por tarea (MSE vs epochs) ======================
from typing import Dict, Any, List, Tuple, Optional
import json

def _safe_json_load(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _find_first_file(folder: Path, candidates=("manifest.json", "metrics.json")) -> Optional[Path]:
    for c in candidates:
        p = folder / c
        if p.exists():
            return p
    return None

def _maybe_smooth(y: List[float], win: int | None) -> List[float]:
    if not y or not isinstance(win, int) or win is None or win < 2:
        return y
    import numpy as np
    a = np.asarray(y, dtype=float)
    # padding simple en los bordes
    k = min(win, len(a))
    if k < 2:
        return y
    kern = np.ones(k) / k
    z = np.convolve(a, kern, mode="same")
    return z.tolist()

def _read_task_histories(run_dir: Path) -> List[Dict[str, Any]]:
    """Lee train_loss/val_loss por tarea, robusto a manifest.json/metrics.json."""
    out = []
    for td in sorted(run_dir.glob("task_*")):
        jf = _find_first_file(td)
        if jf is None:
            continue
        man = _safe_json_load(jf)
        hist = (man.get("history") or {})
        tr = hist.get("train_loss") or []
        va = hist.get("val_loss") or []
        out.append({
            "task_dir": td.name,
            "task_idx": _task_idx_from_name(td.name),
            "train_loss": tr,
            "val_loss": va,
            "epochs": man.get("epochs"),
            "batch_size": man.get("batch_size"),
            "lr": man.get("lr"),
            "amp": man.get("amp"),
            "seed": man.get("seed"),
        })
    # ordenar por idx si es posible
    out.sort(key=lambda r: (9999 if r["task_idx"] is None else r["task_idx"], r["task_dir"]))
    return out

def _task_idx_from_name(name: str) -> Optional[int]:
    # admite "task_1_nombre", "task_02_x", etc.
    import re
    m = re.match(r"task[_-]?(\d+)", name.strip().lower())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def plot_loss_curves_for_run(
    run_dir: Path | str,
    out_base: Path | str,
    smooth_window: int | None = None,
    ylog: bool = False,
    title_prefix: str = "Loss (MSE) vs Epoch",
) -> Path:
    """
    Dibuja curvas de pérdida por tarea para un run:
      - Una figura por tarea (train y val)
      - Una figura grid con todas las tareas
    Parámetros:
      smooth_window: si se da (ej. 3/5), aplica media móvil simple.
      ylog: True para escala log en Y.
    """
    run_dir = Path(run_dir)
    outdir = Path(out_base) / run_dir.name
    outdir.mkdir(parents=True, exist_ok=True)

    rows = _read_task_histories(run_dir)
    if not rows:
        # crea un marcador para que sepas que no había datos
        (outdir / "NO_HISTORY.txt").write_text(
            "Este run no tiene history.train_loss/val_loss en los task_*.",
            encoding="utf-8",
        )
        return outdir

    # 1) Una figura por tarea
    for r in rows:
        tr = [float(x) for x in r["train_loss"]] if r["train_loss"] else []
        va = [float(x) for x in r["val_loss"]] if r["val_loss"] else []
        tr_s = _maybe_smooth(tr, smooth_window)
        va_s = _maybe_smooth(va, smooth_window)
        ne = max(len(tr_s), len(va_s))
        if ne == 0:
            continue

        plt.figure(figsize=(6, 4))
        if tr_s:
            plt.plot(range(1, len(tr_s) + 1), tr_s, label="train")
        if va_s:
            plt.plot(range(1, len(va_s) + 1), va_s, label="val")
        plt.xlabel("Epoch")
        plt.ylabel("MSE (loss)")
        if ylog:
            plt.yscale("log")
        ttl = f"{title_prefix} — {r['task_dir']}"
        plt.title(ttl)
        plt.legend()
        plt.tight_layout()
        fname = f"loss_curve__{r['task_dir']}"
        plt.savefig((outdir / fname).with_suffix(".png"), dpi=160)
        plt.savefig((outdir / fname).with_suffix(".svg"))
        plt.show()

    # 2) Grid de todas las tareas
    import math
    K = len(rows)
    cols = 3 if K >= 6 else 2 if K >= 3 else 1
    rows_n = math.ceil(K / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(6 * cols, 3.5 * rows_n), squeeze=False)
    ax_it = iter(axes.flatten())
    for r in rows:
        ax = next(ax_it, None)
        if ax is None:
            break
        tr = [float(x) for x in r["train_loss"]] if r["train_loss"] else []
        va = [float(x) for x in r["val_loss"]] if r["val_loss"] else []
        tr_s = _maybe_smooth(tr, smooth_window)
        va_s = _maybe_smooth(va, smooth_window)
        if tr_s:
            ax.plot(range(1, len(tr_s) + 1), tr_s, label="train")
        if va_s:
            ax.plot(range(1, len(va_s) + 1), va_s, label="val")
        ax.set_title(r["task_dir"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (loss)")
        if ylog:
            ax.set_yscale("log")
        ax.legend()

    # si sobran axes, los apagamos
    for ax in ax_it:
        ax.axis("off")

    fig.suptitle(f"{title_prefix} — {run_dir.name}", y=1.01)
    fig.tight_layout()
    fname = f"loss_curves__grid"
    fig.savefig((outdir / fname).with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig((outdir / fname).with_suffix(".svg"), bbox_inches="tight")
    plt.show()

    return outdir

def plot_loss_curves_all_runs(
    outputs_root: Path | str,
    out_base: Path | str,
    smooth_window: int | None = None,
    ylog: bool = False,
) -> Path:
    """
    Lanza plot_loss_curves_for_run para todos los runs continual_* en outputs_root.
    """
    outputs_root = Path(outputs_root)
    base = Path(out_base) / "plots_loss_curves"
    base.mkdir(parents=True, exist_ok=True)
    for rd in sorted(outputs_root.glob("continual_*")):
        try:
            plot_loss_curves_for_run(rd, base, smooth_window=smooth_window, ylog=ylog)
        except Exception:
            # seguimos con los demás aunque uno falle
            pass
    return base

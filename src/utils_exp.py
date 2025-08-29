# src/utils_exp.py
from __future__ import annotations
from pathlib import Path
import re, json
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd

# ----------------------- Nombres y regex comunes -----------------------
ALLOWED_ENC = r"(rate|latency|raw|image)"
_PAT = re.compile(
    rf"^continual_"
    rf"(?P<preset>[^_]+)_"                 # preset
    rf"(?P<tag>.+)_"                       # tag (method y opcional lam_*)
    rf"(?P<enc>{ALLOWED_ENC})"             # encoder
    rf"(?:_model\-(?P<model>.+?))?"        # modelo (opcional)
    rf"(?:_seed_(?P<seed>\d+))?$"          # seed (opcional)
)

def parse_exp_name(name: str) -> Dict[str, Any]:
    """Parsea nombres tipo:
       continual_<preset>_<tag>_<encoder>[_model-<model>][_seed_<seed>]
       Devuelve: preset, method, lambda, encoder, model, seed
    """
    meta = {"preset": None, "method": None, "lambda": None,
            "encoder": None, "seed": None, "model": None}
    m = _PAT.match(name)
    if not m:
        return meta
    preset = m.group("preset"); tag = m.group("tag"); enc = m.group("enc")
    seed = m.group("seed"); model = m.group("model")
    mlam = re.search(r"_lam_([^_]+)", tag)
    if mlam:
        lam = mlam.group(1)
        method = tag.replace(f"_lam_{lam}", "")
    else:
        lam = None
        method = tag
    meta.update({
        "preset": preset,
        "method": method,
        "lambda": lam,
        "encoder": enc,
        "seed": int(seed) if seed is not None else None,
        "model": model
    })
    return meta

def safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

# ----------------------- Métricas por run -----------------------
def _find_first_last_tasks(res: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Primera y última tarea; si no hay claves after_*, usa orden alfabético."""
    task_names = list(res.keys())
    if len(task_names) < 2:
        return None, None
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
    """Devuelve un dict con claves estándar usadas por los notebooks."""
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
        "forget_rel_%": forget_rel,   # <-- nombre canónico
        "c2_mae": c2_mae,
    }

# ----------------------- Tabla por runs + agregados -----------------------
def build_runs_df(outputs_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for exp_dir in sorted((outputs_root).glob("continual_*")):
        name = exp_dir.name
        meta = parse_exp_name(name)
        if meta["preset"] is None:
            continue
        results_path = exp_dir / "continual_results.json"
        if not results_path.exists():
            continue
        res = safe_read_json(results_path)
        if not res:
            continue

        m = extract_metrics(res)
        c1, c2 = m["c1"], m["c2"]

        rows.append({
            "exp": name,
            "preset": meta["preset"],
            "method": meta["method"],
            "lambda": meta["lambda"],
            "encoder": meta["encoder"],
            "model": meta["model"],
            "seed": meta["seed"],
            "c1_name": c1, "c2_name": c2,
            "c1_mae": m["c1_mae"],
            "c1_after_c2_mae": m["c1_after_c2_mae"],
            "c1_forgetting_mae_abs": (m["c1_after_c2_mae"] - m["c1_mae"]) if pd.notna(m["c1_mae"]) else float("nan"),
            "c1_forgetting_mae_rel_%": m["forget_rel_%"],
            "c2_mae": m["c2_mae"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["lambda_num"] = pd.to_numeric(df["lambda"], errors="coerce")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    df = df.sort_values(
        by=["preset", "method", "encoder", "model", "lambda_num", "seed"],
        na_position="last", ignore_index=True
    )
    return df

def aggregate_and_show(df: pd.DataFrame, outputs_root: Path) -> pd.DataFrame:
    """Agrega, guarda CSV y devuelve el DataFrame agregado (también imprime una vista legible)."""
    if df.empty:
        print("No hay filas (¿no existen JSONs o solo hubo 1 tarea por run?).")
        return df

    # ---- vista detalle (igual que en notebook) ----
    from IPython.display import display
    display(df[[
        "exp","preset","method","lambda","encoder","model","seed",
        "c1_name","c2_name","c1_mae","c1_after_c2_mae",
        "c1_forgetting_mae_abs","c1_forgetting_mae_rel_%","c2_mae","lambda_num"
    ]])

    # ---- agregados ----
    cols_metrics = ["c1_mae", "c1_after_c2_mae", "c1_forgetting_mae_abs", "c1_forgetting_mae_rel_%", "c2_mae"]
    gdf = df.copy()
    if "lambda_num" not in gdf.columns:
        gdf["lambda_num"] = pd.to_numeric(gdf["lambda"], errors="coerce")
    agg = (gdf
           .groupby(["preset", "method", "encoder", "lambda", "lambda_num"], dropna=False)[cols_metrics]
           .agg(["mean", "std", "count"])
           .reset_index())
    agg.columns = ["_".join(filter(None, map(str, col))).rstrip("_")
                   for col in agg.columns.to_flat_index()]
    agg = agg.sort_values(by=["preset", "method", "encoder", "lambda_num"],
                          na_position="last", ignore_index=True)

    # ---- persistir y mostrar "bonito" ----
    summary_dir = outputs_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    out_csv = summary_dir / "continual_summary_agg.csv"
    agg.to_csv(out_csv, index=False)
    print("Guardado:", out_csv)

    def fmt(x, prec=4): return "" if pd.isna(x) else f"{x:.{prec}f}"
    show = agg.copy()
    count_cols = [c for c in show.columns if c.endswith("_count")]
    if count_cols:
        show["count"] = show[count_cols[0]].astype("Int64")
        show = show.drop(columns=count_cols)
    for c in [c for c in show.columns if c.endswith("_mean") or c.endswith("_std")]:
        show[c] = show[c].map(lambda v: fmt(v, 4))
    cols = [
        "preset", "method", "encoder", "lambda",
        "c1_mae_mean", "c1_forgetting_mae_rel_%_mean", "c2_mae_mean",
        "c1_mae_std",  "c1_forgetting_mae_rel_%_std",  "c2_mae_std",
        "count"
    ]
    cols = [c for c in cols if c in show.columns]
    show = show[cols].rename(columns={
        "preset": "preset", "method": "método", "encoder": "codificador", "lambda": "λ",
        "c1_mae_mean": "MAE Tarea1 (media)",
        "c1_forgetting_mae_rel_%_mean": "Olvido T1 (%) (media)",
        "c2_mae_mean": "MAE Tarea2 (media)",
        "c1_mae_std": "MAE Tarea1 (σ)",
        "c1_forgetting_mae_rel_%_std": "Olvido T1 (%) (σ)",
        "c2_mae_std": "MAE Tarea2 (σ)",
        "count": "n (semillas)"
    })
    display(show)
    return agg

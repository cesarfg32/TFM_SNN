# src/prep/data_prep.py
# -*- coding: utf-8 -*-
"""
Preparación offline del dataset Udacity:
- Limpieza de rutas (normaliza y las deja relativas a IMG/).
- Filtro de filas cuya imagen no exista.
- (Opcional) Expansión de cámaras: genera filas nuevas con 'center' apuntando
  a la imagen 'left' o 'right' y corrige el `steering` en ±shift (clip [-1,1]).
- Split estratificado por bins de 'steering' → train/val/test.
- (Opcional) Oversampling por bins en train → train_balanced.csv.
- Escritura de:
    * canonical.csv (limpio sin balanceo)
    * train.csv / val.csv / test.csv
    * train_balanced.csv (si se pidió oversampling)
    * tasks.json (apunta a train.csv)
    * tasks_balanced.json (apunta a train_balanced.csv si existe)
    * prep_manifest.json (resumen reproducible de la preparación)

NOTA DE COMPATIBILIDAD:
- Tu pipeline de entrenamiento usa `UdacityCSV` que coge la columna `center`.
- Por eso, si activas la expansión de cámaras, generamos filas nuevas
  REASIGNANDO `center` = (left o right), y ajustamos el `steering`.
  Así no hay que cambiar nada en los DataLoaders y el modelo ve más variedad.

Uso CLI recomendado desde la raíz del repo:
    python -m tools.make_splits_balanced --root . --runs circuito1 circuito2 \
      --use-left-right --steer-shift 0.2 \
      --bins 21 --train 0.70 --val 0.15 \
      --target-per-bin auto --cap-per-bin 12000 \
      --seed 42
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union
import argparse, json
import numpy as np
import pandas as pd


# ------------------------------- Utils --------------------------------
def _json_default(o):
    # Paths ⇒ str
    if isinstance(o, Path):
        return str(o)
    # numpy escalares ⇒ tipos Python
    try:
        import numpy as _np
        if isinstance(o, (_np.integer, _np.floating)):
            return o.item()
    except Exception:
        pass
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

def _fix_path(s: str) -> str:
    if pd.isna(s): 
        return s
    s = str(s).strip().strip('"').strip("'").replace("\\", "/")
    return s

def _rel_to_img(s: str) -> str:
    """Devuelve ruta relativa desde 'IMG/...' si aparece; si no, la deja igual."""
    if pd.isna(s):
        return s
    low = s.lower()
    k = low.rfind("img/")
    return s[k:] if k != -1 else s

def _clip_steer(x: float) -> float:
    return float(np.clip(float(x), -1.0, 1.0))


# --------------------------- Carga y limpieza --------------------------

CSV_COLS = ["center","left","right","steering","throttle","brake","speed"]

def load_raw_log(base: Path) -> pd.DataFrame:
    """Lee driving_log.csv, normaliza rutas y filtra filas sin imágenes."""
    csv = base / "driving_log.csv"
    assert csv.exists(), f"No existe {csv}"
    df = pd.read_csv(csv, header=None, names=CSV_COLS)

    # Normaliza y relativiza
    for cam in ["center", "left", "right"]:
        df[cam] = df[cam].map(_fix_path).map(_rel_to_img)

    # Convierte numéricos
    for c in ["steering","throttle","brake","speed"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filtra filas sin imagen (al menos center; y si quieres, también left/right)
    df = df[df["center"].map(lambda s: (base / s).exists())]
    # Si quieres filtrar también left/right, descomenta:
    # for cam in ["left","right"]:
    #     df = df[(base / df[cam]).apply(lambda p: p.exists())]

    df = df.dropna(subset=["steering"]).reset_index(drop=True)
    return df

def _rel_to_run_from_log(s: str, log_dir: Path, run_root: Path) -> str:
    """Normaliza s y la convierte a ruta RELATIVA a run_root (p.ej. 'vuelta1/IMG/...').
    - Si s contiene 'IMG/', tomamos desde ahí (ignora prefijos absolutos tipo C:/...).
    - Si no, resolvemos relativo a la carpeta del driving_log (log_dir) y luego a run_root.
    """
    if pd.isna(s):
        return s
    s = _fix_path(s)

    # Preferencia: cortar desde 'IMG/' si aparece
    low = s.lower()
    k = low.rfind("img/")
    if k != -1:
        abs_p = (log_dir / s[k:]).resolve()
    else:
        p = Path(s)
        abs_p = p if p.is_absolute() else (log_dir / p)
        abs_p = abs_p.resolve()

    # Intentar relativo a run_root
    try:
        rel = abs_p.relative_to(run_root.resolve())
        return rel.as_posix()
    except Exception:
        # Último recurso: recortar desde 'IMG/' en la ruta ya resuelta
        low2 = str(abs_p).lower()
        k2 = low2.rfind("img/")
        if k2 != -1:
            return str(abs_p)[k2:].replace("\\", "/")
        # Fallback final: nombre de archivo
        return abs_p.name


def load_raw_logs_merged(run_root: Path) -> pd.DataFrame:
    """Fusiona TODOS los driving_log.csv bajo run_root (recursivo), ignorando 'aug/'.
    Reescribe center/left/right a rutas RELATIVAS a run_root: 'vueltaX/IMG/...'
    Filtra filas cuya imagen CENTER no exista.
    """
    logs = [p for p in run_root.rglob("driving_log.csv") if "aug" not in p.parts]
    if not logs:
        raise FileNotFoundError(f"No se encontraron driving_log.csv bajo {run_root}")
    dfs = []
    for lp in sorted(logs):
        df = pd.read_csv(lp, header=None, names=CSV_COLS)
        # normaliza y relativiza respecto al run_root, resolviendo contra lp.parent
        for cam in ["center", "left", "right"]:
            if cam in df.columns:
                df[cam] = df[cam].map(lambda s: _rel_to_run_from_log(s, lp.parent, run_root))
        # numéricos + filtrado
        for c in ["steering", "throttle", "brake", "speed"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[df["center"].map(lambda s: (run_root / s).exists())]
        df = df.dropna(subset=["steering"]).reset_index(drop=True)
        dfs.append(df)
    if not dfs:
        raise RuntimeError(f"No se generó ningún DF válido para {run_root}")
    dfs = [d for d in dfs if not d.empty]
    if not dfs:
        raise RuntimeError(f"Todas las filas fueron filtradas por inexistentes en {run_root}. "
                        f"¿Rutas del CSV no casan con el árbol de imágenes?")
    return pd.concat(dfs, ignore_index=True)


# ----------------------- Expansión de cámaras -------------------------

def expand_cameras_into_center(df: pd.DataFrame, steer_shift: float) -> pd.DataFrame:
    """
    Devuelve un nuevo DF donde:
      - Filas originales: center = center (steering intacto)
      - Filas extra para LEFT:  center = left,  steering += shift (clip)
      - Filas extra para RIGHT: center = right, steering -= shift (clip)

    Mantiene el esquema original de columnas (center,left,right,steering,...),
    pero el entrenamiento solo usa `center`, así funcionan tus DataLoaders tal cual.
    """
    base = df.copy()

    left_df  = df.copy()
    left_df["center"]   = df["left"]
    left_df["steering"] = df["steering"].astype(float) + float(steer_shift)
    left_df["steering"] = left_df["steering"].map(_clip_steer)

    right_df = df.copy()
    right_df["center"]   = df["right"]
    right_df["steering"] = df["steering"].astype(float) - float(steer_shift)
    right_df["steering"] = right_df["steering"].map(_clip_steer)

    out = pd.concat([base, left_df, right_df], ignore_index=True)
    return out


# ---------------------- Split estratificado por bins ------------------

def stratified_split(
    df: pd.DataFrame,
    bins: int = 21,
    train: float = 0.70,
    val: float = 0.15,
    seed: int = 42,
    return_edges: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Split estratificado por histogramas de `steering` (pd.cut).

    - `bins`: nº de bordes (steering se recorta a [-1,1] para asegurar cobertura).
    - Reparte por bin con RNG fijo (reproducible).
    """
    s = df["steering"].astype(float).clip(-1.0, 1.0)
    edges = np.linspace(-1.0, 1.0, bins)
    lab = pd.cut(s, bins=edges, include_lowest=True, labels=False)

    rng = np.random.default_rng(seed)
    parts = []
    for b in sorted(lab.dropna().unique()):
        g = df[lab == b]
        idx = np.arange(len(g))
        rng.shuffle(idx)
        n = len(idx)
        ntr = int(round(n * train))
        nva = int(round(n * val))
        parts.append((g.iloc[idx[:ntr]],
                      g.iloc[idx[ntr:ntr+nva]],
                      g.iloc[idx[ntr+nva:]]))

    tr = pd.concat([a for a,_,_ in parts], ignore_index=True)
    va = pd.concat([b for _,b,_ in parts], ignore_index=True)
    te = pd.concat([c for _,_,c in parts], ignore_index=True)
    if return_edges:
        return tr, va, te, edges
    return tr, va, te

def verify_processed_splits(proc_root: Path, runs: list[str]) -> None:
    missing = []
    for run in runs:
        base = proc_root / run
        for part in ["train","val","test"]:
            p = base / f"{part}.csv"
            if not p.exists():
                missing.append(str(p))
    if missing:
        raise FileNotFoundError("Faltan CSV obligatorios:\n" + "\n".join(" - " + m for m in missing))

# -------------------------- Oversampling por bins ---------------------

def make_balanced_by_bins(
    train_df: pd.DataFrame,
    bins: int = 21,
    target_per_bin: str | int = "auto",
    cap_per_bin: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Oversampling por bins sobre el split de entrenamiento.

    - target_per_bin = "auto": usa min(max_bin, cap_per_bin) como objetivo.
      (es decir, iguala todos los bins al más poblado, con techo cap_per_bin)
    - o pasa un entero (p.ej., 8000) para fijar objetivo por bin.
    """
    s = train_df["steering"].astype(float).clip(-1.0, 1.0)
    edges = np.linspace(-1.0, 1.0, bins)
    lab = pd.cut(s, bins=edges, include_lowest=True, labels=False)

    counts = lab.value_counts().sort_index()
    max_bin = int(counts.max()) if len(counts) else 0
    if isinstance(target_per_bin, str) and target_per_bin == "auto":
        target = max_bin
        if cap_per_bin is not None:
            target = min(target, int(cap_per_bin))
    else:
        target = int(target_per_bin)

    rng = np.random.default_rng(seed)
    out_parts = []
    for b in range(len(edges) - 1):
        g = train_df[lab == b]
        if g.empty:
            continue
        k = target - len(g)
        if k <= 0:
            out_parts.append(g)
        else:
            # Oversampling con reemplazo para igualar al objetivo
            idx = rng.integers(low=0, high=len(g), size=k)
            dup = g.iloc[idx]
            out_parts.append(pd.concat([g, dup], ignore_index=True))

    balanced = pd.concat(out_parts, ignore_index=True) if out_parts else train_df.copy()
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # baraja
    return balanced


# --------------------------- Escritura salidas ------------------------

def _write_splits(out_dir: Path, tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.csv").write_text("", encoding="utf-8")  # pre-crea vacío por atomicidad
    tr.to_csv(out_dir / "train.csv", index=False)
    va.to_csv(out_dir / "val.csv", index=False)
    te.to_csv(out_dir / "test.csv", index=False)

def _write_tasks_json(proc_root: Path, runs: list[str], balanced: bool):
    obj = {"tasks_order": runs, "splits": {}}
    name = "tasks_balanced.json" if balanced else "tasks.json"
    for run in runs:
        base = proc_root / run
        tr = base / ("train_balanced.csv" if balanced else "train.csv")
        obj["splits"][run] = {
            "train": str((tr).resolve()),
            "val":   str((base / "val.csv").resolve()),
            "test":  str((base / "test.csv").resolve()),
        }
    (proc_root / name).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return proc_root / name


# ------------------------------- Main ---------------------------------
@dataclass
class PrepConfig:
    root: Path
    runs: list[str]
    use_left_right: bool
    steer_shift: float
    bins: int
    train: float
    val: float
    seed: int
    target_per_bin: Optional[Union[str, int]] = None
    cap_per_bin: Optional[int] = None
    merge_subruns: bool = False 

def run_prep(cfg: PrepConfig) -> dict:
    RAW  = cfg.root / "data" / "raw" / "udacity"
    PROC = cfg.root / "data" / "processed"
    PROC.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config": asdict(cfg),
        "runs": {},
    }
    for run in cfg.runs:
        base = RAW / run
        out_dir = PROC / run
        out_dir.mkdir(parents=True, exist_ok=True)

        df = load_raw_logs_merged(base) if cfg.merge_subruns else load_raw_log(base)
        df.to_csv(out_dir / "canonical.csv", index=False)

        if cfg.use_left_right:
            df = expand_cameras_into_center(df, cfg.steer_shift)

        tr, va, te = stratified_split(df, bins=cfg.bins, train=cfg.train, val=cfg.val, seed=cfg.seed)
        _write_splits(out_dir, tr, va, te)

        # === SOLO si se pide balanceo por duplicación de filas (LEGACY) ===
        trb_path = None
        if cfg.target_per_bin not in (None, "none", ""):
            tr_balanced = make_balanced_by_bins(
                tr,
                bins=cfg.bins,
                target_per_bin=cfg.target_per_bin,
                cap_per_bin=cfg.cap_per_bin,
                seed=cfg.seed
            )
            if len(tr_balanced) > len(tr):
                trb_path = out_dir / "train_balanced.csv"
                tr_balanced.to_csv(trb_path, index=False)

        manifest["runs"][run] = {
            "canonical": str(out_dir / "canonical.csv"),
            "train":     str(out_dir / "train.csv"),
            "val":       str(out_dir / "val.csv"),
            "test":      str(out_dir / "test.csv"),
            "train_balanced": str(trb_path) if trb_path else None,
            "stats": {
                "n_canonical": int(len(pd.read_csv(out_dir / "canonical.csv"))),
                "n_train":     int(len(tr)),
                "n_val":       int(len(va)),
                "n_test":      int(len(te)),
                "n_train_balanced": (int(len(pd.read_csv(trb_path))) if trb_path else None),
            }
        }

    tasks_path = _write_tasks_json(PROC, cfg.runs, balanced=False)
    balanced_exists = all((PROC / run / "train_balanced.csv").exists() for run in cfg.runs)
    tasks_balanced_path = _write_tasks_json(PROC, cfg.runs, balanced=True) if balanced_exists else None

    manifest["outputs"] = {
        "tasks_json": str(tasks_path),
        "tasks_balanced_json": str(tasks_balanced_path) if tasks_balanced_path else None,
    }
    (PROC / "prep_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8"
    )
    return manifest


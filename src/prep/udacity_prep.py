# src/prep/udacity_prep.py
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
import argparse, json
import numpy as np
import pandas as pd


# ------------------------------- Utils --------------------------------

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
    df = df[(base / df["center"]).apply(lambda p: p.exists())]
    # Si quieres filtrar también left/right, descomenta:
    # for cam in ["left","right"]:
    #     df = df[(base / df[cam]).apply(lambda p: p.exists())]

    df = df.dropna(subset=["steering"]).reset_index(drop=True)
    return df


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    return tr, va, te


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
            "train": str(tr),
            "val":   str(base / "val.csv"),
            "test":  str(base / "test.csv"),
        }
    (proc_root / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
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
    target_per_bin: str | int
    cap_per_bin: int | None

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
        df = load_raw_log(base)
        df.to_csv(PROC / run / "canonical.csv", index=False)

        # expansión de cámaras (opcional)
        if cfg.use_left_right:
            df = expand_cameras_into_center(df, cfg.steer_shift)

        # split estratificado
        tr, va, te = stratified_split(df, bins=cfg.bins, train=cfg.train, val=cfg.val, seed=cfg.seed)
        _write_splits(PROC / run, tr, va, te)

        # oversampling por bins en train (opcional si target auto o int > 0)
        tr_balanced = make_balanced_by_bins(
            tr, bins=cfg.bins, target_per_bin=cfg.target_per_bin, cap_per_bin=cfg.cap_per_bin, seed=cfg.seed
        )
        if len(tr_balanced) > len(tr):
            tr_balanced.to_csv(PROC / run / "train_balanced.csv", index=False)

        manifest["runs"][run] = {
            "canonical": str(PROC / run / "canonical.csv"),
            "train":     str(PROC / run / "train.csv"),
            "val":       str(PROC / run / "val.csv"),
            "test":      str(PROC / run / "test.csv"),
            "train_balanced": str(PROC / run / "train_balanced.csv") if (PROC / run / "train_balanced.csv").exists() else None,
            "stats": {
                "n_canonical": int(len(pd.read_csv(PROC / run / "canonical.csv"))),
                "n_train":     int(len(tr)),
                "n_val":       int(len(va)),
                "n_test":      int(len(te)),
                "n_train_balanced": int(len(tr_balanced)) if (PROC / run / "train_balanced.csv").exists() else None,
            }
        }

    # tasks.json (normal) y tasks_balanced.json (si procede)
    tasks_path = _write_tasks_json(PROC, cfg.runs, balanced=False)
    balanced_exists = all((PROC / run / "train_balanced.csv").exists() for run in cfg.runs)
    tasks_balanced_path = None
    if balanced_exists:
        tasks_balanced_path = _write_tasks_json(PROC, cfg.runs, balanced=True)

    manifest["outputs"] = {
        "tasks_json": str(tasks_path),
        "tasks_balanced_json": str(tasks_balanced_path) if tasks_balanced_path else None,
    }

    (PROC / "prep_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Prep Udacity: limpieza, split y balanceo por bins (opcional).")
    ap.add_argument("--root", type=Path, default=Path("."), help="Raíz del repo (contiene data/).")
    ap.add_argument("--runs", nargs="+", required=True, help="Nombres de recorridos: p.ej. circuito1 circuito2")
    ap.add_argument("--use-left-right", action="store_true", help="Expande cámaras left/right reubicándolas en 'center'.")
    ap.add_argument("--steer-shift", type=float, default=0.2, help="Corrección de steering para left/right (±shift).")
    ap.add_argument("--bins", type=int, default=21, help="Nº de bins para estratificar/balancear steering.")
    ap.add_argument("--train", type=float, default=0.70, help="Proporción de train.")
    ap.add_argument("--val", type=float, default=0.15, help="Proporción de val (resto es test).")
    ap.add_argument("--seed", type=int, default=42, help="Semilla global.")
    ap.add_argument("--target-per-bin", default="auto", help="'auto' o un entero (objetivo por bin en train).")
    ap.add_argument("--cap-per-bin", type=int, default=12000, help="Techo por bin cuando target_per_bin='auto'.")
    args = ap.parse_args()

    target = args.target_per_bin
    if isinstance(target, str) and target.lower() != "auto":
        try:
            target = int(target)  # permite pasar "--target-per-bin 8000"
        except Exception:
            raise ValueError("--target-per-bin debe ser 'auto' o un entero.")

    cfg = PrepConfig(
        root=args.root,
        runs=list(args.runs),
        use_left_right=bool(args.use_left_right),
        steer_shift=float(args.steer_shift),
        bins=int(args.bins),
        train=float(args.train),
        val=float(args.val),
        seed=int(args.seed),
        target_per_bin=target,
        cap_per_bin=int(args.cap_per_bin) if args.cap_per_bin is not None else None,
    )
    manifest = run_prep(cfg)
    print("OK:", (Path(cfg.root)/"data"/"processed"/"prep_manifest.json"))
    if manifest["outputs"]["tasks_balanced_json"]:
        print(" - tasks_balanced.json:", manifest["outputs"]["tasks_balanced_json"])
    print(" - tasks.json:", manifest["outputs"]["tasks_json"])


if __name__ == "__main__":
    main()

# tools/prep_udacity.py
from __future__ import annotations
import argparse
from pathlib import Path

# --- asegurar raíz del repo en sys.path ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------

from src.prep.data_prep import run_prep, PrepConfig

def main():
    ap = argparse.ArgumentParser(
        description="Prep Udacity: limpieza, split estratificado y balanceo offline (opcional)."
    )
    ap.add_argument("--root", type=Path, default=Path("."), help="Raíz del repo (contiene data/).")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Lista de recorridos, p.ej. circuito1 circuito2")
    ap.add_argument("--use-left-right", action="store_true",
                    help="Expande cámaras left/right reubicándolas en 'center'.")
    ap.add_argument("--steer-shift", type=float, default=0.2,
                    help="Corrección de steering para left/right (±shift).")
    ap.add_argument("--bins", type=int, default=21,
                    help="Nº de bins para estratificar/balancear steering.")
    ap.add_argument("--train", type=float, default=0.70, help="Proporción de train.")
    ap.add_argument("--val",   type=float, default=0.15, help="Proporción de val (resto es test).")
    ap.add_argument("--seed",  type=int,   default=42,   help="Semilla global.")
    ap.add_argument("--target-per-bin", default="auto",
                    help="'auto' o un entero (objetivo por bin en train).")
    ap.add_argument("--cap-per-bin", type=int, default=12000,
                    help="Techo por bin cuando target_per_bin='auto' (None = sin techo).")

    args = ap.parse_args()

    # Normaliza target-per-bin
    target = args.target_per_bin
    if isinstance(target, str) and target.lower() != "auto":
        try:
            target = int(target)
        except Exception:
            ap.error("--target-per-bin debe ser 'auto' o un entero")

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

    print("OK:", Path(cfg.root) / "data" / "processed" / "prep_manifest.json")
    if manifest["outputs"]["tasks_balanced_json"]:
        print(" - tasks_balanced.json:", manifest["outputs"]["tasks_balanced_json"])
    print(" - tasks.json:", manifest["outputs"]["tasks_json"])

if __name__ == "__main__":
    main()

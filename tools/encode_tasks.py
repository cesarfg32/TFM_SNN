# tools/encode_tasks.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

# Asegurar raíz del repo en sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prep.encode_offline import encode_csv_to_h5

def _p(s: str | Path) -> Path:
    p = Path(s)
    return p if p.is_absolute() else (ROOT / p)

def main():
    ap = argparse.ArgumentParser(
        description="Codifica en H5 (offline) todos los splits definidos en tasks.json / tasks_balanced.json."
    )
    ap.add_argument("--tasks-file", default="data/processed/tasks_balanced.json",
                    help="Ruta a tasks.json o tasks_balanced.json")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="Opcional: lista de runs a incluir (por nombre). Si no se indica, usa tasks_order completa.")
    ap.add_argument("--encoder", choices=["rate","latency","raw"], default="rate")
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--gain", type=float, default=0.5,
                    help="Solo aplica a encoder=rate (ignorado en latency/raw)")
    ap.add_argument("--w", type=int, default=200)
    ap.add_argument("--h", type=int, default=66)
    ap.add_argument("--rgb", action="store_true", help="Por defecto codifica en gris; añade --rgb para color.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="Reescribe H5 si ya existe.")
    args = ap.parse_args()

    tasks = json.loads(_p(args.tasks_file).read_text(encoding="utf-8"))
    order = tasks["tasks_order"] if args.runs is None else [r for r in args.runs if r in tasks["tasks_order"]]

    for run in order:
        splits = tasks["splits"][run]
        base_dir = ROOT / "data" / "raw" / "udacity" / run

        for split_name in ["train", "val", "test"]:
            csv_path = _p(splits[split_name])
            out_dir  = csv_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)

            suffix_gain = f"gain{args.gain}" if args.encoder == "rate" else "gain0"
            color = "rgb" if args.rgb else "gray"
            out_name = f"{split_name}_{args.encoder}_T{args.T}_{suffix_gain}_{color}_{args.w}x{args.h}.h5"
            out_path = out_dir / out_name

            if out_path.exists() and not args.force:
                print(f"[SKIP] {out_path.name} (ya existe)")
                continue

            encode_csv_to_h5(
                csv_df_or_path=csv_path,
                base_dir=base_dir,
                out_path=out_path,
                encoder=args.encoder,
                T=args.T,
                gain=args.gain,
                size_wh=(args.w, args.h),
                to_gray=(not args.rgb),
                seed=args.seed,
            )
            print("OK:", out_path)

if __name__ == "__main__":
    main()

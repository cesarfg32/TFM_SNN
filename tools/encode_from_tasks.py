# tools/encode_from_tasks.py
from __future__ import annotations

# --- asegurar raíz del repo en sys.path ---
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------

import argparse
from src.prep.encode_offline import encode_csv_to_h5


def main():
    ap = argparse.ArgumentParser(
        description="Genera H5 de spikes (offline) para todos los runs/splits definidos en tasks.json|tasks_balanced.json"
    )
    ap.add_argument("--tasks-file", required=True, help="p. ej., data/processed/tasks.json")
    ap.add_argument("--encoder", choices=["rate", "latency", "raw"], default="rate")
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--gain", type=float, default=0.5, help="Sólo aplica a encoder='rate'")
    ap.add_argument("--w", type=int, default=160)
    ap.add_argument("--h", type=int, default=80)
    ap.add_argument("--rgb", action="store_true", help="Por defecto gris; si pasas --rgb, to_gray=False")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only-missing", action="store_true", help="No sobreescribir si ya existe (por defecto ON)")
    ap.add_argument("--overwrite", action="store_true", help="Forzar sobrescritura (incompatible con --only-missing)")

    args = ap.parse_args()

    if args.overwrite and args.only_missing:
        raise SystemExit("Usa --overwrite *o* --only-missing, pero no ambos.")

    tasks = json.loads(Path(args.tasks_file).read_text(encoding="utf-8"))
    runs = tasks["tasks_order"]

    proc = ROOT / "data" / "processed"
    raw_root = ROOT / "data" / "raw" / "udacity"

    splits = ("train", "val", "test")
    made = []

    for run in runs:
        base_dir = raw_root / run
        assert base_dir.exists(), f"No existe base_dir: {base_dir}"

        paths = tasks["splits"][run]
        for split in splits:
            csv_path = Path(paths[split])
            if not csv_path.exists():
                print(f"[WARN] Falta CSV para {run}/{split}: {csv_path} -> salto")
                continue

            out_dir = proc / run
            out_dir.mkdir(parents=True, exist_ok=True)

            color_tag = "rgb" if args.rgb else "gray"
            gain_tag = (args.gain if args.encoder == "rate" else 0.0)
            out_name = f"{split}_{args.encoder}_T{args.T}_gain{gain_tag}_{color_tag}_{args.w}x{args.h}.h5"
            out_path = out_dir / out_name

            if args.only_missing and out_path.exists():
                print(f"✓ Ya existe, omito: {out_path.name}")
                continue

            if args.overwrite and out_path.exists():
                out_path.unlink(missing_ok=True)

            print(f"[{run}] {split} -> {out_name}")
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
            made.append(str(out_path))

    print("\nHecho. Generados/actualizados:", len(made))
    for p in made[:6]:
        print(" -", p)
    if len(made) > 6:
        print(" ...")
    print("\nRecuerda: el entrenamiento buscará H5 por **atributos**, no por nombre (ver paso 2).")


if __name__ == "__main__":
    main()

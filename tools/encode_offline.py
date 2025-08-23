# tools/encode_offline.py
from __future__ import annotations
import argparse
from pathlib import Path
import sys

# Asegurar raíz del repo en sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Librería que hace el trabajo real
from src.prep.encode_offline import encode_csv_to_h5


def main():
    ap = argparse.ArgumentParser(
        description="Codifica un CSV (train/val/test) a H5 de spikes (offline)."
    )
    ap.add_argument("--csv",        required=True, type=Path, help="Ruta al CSV (split).")
    ap.add_argument("--base-dir",   required=True, type=Path, help="Directorio base de imágenes.")
    ap.add_argument("--out",        required=True, type=Path, help="Ruta de salida .h5.")
    ap.add_argument("--encoder",    required=True, choices=["rate", "latency", "raw"],
                    help="Codificador temporal (image queda fuera de offline).")
    ap.add_argument("--T",          type=int, default=10, help="Ventana temporal.")
    ap.add_argument("--gain",       type=float, default=0.5,
                    help="Solo aplica a encoder=rate (se ignora en latency/raw).")
    ap.add_argument("--w",          type=int, default=200, help="Ancho de reescalado.")
    ap.add_argument("--h",          type=int, default=66,  help="Alto de reescalado.")
    ap.add_argument("--rgb",        action="store_true", help="Por defecto: gris; con --rgb usa color.")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--force",      action="store_true",
                    help="Reescribe si el .h5 ya existe (por defecto: no).")
    args = ap.parse_args()

    csv  = args.csv
    base = args.base_dir
    out  = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not args.force:
        print(f"[SKIP] {out} ya existe (usa --force para sobrescribir).")
        return

    encode_csv_to_h5(
        csv_df_or_path=csv,
        base_dir=base,
        out_path=out,
        encoder=args.encoder,
        T=args.T,
        gain=(args.gain if args.encoder == "rate" else 0.0),
        size_wh=(args.w, args.h),
        to_gray=(not args.rgb),
        seed=args.seed,
    )
    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()

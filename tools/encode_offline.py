# tools/encode_offline.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

# --- asegurar raíz del repo en sys.path ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------
from src.prep.encode_offline import encode_csv_to_h5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", choices=["rate","latency","raw"], default="rate")
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--gain", type=float, default=0.5)
    ap.add_argument("--w", type=int, default=200)
    ap.add_argument("--h", type=int, default=66)
    ap.add_argument("--rgb", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    encode_csv_to_h5(
        csv_df_or_path=Path(args.csv),
        base_dir=Path(args.base_dir),
        out_path=Path(args.out),
        encoder=args.encoder,
        T=args.T,
        gain=args.gain,
        size_wh=(args.w, args.h),
        to_gray=(not args.rgb),
        seed=args.seed,
    )
    print("Guardado:", args.out)

if __name__ == "__main__":
    main()

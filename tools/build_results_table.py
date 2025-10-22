# scripts/build_results_table.py
# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
from src.results_io import build_results_table

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--out_csv", default="outputs/summary/results_table.csv")
    args = ap.parse_args()

    root = Path(args.root)
    df = build_results_table(root)
    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[OK] Tabla creada: {out} ({len(df)} filas)")

if __name__ == "__main__":
    main()

# scripts/plots_per_run.py
# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
from src.plots import make_run_plots

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_subdir", default="plots")
    args = ap.parse_args()
    out = make_run_plots(Path(args.run_dir), args.out_subdir)
    print(f"[OK] Gráficas por-run en: {out}")

if __name__ == "__main__":
    main()

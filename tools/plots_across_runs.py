# scripts/plots_across_runs.py
# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import pandas as pd
from src.plots import plot_across_runs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="outputs/summary/results_table.csv")
    ap.add_argument("--out", default="outputs/summary/plots")
    args = ap.parse_args()

    df = pd.read_csv(args.table)
    outdir = plot_across_runs(df, Path(args.out))
    print(f"[OK] Gráficas comparativas en: {outdir}")

if __name__ == "__main__":
    main()

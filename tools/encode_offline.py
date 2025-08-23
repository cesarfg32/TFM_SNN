# tools/encode_tasks.py (opcional)
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_preset
from src.prep.encode_offline import encode_csv_to_h5
from src.datasets import ImageTransform

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/presets.yaml"))
    ap.add_argument("--preset", required=True, choices=["fast","std","accurate"])
    ap.add_argument("--tasks-file", type=Path, default=Path("data/processed/tasks.json"))
    args = ap.parse_args()

    cfg = load_preset(args.config, args.preset)
    mw, mh, to_gray = cfg["model"]["img_w"], cfg["model"]["img_h"], cfg["model"]["to_gray"]
    T, enc, gain = cfg["data"]["T"], cfg["data"]["encoder"], cfg["data"]["gain"]
    assert enc in {"rate","latency","raw"}, "encode_offline aplica a rate/latency/raw"

    with open(args.tasks_file, "r", encoding="utf-8") as f:
        tasks_json = json.load(f)

    for run in tasks_json["tasks_order"]:
        paths = tasks_json["splits"][run]
        base  = ROOT / "data" / "raw" / "udacity" / run
        outdir= ROOT / "data" / "processed" / run
        outdir.mkdir(parents=True, exist_ok=True)
        for split in ["train","val","test"]:
            csv = Path(paths[split]) if Path(paths[split]).is_absolute() else (ROOT / paths[split])
            out = outdir / f"{split}_{enc}_T{T}_gain{gain if enc=='rate' else 0}_{'gray' if to_gray else 'rgb'}_{mw}x{mh}.h5"
            encode_csv_to_h5(csv_df_or_path=csv, base_dir=base, out_path=out,
                             encoder=enc, T=T, gain=gain, size_wh=(mw, mh),
                             to_gray=to_gray, seed=cfg["data"]["seed"])
            print("OK:", out)
if __name__ == "__main__":
    main()

# tools/encode_tasks.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_preset
from src.prep.encode_offline import encode_csv_to_h5


def main() -> None:
    ap = argparse.ArgumentParser(description="Codifica tasks*.json → H5 (formato oficial v2).")
    ap.add_argument("--preset", required=True, choices=["fast", "std", "accurate"])
    ap.add_argument("--config", default=str(ROOT / "configs" / "presets.yaml"))
    ap.add_argument(
        "--tasks-file",
        default=None,
        help="Override: ruta a tasks.json / tasks_balanced.json",
    )
    ap.add_argument(
        "--only-missing",
        action="store_true",
        default=True,
        help="No sobrescribe si el H5 ya existe (por defecto ON).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Forzar sobrescritura (incompatible con --only-missing).",
    )
    args = ap.parse_args()

    if args.overwrite and args.only_missing:
        raise SystemExit("Usa --overwrite *o* --only-missing, pero no ambos.")

    cfg   = load_preset(Path(args.config), args.preset)
    data  = cfg["data"]
    model = cfg["model"]
    prep  = cfg.get("prep", {})

    mw, mh   = int(model["img_w"]), int(model["img_h"])
    to_gray  = bool(model["to_gray"])
    crop_top    = int(model.get("crop_top", 0) or 0)
    crop_bottom = int(model.get("crop_bottom", 0) or 0)

    enc  = data["encoder"]
    T    = int(data["T"])
    gain = float(data["gain"])

    RAW  = ROOT / "data" / "raw" / "udacity"
    PROC = ROOT / "data" / "processed"

    # Select tasks file
    if args.tasks_file:
        tasks_file = Path(args.tasks_file)
    else:
        tb = PROC / prep.get("tasks_balanced_file_name", "tasks_balanced.json")
        tasks_file = tb if tb.exists() else (PROC / prep.get("tasks_file_name", "tasks.json"))

    tasks = json.loads(tasks_file.read_text(encoding="utf-8"))

    for run in tasks["tasks_order"]:
        paths = tasks["splits"][run]
        base  = RAW / run
        outdir = PROC / run
        outdir.mkdir(parents=True, exist_ok=True)

        for split in ["train", "val", "test"]:
            csv_rel = Path(paths[split])
            csv = csv_rel if csv_rel.is_absolute() else (ROOT / csv_rel)

            suffix_gain = (gain if enc == "rate" else 0)
            color = "gray" if to_gray else "rgb"
            crop_suffix = ""
            if crop_top or crop_bottom:
                crop_suffix = f"_ct{crop_top}_cb{crop_bottom}"

            out = outdir / f"{split}_{enc}_T{T}_gain{suffix_gain}_{color}_{mw}x{mh}{crop_suffix}.h5"

            if args.only_missing and out.exists():
                print(f"✓ Ya existe, omito: {out.name}")
                continue
            if args.overwrite and out.exists():
                out.unlink(missing_ok=True)

            encode_csv_to_h5(
                csv_df_or_path=csv,
                base_dir=base,
                out_path=out,
                encoder=enc,
                T=T,
                gain=gain,
                size_wh=(mw, mh),
                to_gray=to_gray,
                crop_top=crop_top,
                crop_bottom=crop_bottom,
                seed=int(data.get("seed", 42)),
            )
            print("OK:", out)


if __name__ == "__main__":
    main()

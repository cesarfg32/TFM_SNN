from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_preset
from src.prep.data_prep import PrepConfig, run_prep   # tu módulo existente
from src.prep.augment_offline import balance_train_with_augmented_images
from src.prep.encode_offline import encode_csv_to_h5


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Prep offline: splits → (opcional) balanceo por imágenes → "
            "tasks_balanced.json → (opcional) encode H5."
        )
    )
    ap.add_argument("--preset", required=True, choices=["fast", "std", "accurate"])
    ap.add_argument("--config", default=str(ROOT / "configs" / "presets.yaml"))
    ap.add_argument(
        "--encode",
        action="store_true",
        help="Fuerza encode H5 (ignora el flag del preset).",
    )
    args = ap.parse_args()

    cfg   = load_preset(Path(args.config), args.preset)
    prep  = cfg.get("prep", {})
    data  = cfg.get("data", {})
    model = cfg.get("model", {})

    RAW  = ROOT / "data" / "raw" / "udacity"
    PROC = ROOT / "data" / "processed"

    # --- 1) QC + SPLITS -------------------------------------------------------
    if prep.get("runs"):
        runs = list(prep["runs"])
    else:
        runs = sorted([
            d.name for d in RAW.iterdir()
            if d.is_dir() and any(p.name == "driving_log.csv" for p in d.rglob("driving_log.csv"))
        ])

    pcfg = PrepConfig(
        root=ROOT,
        runs=runs,
        use_left_right=bool(prep.get("use_left_right", True)),
        steer_shift=float(prep.get("steer_shift", 0.2)),
        bins=int(prep.get("bins", 50)),
        train=float(prep.get("train", 0.70)),
        val=float(prep.get("val", 0.15)),
        seed=int(prep.get("seed", 42)),
        target_per_bin=None,
        cap_per_bin=None,
        merge_subruns=bool(prep.get("merge_subruns", True)),
    )
    manifest = run_prep(pcfg)
    tasks_json_path = PROC / prep.get("tasks_file_name", "tasks.json")
    print("OK SPLITS:", tasks_json_path)

    # --- 2) BALANCEO OFFLINE por IMÁGENES (opcional) --------------------------
    bal = prep.get("balance_offline", {})
    if bal.get("mode", "none") == "images":
        stats_all = {}
        for run in runs:
            base_dir  = RAW / run
            out_dir   = PROC / run
            train_csv = out_dir / "train.csv"

            out_csv, stats = balance_train_with_augmented_images(
                train_csv=train_csv,
                raw_run_dir=base_dir,
                out_run_dir=out_dir,
                bins=int(prep.get("bins", 50)),
                target_per_bin=bal.get("target_per_bin", "auto"),
                cap_per_bin=bal.get("cap_per_bin", None),
                seed=int(prep.get("seed", 42)),
                aug=bal.get("aug", {}),
            )
            stats_all[run] = stats
            print(f"[{run}] +{stats['generated']} nuevas → {out_csv.name}")

        tb = {"tasks_order": runs, "splits": {}}
        for run in runs:
            d = str((PROC / run).resolve())
            tb["splits"][run] = {
                "train": f"{d}/train_balanced.csv",
                "val":   f"{d}/val.csv",
                "test":  f"{d}/test.csv",
            }
        tasks_balanced_path = PROC / prep.get("tasks_balanced_file_name", "tasks_balanced.json")
        tasks_balanced_path.write_text(json.dumps(tb, indent=2), encoding="utf-8")
        print("OK BALANCED:", tasks_balanced_path)

        (PROC / "prep_manifest_aug.json").write_text(json.dumps(stats_all, indent=2), encoding="utf-8")

    # --- 3) ENCODE H5 (opcional) ---------------------------------------------
    want_encode_cfg = bool(prep.get("encode_h5", False))
    want_encode = args.encode or want_encode_cfg
    if want_encode:
        use_balanced_cfg = bool(prep.get("use_balanced_tasks", False))
        tb_name = prep.get("tasks_balanced_file_name", "tasks_balanced.json")
        t_name  = prep.get("tasks_file_name", "tasks.json")
        cand_bal = PROC / tb_name
        tasks_file = cand_bal if (use_balanced_cfg and cand_bal.exists()) else (PROC / t_name)
        print("ENCODE H5 desde:", tasks_file.name)

        mw, mh   = int(model.get("img_w", 200)), int(model.get("img_h", 66))
        to_gray  = bool(model.get("to_gray", True))
        crop_top    = int(model.get("crop_top", 0) or 0)
        crop_bottom = int(model.get("crop_bottom", 0) or 0)

        enc_mode = data.get("encoder", "rate")   # rate | latency | raw
        T        = int(data.get("T", 10))
        gain     = float(data.get("gain", 0.5))
        seed     = int(data.get("seed", 42))

        tasks = json.loads(tasks_file.read_text(encoding="utf-8"))
        for run in tasks["tasks_order"]:
            paths  = tasks["splits"][run]
            base   = RAW / run
            outdir = PROC / run
            outdir.mkdir(parents=True, exist_ok=True)

            for split in ("train", "val", "test"):
                csv_path = Path(paths[split])
                if not csv_path.is_absolute():
                    csv_path = ROOT / csv_path

                suffix_gain = (gain if enc_mode == "rate" else 0)
                color = "gray" if to_gray else "rgb"
                crop_suffix = ""
                if crop_top or crop_bottom:
                    crop_suffix = f"_ct{crop_top}_cb{crop_bottom}"

                out = outdir / f"{split}_{enc_mode}_T{T}_gain{suffix_gain}_{color}_{mw}x{mh}{crop_suffix}.h5"

                encode_csv_to_h5(
                    csv_df_or_path=csv_path,
                    base_dir=base,
                    out_path=out,
                    encoder=enc_mode,
                    T=T,
                    gain=gain,
                    size_wh=(mw, mh),
                    to_gray=to_gray,
                    crop_top=crop_top,
                    crop_bottom=crop_bottom,
                    seed=seed,
                )
                print("OK H5:", out)


if __name__ == "__main__":
    main()

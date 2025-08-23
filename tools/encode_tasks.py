# tools/encode_tasks.py
from __future__ import annotations
import argparse, json, os, hashlib, tempfile
from pathlib import Path
import sys
from datetime import datetime, timezone

import h5py  # pip install h5py

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_preset
from src.prep.encode_offline import encode_csv_to_h5  # tu función de codificación


# --------------------- utilidades manifest ---------------------
def _file_sha256(path: Path, chunk=1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _read_h5_attrs(p: Path) -> dict:
    out = {}
    try:
        with h5py.File(p, "r") as h5:
            out = {k: h5.attrs[k] for k in h5.attrs.keys()}
            out["_N"] = int(h5["spikes"].shape[0]) if "spikes" in h5 else None
    except Exception as e:
        out["_error"] = str(e)
    norm = {}
    for k, v in out.items():
        try:
            norm[k] = v.item() if hasattr(v, "item") else v
        except Exception:
            norm[k] = str(v)
    return norm

def update_prep_manifest(
    manifest_path: Path,
    *,
    out_h5: Path,
    source_csv: Path,
    encoder: str,
    T: int,
    gain: float,
    size_wh: tuple[int, int],
    to_gray: bool,
    cmdline: str,
) -> None:
    """Upsert por clave = ruta del .h5. Escritura atómica."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    attrs = _read_h5_attrs(out_h5)
    now_iso = datetime.now(timezone.utc).isoformat()

    record = {
        "out": str(out_h5),
        "created_utc": now_iso,
        "sha256": _file_sha256(out_h5) if out_h5.exists() else None,
        "size_bytes": os.path.getsize(out_h5) if out_h5.exists() else None,
        "source_csv": str(source_csv),
        "encoder": str(encoder),
        "T": int(T),
        "gain": float(gain),
        "size_wh": [int(size_wh[0]), int(size_wh[1])],
        "to_gray": bool(to_gray),
        "h5_attrs": attrs,
        "cmdline": cmdline,
    }

    data = {"version": 1, "entries": {}}
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(data.get("entries", {}), dict):
                data = {"version": 1, "entries": {}}
        except Exception:
            data = {"version": 1, "entries": {}}

    data["entries"][str(out_h5)] = record  # upsert

    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, manifest_path)
    print(f"[manifest] actualizado: {manifest_path} (clave: {out_h5.name})")


# --------------------- script principal ---------------------
def _fmt_gain(g: float) -> str:
    s = f"{g:.6g}"
    try:
        s = str(float(s))
    except Exception:
        pass
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/presets.yaml"))
    ap.add_argument("--preset", required=True, choices=["fast", "std", "accurate"])
    ap.add_argument("--tasks-file", type=Path, default=Path("data/processed/tasks.json"))
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-codifica aunque el .h5 ya exista (por defecto, se salta).")
    ap.add_argument("--manifest-scope", choices=["per-run", "global"], default="per-run",
                    help="Dónde guardar prep_manifest.json (por circuito o global).")
    args = ap.parse_args()

    cfg = load_preset(args.config, args.preset)
    mw, mh = int(cfg["model"]["img_w"]), int(cfg["model"]["img_h"])
    to_gray = bool(cfg["model"]["to_gray"])

    T = int(cfg["data"]["T"])
    enc = str(cfg["data"]["encoder"])
    gain = float(cfg["data"]["gain"])
    seed = int(cfg["data"]["seed"])
    assert enc in {"rate", "latency", "raw"}, "encode_offline aplica a rate/latency/raw"

    with open(args.tasks_file, "r", encoding="utf-8") as f:
        tasks_json = json.load(f)

    for run in tasks_json["tasks_order"]:
        paths = tasks_json["splits"][run]
        base = ROOT / "data" / "raw" / "udacity" / run
        outdir = ROOT / "data" / "processed" / run
        outdir.mkdir(parents=True, exist_ok=True)

        manifest_path = (outdir / "prep_manifest.json") if (args.manifest_scope == "per-run") \
                        else (ROOT / "data" / "processed" / "prep_manifest.json")

        for split in ["train", "val", "test"]:
            csv_rel_or_abs = Path(paths[split])
            csv = csv_rel_or_abs if csv_rel_or_abs.is_absolute() else (ROOT / csv_rel_or_abs)

            parts = [split, enc, f"T{T}"]
            if enc == "rate":
                parts.append(f"gain{_fmt_gain(gain)}")
            parts.append("gray" if to_gray else "rgb")
            parts.append(f"{mw}x{mh}")
            out_name = "_".join(parts) + ".h5"

            out = outdir / out_name

            if out.exists() and not args.overwrite:
                print(f"SKIP (ya existe): {out}")
            else:
                encode_csv_to_h5(
                    csv_df_or_path=csv,
                    base_dir=base,
                    out_path=out,
                    encoder=enc,
                    T=T,
                    gain=(gain if enc == "rate" else 0.0),
                    size_wh=(mw, mh),
                    to_gray=to_gray,
                    seed=seed,
                )
                print("OK:", out)

            if out.exists():
                update_prep_manifest(
                    manifest_path,
                    out_h5=out,
                    source_csv=csv,
                    encoder=enc,
                    T=T,
                    gain=(gain if enc == "rate" else 0.0),
                    size_wh=(mw, mh),
                    to_gray=to_gray,
                    cmdline=" ".join(sys.argv),
                )
            else:
                print(f"[WARN] no se pudo registrar en manifest (no existe): {out}")

if __name__ == "__main__":
    main()

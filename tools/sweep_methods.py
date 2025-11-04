# tools/sweep_methods.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, time, json, argparse, traceback, gc
from pathlib import Path
from typing import Any, Dict, List
import multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# ----------------- EXPS embebido (puedes editar aquí) -----------------
DEFAULT_EXPS: List[Dict[str, Any]] = [
    # ---------- SCA-SNN ----------
    dict(
        method="sca-snn",
        params={"attach_to":"f6","flatten_spatial":False,"num_bins":50,"anchor_batches":16,
                "beta":0.55,"bias":0.00,"soft_mask_temp":0.30,"verbose":False,"log_every":65536},
        tag="sca_looser_b055_bias000_t030_ab16"
    ),
    dict(
        method="sca-snn",
        params={"attach_to":"f6","flatten_spatial":False,"num_bins":80,"anchor_batches":16,
                "beta":0.65,"bias":0.05,"soft_mask_temp":0.00,"verbose":False,"log_every":65536},
        tag="sca_hard_b065_bias005_t000_bins80"
    ),
    dict(
        method="sca-snn",
        params={"attach_to":"f6","flatten_spatial":False,"num_bins":50,"anchor_batches":24,
                "beta":0.60,"bias":0.10,"soft_mask_temp":0.50,"verbose":False,"log_every":65536},
        tag="sca_moreanchors_b060_bias010_t050_ab24"
    ),

    # ---------- SA-SNN ----------
    dict(
        method="sa-snn",
        params={"attach_to":"f6","k":8,"tau":28.0,"th_min":1.0,"th_max":2.0,
                "p":2_000_000,"vt_scale":1.0,"flatten_spatial":False,
                "assume_binary_spikes":False,"reset_counters_each_task":False},
        tag="sa_ref_k8_tau28_p2m"
    ),
    dict(
        method="sa-snn",
        params={"attach_to":"f6","k":6,"tau":24.0,"th_min":1.0,"th_max":2.0,
                "p":2_000_000,"vt_scale":1.2,"flatten_spatial":False,
                "assume_binary_spikes":False,"reset_counters_each_task":False},
        tag="sa_k6_tau24_vt1p2"
    ),
    dict(
        method="sa-snn",
        params={"attach_to":"f6","k":10,"tau":32.0,"th_min":1.0,"th_max":2.0,
                "p":5_000_000,"vt_scale":1.0,"flatten_spatial":False,
                "assume_binary_spikes":False,"reset_counters_each_task":False},
        tag="sa_k10_tau32_p5m"
    ),
    dict(
        method="sa-snn",
        params={"attach_to":"f6","k":8,"tau":28.0,"th_min":1.0,"th_max":2.0,
                "p":2_000_000,"vt_scale":1.0,"flatten_spatial":True,
                "assume_binary_spikes":False,"reset_counters_each_task":False},
        tag="sa_k8_tau28_flatten"
    ),

    # ---------- AS-SNN ----------
    dict(
        method="as-snn",
        params={"gamma_ratio":0.30,"lambda_a":1.59168,"ema":0.90,
                "attach_to":"f6","measure_at":"modules","penalty_mode":"l1",
                "do_synaptic_scaling":False},
        tag="as_ref_gr03_lam1p59168"
    ),
    dict(
        method="as-snn",
        params={"gamma_ratio":0.25,"lambda_a":1.20,"ema":0.90,
                "attach_to":"f6","measure_at":"modules","penalty_mode":"l1",
                "do_synaptic_scaling":False},
        tag="as_soft_gr025_lam1p20"
    ),
    dict(
        method="as-snn",
        params={"gamma_ratio":0.35,"lambda_a":1.80,"ema":0.95,
                "attach_to":"f6","measure_at":"modules","penalty_mode":"l1",
                "do_synaptic_scaling":True,"scale_clip":(0.5,2.0),"scale_bias":False},
        tag="as_scaling_gr035_lam1p80_ema095"
    ),

    # ---------- baseline ----------
    dict(method="naive", params={}, tag="baseline_naive"),
]

# ----------------- worker: un run por PROCESO -----------------
def worker_run(exp: Dict[str, Any], preset: str, out_root: str, safe_dl: int, amp: str) -> str:
    """Ejecuta un experimento en un subproceso y devuelve la ruta del run."""
    import torch
    from src.utils import load_preset, build_task_list_for, build_components_for
    from src.runner import run_continual

    # Config
    cfg = load_preset(ROOT / "configs" / "presets.yaml", preset)
    cfg["continual"]["method"]  = exp["method"]
    cfg["continual"]["params"]  = exp["params"]
    cfg.setdefault("naming", {})
    cfg["naming"]["tag"] = exp["tag"]

    # Dataloader "seguro" si se pide
    # 0: no tocar; 1: desactiva persistent_workers; 2: además num_workers=0 y pin_memory=False
    if safe_dl >= 1:
        cfg["data"]["persistent_workers"] = False
    if safe_dl >= 2:
        cfg["data"]["num_workers"] = 0
        cfg["data"]["pin_memory"] = False

    # AMP override
    if amp.lower() == "on":
        cfg["optim"]["amp"] = True
    elif amp.lower() == "off":
        cfg["optim"]["amp"] = False
    # "auto" => deja lo del preset

    # Build
    tfm, make_loader_fn, make_model_fn = build_components_for(cfg, ROOT)
    task_list, _ = build_task_list_for(cfg, ROOT)

    # Info mínima en consola
    print(f"\n[RUN] preset={preset} | method={exp['method']} | tag={exp['tag']}")
    print(f"[RUN] safe_dl={safe_dl} | amp={cfg['optim'].get('amp', None)}")

    out_dir, _ = run_continual(
        task_list=task_list,
        make_loader_fn=make_loader_fn,
        make_model_fn=make_model_fn,
        tfm=tfm,
        cfg=cfg,
        preset_name=preset,
        out_root=Path(out_root),
        verbose=True,
    )
    # Limpieza explícita antes de salir del hijo
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()
    return str(out_dir)

def main():
    ap = argparse.ArgumentParser("Sweep de métodos (un proceso por run)")
    ap.add_argument("--preset", default="accurate", help="Nombre de preset en configs/presets.yaml")
    ap.add_argument("--sweep-file", default="", help="JSON con {'exps':[...]} (opcional)")
    ap.add_argument("--only-methods", default="", help="Filtro por método: ej. 'sca-snn,sa-snn'")
    ap.add_argument("--start", type=int, default=0, help="Índice inicial (inclusive) en la lista de EXPS")
    ap.add_argument("--end",   type=int, default=-1, help="Índice final (inclusive); -1 hasta el final")
    ap.add_argument("--out", default=str(ROOT / "outputs"), help="Carpeta de salida")
    ap.add_argument("--sleep", type=float, default=1.0, help="Pausa entre runs (seg)")
    ap.add_argument("--safe-dataloader", type=int, default=1, choices=[0,1,2],
                    help="0=sin tocar; 1=no persistentes; 2=workers 0 + pin_memory False")
    ap.add_argument("--amp", default="auto", choices=["auto","on","off"], help="Forzar AMP on/off o dejarlo en preset")
    ap.add_argument("--dry", action="store_true", help="Solo listar qué se ejecutaría")
    args = ap.parse_args()

    # Carga EXPS
    exps: List[Dict[str, Any]]
    if args.sweep_file:
        data = json.loads(Path(args.sweep_file).read_text(encoding="utf-8"))
        exps = data.get("exps", [])
    else:
        exps = DEFAULT_EXPS

    if args.only_methods:
        keep = set(s.strip().lower() for s in args.only_methods.split(",") if s.strip())
        exps = [e for e in exps if str(e.get("method","")).lower() in keep]

    if args.end >= 0:
        exps = exps[args.start: args.end+1]
    else:
        exps = exps[args.start:]

    print(f"[INFO] {len(exps)} runs a ejecutar. OUT={args.out}")
    for i, e in enumerate(exps, 1):
        print(f"  {i:02d}. {e['method']} :: {e.get('tag','')}")

    if args.dry:
        return

    # Asegura spawn (más seguro para CUDA)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    for i, exp in enumerate(exps, 1):
        print(f"\n=== [{i}/{len(exps)}] {exp['method']} | tag={exp.get('tag','')} ===")
        p_conn, c_conn = mp.Pipe(duplex=False)
        proc = mp.Process(
            target=_child_entry,
            args=(exp, args.preset, args.out, args.safe_dataloader, args.amp, c_conn),
            daemon=False,
        )
        proc.start()
        c_conn.close()
        run_dir = None
        err = None
        try:
            # Espera resultado (run_dir o error)
            msg = p_conn.recv()
            if isinstance(msg, dict) and msg.get("ok"):
                run_dir = msg.get("run_dir")
                print("[OK] run_dir:", run_dir)
            else:
                err = msg.get("error", "Unknown error")
        except EOFError:
            err = "Proceso hijo terminó sin enviar resultado."
        finally:
            proc.join()
            p_conn.close()

        if err:
            print(f"[ERROR] {err}")
        time.sleep(args.sleep)

def _child_entry(exp, preset, out_root, safe_dl, amp, conn):
    try:
        run_dir = worker_run(exp, preset, out_root, safe_dl, amp)
        conn.send({"ok": True, "run_dir": run_dir})
    except Exception as e:
        tb = traceback.format_exc()
        conn.send({"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": tb})
    finally:
        try:
            conn.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()

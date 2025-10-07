# src/telemetry.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import json, time, sys, platform

try:
    from codecarbon import EmissionsTracker  # opcional
except Exception:
    EmissionsTracker = None

import torch


def system_snapshot() -> dict:
    """Snapshot ligero de entorno/hardware para adjuntar a los logs."""
    gpu = None
    if torch.cuda.is_available():
        try:
            gpu = torch.cuda.get_device_name(0)
        except Exception:
            gpu = "cuda_available"
    return {
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", None),
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu_name": gpu,
        "platform": platform.platform(),
        "tf32_matmul": getattr(torch.backends.cuda.matmul, "allow_tf32", False),
        "cudnn_benchmark": getattr(torch.backends.cudnn, "benchmark", False),
    }


def log_telemetry_event(out_dir: Path | str, data: dict) -> None:
    """Append JSONL event en outputs/<run>/telemetry.jsonl"""
    p = Path(out_dir) / "telemetry.jsonl"
    rec = {"ts": time.time(), **data}
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


@contextmanager
def carbon_tracker_ctx(
    out_dir: Path | str,
    project_name: str,
    offline: bool = True,
    country_iso_code: str | None = None,
    measure_power_secs: int = 15,  # mayor = menos overhead
):
    """
    Contexto seguro: si CodeCarbon no está, hace no-op.
    Al salir, expone .final_emissions (kgCO2eq) si aplica.
    """
    class _Dummy:
        final_emissions = None
        def stop(self): return None

    if EmissionsTracker is None:
        yield _Dummy()
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cc_dir = out_dir / "codecarbon"
    cc_dir.mkdir(parents=True, exist_ok=True)

    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir=str(cc_dir),
        measure_power_secs=measure_power_secs,
        save_to_file=True,     # guarda emissions.csv
        offline=offline,       # evita llamadas a red
        country_iso_code=country_iso_code,  # e.g., "ESP"; si None, CodeCarbon intenta inferir
    )
    tracker.start()
    try:
        yield tracker
    finally:
        try:
            emissions = tracker.stop()  # kgCO2eq del contexto
        except Exception:
            emissions = None
        # adjunta para lectura sencilla
        try:
            tracker.final_emissions = emissions
        except Exception:
            pass


def read_emissions_kg(out_dir: Path | str) -> float | None:
    """
    Suma 'emissions' de codecarbon/emissions.csv si existe.
    No requiere pandas.
    """
    import csv
    p = Path(out_dir) / "codecarbon" / "emissions.csv"
    if not p.exists():
        return None
    total = 0.0
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                total += float(row.get("emissions", 0.0))
            except Exception:
                pass
    return total if total > 0 else None

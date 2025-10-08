# src/telemetry.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
import os
import platform
import time
import inspect
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

# Intento opcional de importar CodeCarbon
try:
    from codecarbon import EmissionsTracker  # type: ignore
except Exception:
    EmissionsTracker = None  # fallback sin dependencia


# -----------------------------
# Utilidades de logging simple
# -----------------------------
def _write_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def log_telemetry_event(out_dir: Path | str, payload: Dict[str, Any]) -> None:
    out_dir = Path(out_dir)
    payload = dict({"ts": time.time()}, **payload)
    _write_jsonl(out_dir / "telemetry.jsonl", payload)


def system_snapshot() -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch  # lazy import
        snap.update(
            {
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
            }
        )
        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            snap.update(
                {
                    "gpu_name": torch.cuda.get_device_name(i),
                    "gpu_capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
                    "gpu_mem_total_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                }
            )
    except Exception:
        pass
    return snap


# ----------------------------------------------------
# Context manager de CodeCarbon (o no-op si no existe)
# ----------------------------------------------------
def carbon_tracker_ctx(
    out_dir: Path | str,
    project_name: str,
    offline: bool = True,
    country_iso_code: Optional[str] = None,
    measure_power_secs: int = 15,
    log_level: str = "warning",
):
    """
    Devuelve un context manager. Si CodeCarbon está instalado -> EmissionsTracker.
    Si no, devuelve un no-op con atributo .final_emissions = None.
    Soporta versiones antiguas/nuevas (filtra kwargs incompatibles).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fallback si no hay CodeCarbon
    if EmissionsTracker is None:
        @contextmanager
        def _noop():
            class _Dummy:
                final_emissions = None  # kg
            yield _Dummy()
        return _noop()

    # Construimos kwargs compatibles con la versión instalada
    base_kwargs: Dict[str, Any] = {
        "project_name": project_name,
        "output_dir": str(out_dir),
        "save_to_file": True,            # genera emissions.csv
        "measure_power_secs": measure_power_secs,
        "log_level": log_level,
    }
    if country_iso_code:
        base_kwargs["country_iso_code"] = country_iso_code

    # Inspecciona la firma de __init__ para filtrar & mapear
    try:
        sig = inspect.signature(EmissionsTracker.__init__)
        params = set(sig.parameters.keys())
    except Exception:
        params = set()

    # Algunas versiones aceptan 'offline'; otras usan 'save_to_api'
    if "offline" in params:
        base_kwargs["offline"] = bool(offline)
    elif "save_to_api" in params:
        # Offline => no intentes subir a la API
        base_kwargs["save_to_api"] = False

    # Filtra solo los kwargs soportados por esta versión
    filtered = {k: v for k, v in base_kwargs.items() if (not params or k in params)}

    tracker = EmissionsTracker(**filtered)

    @contextmanager
    def _ctx():
        tracker.start()
        try:
            yield tracker
        finally:
            try:
                final = tracker.stop()  # kg CO2e (o None)
            except Exception:
                final = None
            try:
                setattr(tracker, "final_emissions", final)
            except Exception:
                pass

    return _ctx()


# ----------------------------------------------------
# Lectura de emisiones desde emissions.csv (CodeCarbon)
# ----------------------------------------------------
def read_emissions_kg(out_dir: Path | str) -> Optional[float]:
    """
    Busca emissions.csv en out_dir. Devuelve la última columna 'emissions' (kg) si existe.
    Robusto a encabezados con distinto orden.
    """
    csv_path = Path(out_dir) / "emissions.csv"
    if not csv_path.exists():
        return None

    try:
        last_val: Optional[float] = None
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            field = None
            for c in reader.fieldnames or []:
                if c.lower().strip() == "emissions":
                    field = c
                    break
            if field is None:
                return None
            for row in reader:
                v = row.get(field, "")
                try:
                    last_val = float(v)
                except Exception:
                    continue
        return last_val
    except Exception:
        return None

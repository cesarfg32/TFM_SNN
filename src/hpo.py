# src/hpo.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from typing import Mapping

# Nota: extract_metrics vive en utils_exp y devuelve "forget_rel_%"
# Lo dejamos separado para que los notebooks puedan importarlo desde utils_exp.
# from src.utils_exp import extract_metrics  # (si quieres re-exportarlo, descomenta)

def _get_metric(metrics: Mapping[str, float], key: str) -> float:
    """Lee una métrica del dict con compatibilidad legacy."""
    # Compatibilidad: si piden "c1_forgetting_mae_rel_%" usa "forget_rel_%"
    if key == "c1_forgetting_mae_rel_%" and "forget_rel_%" in metrics:
        key = "forget_rel_%"
    try:
        return float(metrics.get(key, math.nan))
    except Exception:
        return math.nan

def objective_value(metrics: Mapping[str, float],
                    key: str = "c1_forgetting_mae_rel_%",
                    mode: str = "min") -> float:
    """
    Escalar objetivo a partir de UNA métrica.
    Default (legacy): minimiza % de olvido en Tarea1.
    Usa _get_metric para mapear "c1_forgetting_mae_rel_%" -> "forget_rel_%".
    """
    val = _get_metric(metrics, key)
    if math.isnan(val):
        return math.inf if mode == "min" else -math.inf
    return val if mode == "min" else -val

def composite_objective(metrics: Mapping[str, float], alpha: float = 0.5) -> float:
    """
    Objetivo compuesto: c2_mae + alpha * max(0, forget_rel_%).
    - Menor es mejor
    - Sencillo y trazable para el TFM: balancea rendimiento final y olvido.
    """
    m2 = _get_metric(metrics, "c2_mae")
    f  = _get_metric(metrics, "forget_rel_%")  # nombre canónico de utils_exp
    if math.isnan(m2) or math.isnan(f):
        return math.inf
    return float(m2 + alpha * max(0.0, f))

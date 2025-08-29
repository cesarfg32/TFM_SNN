# src/hpo.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from src.utils_exp import extract_metrics
from pathlib import Path
from typing import Dict, Any, Mapping, Tuple
import json
import math

def objective_value(metrics: Mapping[str, float],
                    key: str = "c1_forgetting_mae_rel_%",
                    mode: str = "min") -> float:
    """
    Devuelve el escalar de objetivo para HPO a partir del dict de métricas.
    Por defecto minimiza el % de olvido en Tarea1.
      - key: métrica a usar
      - mode: 'min' o 'max'
    En caso de NaN: devuelve +inf para 'min' y -inf para 'max' (empeora el score).
    """
    val = float(metrics.get(key, math.nan))
    if math.isnan(val):
        return math.inf if mode == "min" else -math.inf
    return val if mode == "min" else -val

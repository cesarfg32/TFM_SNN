#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.methods.registry import build_method

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(8,16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16,32,3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(32,1)
    def forward(self, x):
        if x.ndim==5: x=x.mean(0)
        return self.fc(self.conv(x).flatten(1))

if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Tiny().to(dev)
    loss = nn.MSELoss()

    cases = [
        ("naive", {}),
        ("ewc", {"lam": 1e6, "fisher_batches": 3}),
        ("rehearsal", {"buffer_size":128, "replay_ratio":0.25}),
        ("sa-snn", {"attach_to":"fc","k":8,"tau":28,"vt_scale":1.33,"p":2_000_000,
                    "th_min":1.0,"th_max":2.0,"flatten_spatial":False}),
        ("as-snn", {"attach_to":"conv.2","gamma_ratio":0.4,"lambda_a":0.5,"measure_at":"modules"}),
        ("sca-snn", {"attach_to":"fc","num_bins":10,"beta":0.55,"soft_mask_temp":0.3,"anchor_batches":2}),
        ("ewc+sca-snn", {"lam":1e6,"fisher_batches":3,"attach_to":"fc","num_bins":10,"beta":0.4}),
    ]

    for name, params in cases:
        try:
            m = build_method(name, model, loss_fn=loss, device=dev, **params)
            print(f"[OK] build_method('{name}') -> {m.name}")
        except Exception as e:
            print(f"[FAIL] build_method('{name}') lanzó excepción: {e}")

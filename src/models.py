# -*- coding: utf-8 -*-
"""Backbone SNN para regresión de dirección (steering).

Diseño híbrido ligero:
- Front-end CNN no-spiking (reduce dimensionalidad de imagen).
- Capa spiking LIF con surrogate gradient (snnTorch >= 0.9 usa spike_grad).
- Decodificador lineal a escalar (ángulo de dirección).
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNNVisionRegressor(nn.Module):
    def __init__(self, in_channels: int = 1, lif_beta: float = 0.95):
        super().__init__()

        # Front-end CNN (ANN)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),  # (H/2, W/2)
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),           # (H/4, W/4)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # (H/8, W/8)
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((5, 10)),                                   # tamaño fijo
        )
        self.flat_dim = 64 * 5 * 10

        # Neurona spiking LIF (API nueva: spike_grad)
        self.lif = snn.Leaky(beta=lif_beta, spike_grad=surrogate.fast_sigmoid())
        self.fc = nn.Linear(self.flat_dim, 128)
        self.readout = nn.Linear(128, 1)  # salida escalar (steering)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward temporal.

        Args:
            x: Tensor (T, B, C, H, W) con spikes (0/1) o intensidades [0,1].

        Returns:
            y_hat: Tensor (B, 1) con la predicción final (promedio temporal).
        """
        T, B, C, H, W = x.shape
        preds = []

        # Estado de membrana explícito (sin .reset()):
        mem = torch.zeros(B, 128, device=x.device, dtype=x.dtype)

        for t in range(T):
            xt = x[t]                      # (B, C, H, W)
            ft = self.features(xt)         # (B, 64, 5, 10)
            ft = ft.flatten(1)             # (B, flat_dim)
            cur = self.fc(ft)              # (B, 128)
            spk, mem = self.lif(cur, mem)  # actualiza estado con Leaky LIF
            yt = self.readout(mem)         # (B, 1)
            preds.append(yt)

        y_hat = torch.stack(preds, dim=0).mean(0)  # promedio temporal
        return y_hat
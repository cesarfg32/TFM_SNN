# src/models.py
# # -*- coding: utf-8 -*-
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



# ============= PilotNet ANN (dinámico, recomendado 200x66) =============
class PilotNetANN(nn.Module):
    def __init__(self, in_channels: int = 1, input_hw: tuple[int,int] = (66,200)):
        super().__init__()
        self.input_hw = tuple(input_hw)  # (H,W)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),          nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),          nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),          nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),          nn.ReLU(inplace=True),
        )
        flat_dim = self._infer_flat_dim(in_channels, *self.input_hw)
        if self.input_hw != (66,200):
            print(f"[PilotNetANN] Aviso: tamaño {self.input_hw} (el clásico es 66x200) → flat_dim={flat_dim}")
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 100), nn.ReLU(inplace=True),
            nn.Linear(100, 50),       nn.ReLU(inplace=True),
            nn.Linear(50, 10),        nn.ReLU(inplace=True),
            nn.Linear(10, 1),
        )

    def _infer_flat_dim(self, C, H, W) -> int:
        with torch.no_grad():
            z = torch.zeros(1, C, H, W)
            z = self.conv(z)
            return int(z.numel() // z.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T,B,C,H,W) o (B,C,H,W)
        if x.dim() == 4:
            x = x.unsqueeze(0)
        T,B,_,H,W = x.shape
        if (H,W) != self.input_hw:
            raise ValueError(f"PilotNetANN espera {self.input_hw}, recibido {(H,W)}")
        preds = []
        for t in range(T):
            z = self.conv(x[t]).flatten(1)
            y = self.fc(z)
            preds.append(y)
        return torch.stack(preds, 0).mean(0)

# ============= PilotNet SNN (dinámico, recomendado 200x66) =============
class PilotNetSNN(nn.Module):
    def __init__(self, in_channels: int = 1,
                 input_hw: tuple[int,int] = (66,200),
                 beta: float = 0.9, threshold: float = 0.5,
                 learn_beta: bool = True,
                 spike_grad = surrogate.fast_sigmoid()):
        super().__init__()
        self.input_hw = tuple(input_hw)
        self.c1 = nn.Conv2d(in_channels, 24, kernel_size=5, stride=2)
        self.l1 = snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=spike_grad)

        self.c2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.l2 = snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=spike_grad)

        self.c3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.l3 = snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=spike_grad)

        self.c4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.l4 = snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=spike_grad)

        self.c5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.l5 = snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=spike_grad)

        flat_dim = self._infer_flat_dim(in_channels, *self.input_hw)
        if self.input_hw != (66,200):
            print(f"[PilotNetSNN] Aviso: tamaño {self.input_hw} (clásico 66x200) → flat_dim={flat_dim}")

        self.f6 = nn.Linear(flat_dim, 100); self.l6 = snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=spike_grad)
        self.f7 = nn.Linear(100, 50);       self.l7 = snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=spike_grad)
        self.f8 = nn.Linear(50, 10);        self.l8 = snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=spike_grad)
        self.out = nn.Linear(10, 1)

    def _infer_flat_dim(self, C, H, W) -> int:
        with torch.no_grad():
            z = torch.zeros(1, C, H, W)
            for (c,l) in [(self.c1,self.l1),(self.c2,self.l2),(self.c3,self.l3),(self.c4,self.l4),(self.c5,self.l5)]:
                z = c(z); m = torch.zeros_like(z); s,_ = l(z,m); z = s
            return int(z.numel() // z.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(0)
        T,B,_,H,W = x.shape
        if (H,W) != self.input_hw:
            raise ValueError(f"PilotNetSNN espera {self.input_hw}, recibido {(H,W)}")
        dev, dtype = x.device, x.dtype
        m1=m2=m3=m4=m5=None
        mem6 = torch.zeros(B,100, device=dev, dtype=dtype)
        mem7 = torch.zeros(B,50,  device=dev, dtype=dtype)
        mem8 = torch.zeros(B,10,  device=dev, dtype=dtype)
        preds=[]
        for t in range(T):
            z = self.c1(x[t]);  m1 = torch.zeros_like(z) if m1 is None else m1; s, m1 = self.l1(z, m1)
            z = self.c2(s);     m2 = torch.zeros_like(z) if m2 is None else m2; s, m2 = self.l2(z, m2)
            z = self.c3(s);     m3 = torch.zeros_like(z) if m3 is None else m3; s, m3 = self.l3(z, m3)
            z = self.c4(s);     m4 = torch.zeros_like(z) if m4 is None else m4; s, m4 = self.l4(z, m4)
            z = self.c5(s);     m5 = torch.zeros_like(z) if m5 is None else m5; s, m5 = self.l5(z, m5)
            ft = s.flatten(1)
            cur6 = self.f6(ft); sp6, mem6 = self.l6(cur6, mem6)
            cur7 = self.f7(mem6); sp7, mem7 = self.l7(cur7, mem7)
            cur8 = self.f8(mem7); sp8, mem8 = self.l8(cur8, mem8)
            y = self.out(mem8)
            preds.append(y)
        return torch.stack(preds, 0).mean(0)

# ============= helpers para elegir modelo y tfm por nombre =============

def default_tfm_for_model(name: str, *, to_gray: bool = True):
    """Devuelve un ImageTransform adecuado por modelo."""
    from src.datasets import ImageTransform
    n = name.lower()
    if n in {"pilotnet_ann","pilotnet_snn"}:
        # clásico: 200x66
        return ImageTransform(200, 66, to_gray, None)
    # por defecto, el de SNNVision (160x80)
    return ImageTransform(160, 80, to_gray, None)

def build_model(name: str, tfm, **kwargs) -> nn.Module:
    """Factory central para el runner y notebooks."""
    C = 1 if getattr(tfm, "to_gray", True) else 3
    H, W = tfm.h, tfm.w  # ojo: en tu ImageTransform guardas self.h/self.w
    n = name.lower()
    if n == "snn_vision":
        return SNNVisionRegressor(in_channels=C, lif_beta=kwargs.pop("lif_beta", 0.95))
    if n == "pilotnet_ann":
        return PilotNetANN(in_channels=C, input_hw=(H,W))
    if n == "pilotnet_snn":
        return PilotNetSNN(in_channels=C, input_hw=(H,W),
                           beta=kwargs.pop("beta", 0.9),
                           threshold=kwargs.pop("threshold", 0.5),
                           learn_beta=kwargs.pop("learn_beta", True))
    raise ValueError(f"Modelo no reconocido: {name}")

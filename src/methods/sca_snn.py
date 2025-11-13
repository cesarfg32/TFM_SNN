# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from .base import BaseMethod
from src.nn_io import _forward_with_cached_orientation  # orientación consistente

# ------------------------------------------------------------
# Configuración SCA-lite
# ------------------------------------------------------------
@dataclass
class SCAConfig:
    attach_to: Optional[str] = None   # p.ej. "f6" en PilotNetSNN; None -> primera Linear
    flatten_spatial: bool = True

    # Similaridad (anchors)
    num_bins: int = 50
    bin_lo: float = -1.0
    bin_hi: float =  1.0
    anchor_batches: int = 8
    max_per_bin: int = 1024

    # Selective reuse (gating)
    beta: float = 0.5
    bias: float = 0.0
    habit_decay: float = 0.99
    soft_mask_temp: float = 0.0

    # NUEVO: objetivo de fracción de activas (0<r<1) — si se define, ajusta umbral por cuantil
    target_active_frac: Optional[float] = None

    # logging
    verbose: bool = False
    log_every: int = 400

    # compat temporal
    T: Optional[int] = None

# ------------------ utilidades de forma/similitud ------------------
def _find_first_linear(m: nn.Module) -> Optional[nn.Module]:
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            return mod
    return None

def _get_by_path(m: nn.Module, path: str) -> Optional[nn.Module]:
    cur = m
    for p in path.split("."):
        if not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur if isinstance(cur, nn.Module) else None

def _to_TBN(y: torch.Tensor, T_hint: Optional[int], flatten_spatial: bool) -> tuple[torch.Tensor, tuple[int, ...], bool]:
    """Normaliza a (T,B,N) para aplicar máscara por neurona."""
    y = y.contiguous()
    orig = y.shape

    if y.ndim == 2:  # (B,N) ó (T,N)
        return y.unsqueeze(0).contiguous(), orig, True

    if y.ndim == 3:  # (T,B,N) o (B,T,N) o (B,N,T)
        TBN = y
        if T_hint and y.shape[-1] == T_hint:   # (B,N,T) -> (T,B,N)
            TBN = y.permute(2, 0, 1).contiguous()
            return TBN, orig, True
        if T_hint and y.shape[1] == T_hint:    # (B,T,N) -> (T,B,N)
            TBN = y.permute(1, 0, 2).contiguous()
            return TBN, orig, True
        return TBN.contiguous(), orig, False

    if y.ndim == 5:  # (T,B,C,H,W) o (B,T,C,H,W)
        if T_hint and y.shape[0] != T_hint:
            y = y.permute(1, 0, 2, 3, 4).contiguous()
        T, B, C, H, W = y.shape
        if flatten_spatial:
            return y.reshape(T, B, C * H * W).contiguous(), orig, (T_hint is not None)
        return y.flatten(3).mean(3).contiguous(), orig, (T_hint is not None)

    return y.contiguous(), orig, False

def _from_TBN(yTBN: torch.Tensor, orig_shape: tuple[int, ...], need_back: bool, flatten_spatial: bool) -> torch.Tensor:
    yTBN = yTBN.contiguous()
    if len(orig_shape) == 2:
        return yTBN.squeeze(0).contiguous()
    if len(orig_shape) == 3:
        return (yTBN.permute(1, 0, 2).contiguous() if need_back else yTBN)
    if len(orig_shape) == 5:
        if need_back:
            return yTBN.permute(1, 0, *range(2, yTBN.ndim)).contiguous()
        return yTBN
    return yTBN

def _bin_index(y: torch.Tensor, lo: float, hi: float, num_bins: int) -> torch.Tensor:
    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32).contiguous()
    y = y.clamp(min=lo, max=hi)
    width = max(1e-8, (hi - lo))
    t = (y - lo) / width
    idx = torch.floor(t * float(num_bins)).to(torch.int64)
    return idx.clamp_(0, num_bins - 1).contiguous()

def _row_normalize_pos(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.relu(x)
    s = x.sum(dim=1, keepdim=True)
    s = torch.clamp(s, min=eps)
    return x / s

def _kl_symmetric(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    p = (p + eps) / (p.sum(dim=1, keepdim=True) + eps)
    q = (q + eps) / (q.sum(dim=1, keepdim=True) + eps)
    kl_pq = (p * (p.add(eps).log() - q.add(eps).log())).sum(dim=1)
    kl_qp = (q * (q.add(eps).log() - p.add(eps).log())).sum(dim=1)
    out = 0.5 * (kl_pq + kl_qp)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

def _similarity_from_kl(kl_vals: torch.Tensor) -> float:
    s = (1.0 / (1.0 + kl_vals))
    s = torch.clamp(s, 0.0, 1.0)
    return float(s.mean().item())

def _bincount_safe(idx: torch.Tensor, n_bins: int, *, weights: torch.Tensor | None = None) -> torch.Tensor:
    idx = torch.nan_to_num(idx, nan=0.0, posinf=0.0, neginf=0.0)
    idx = idx.to(torch.int64).contiguous().clamp_(0, n_bins - 1)
    if idx.numel() == 0:
        dev = idx.device
        dt = (weights.dtype if (weights is not None and isinstance(weights, torch.Tensor)) else torch.float32)
        return torch.zeros(n_bins, device=dev, dtype=dt)
    if weights is None:
        return torch.bincount(idx, minlength=n_bins).to(torch.float32)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32).contiguous()
    return torch.bincount(idx, weights=weights, minlength=n_bins)

def _topk_safe(score: torch.Tensor, k: int) -> torch.Tensor:
    s = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    k = max(1, int(k))
    if k >= s.numel():
        return torch.arange(s.numel(), device=s.device)
    return torch.topk(s, k=k, largest=True, sorted=False).indices

def _to_single_worker_loader_like(loader: DataLoader) -> DataLoader:
    """Copia un DataLoader con num_workers=0 y sin pin/persistentes."""
    try:
        return DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=getattr(loader, "sampler", None),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=1,
            drop_last=False,
            collate_fn=loader.collate_fn,
            timeout=0,
        )
    except Exception:
        return loader

# ---------------------------- método ----------------------------
class SCA_SNN(BaseMethod):
    name = "sca-snn"

    def __init__(self, *, device: Optional[torch.device] = None, loss_fn: Optional[nn.Module] = None, **kw):
        # Inyectamos device/loss_fn de forma limpia
        super().__init__(device=device, loss_fn=loss_fn)
        self.cfg = SCAConfig(**{k: v for k, v in kw.items() if k in SCAConfig.__annotations__})
        self.name = (
            f"{self.name}"
            f"_bins{self.cfg.num_bins}"
            f"_beta{self.cfg.beta:g}"
            f"_bias{self.cfg.bias:g}"
            f"_temp{self.cfg.soft_mask_temp:g}"
            f"_ab{self.cfg.anchor_batches}"
            f"_flat{int(self.cfg.flatten_spatial)}"
        )
        self._anchors_prev: List[torch.Tensor] = []
        self._target: Optional[nn.Module] = None
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._model_pre_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._w_grad_hook: Optional[torch.utils.hooks.RemovableHandle] = None
        self._attached_name: str = "(unknown)"

        self._N: Optional[int] = None
        self._R: Optional[torch.Tensor] = None
        self._Gacc: Optional[torch.Tensor] = None
        self._sim_min: float = 0.0

        self._step: int = 0
        self._T_seq: int = int(self.cfg.T or 1)
        self._need_grad_hook: bool = True

        # Durante recogida de anclas: no aplicar máscara
        self._suspend_mask: bool = False

        # aviso único si no se puede inferir B en anclas
        self._warned_shape_once: bool = False

    # -------- pre-forward: detectar T --------
    def _model_pre_forward_hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
        if not inputs:
            return
        x = inputs[0]
        if torch.is_tensor(x):
            if x.ndim == 5:
                self._T_seq = int(x.shape[0])
            elif x.ndim == 4:
                self._T_seq = 1
            else:
                self._T_seq = int(self.cfg.T or 1)
        else:
            self._T_seq = int(self.cfg.T or 1)

    # -------- helper robusto: (B,N) a partir del hook --------
    def _norm_BN(self, out: torch.Tensor, B: int) -> torch.Tensor:
        """
        Devuelve (B,N) a partir de 'out'.
        Soporta (B,N), (T,B,N), (B,T,N), (B,N,T), (B,C,H,W), (T,B,C,H,W), etc.
        - Promedia en T si existe.
        - Si flatten_spatial=True, promedia HxW antes de aplanar.
        - Si no se puede inferir B, degrada con un vector medio repetido.
        """
        t = out
        if not torch.is_tensor(t):
            raise RuntimeError("[SCA] Hook no devolvió un tensor válido.")
        t = t.contiguous()

        # Caso directo: (B, N...)
        if t.ndim >= 2 and t.shape[0] == B:
            if t.ndim >= 4 and self.cfg.flatten_spatial:
                t = t.mean(dim=(-1, -2))  # (B,C)
            return t.view(B, -1)

        # Reordenar si B está en otro eje
        if B in t.shape:
            bdim = list(t.shape).index(B)
            perm = [bdim] + [i for i in range(t.ndim) if i != bdim]
            t = t.permute(*perm).contiguous()  # (B, ...)
            if t.ndim >= 3 and t.shape[1] == self._T_seq:  # (B, T, ...)
                t = t.mean(dim=1)
            if t.ndim >= 4 and self.cfg.flatten_spatial:
                t = t.mean(dim=(-1, -2))
            return t.view(B, -1)

        # Temporal al frente: (T,B,...) o (T,...) sin B
        if t.ndim >= 3 and t.shape[0] == self._T_seq:
            if t.ndim >= 3 and t.shape[1] != 0:
                if t.shape[1] == B:
                    t = t.mean(dim=0)  # (B, ...)
                    if t.ndim >= 3 and self.cfg.flatten_spatial:
                        t = t.mean(dim=(-1, -2))
                    return t.view(B, -1)

        # Fallback (benigno)
        if not self._warned_shape_once:
            print(f"[SCA] WARNING: no puedo inferir (B={B}) desde shape={tuple(out.shape)}; "
                  f"usaré media global repetida.")
            self._warned_shape_once = True

        v = out
        if v.ndim >= 3 and self.cfg.flatten_spatial:
            v = v.mean(dim=tuple(range(2, v.ndim)))  # agrega espacial
        if v.ndim >= 2 and v.shape[0] == self._T_seq:
            v = v.mean(dim=0)  # promedia T si va delante
        v = v.view(-1) if v.ndim == 1 else v.mean(dim=0)  # (N,)
        v = torch.nan_to_num(v.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return v.view(1, -1).repeat(B, 1).contiguous()

    # -------- forward hook: gating --------
    def _forward_hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], out: torch.Tensor):
        if not torch.is_tensor(out) or self._N is None or self._R is None:
            return out

        if self._suspend_mask:
            return out

        yTBN, orig, need_back = _to_TBN(out, self._T_seq, self.cfg.flatten_spatial)
        yTBN = torch.nan_to_num(yTBN, nan=0.0, posinf=0.0, neginf=0.0).contiguous()

        _, _, N = yTBN.shape
        if N != self._N:
            dev_target = self._ensure_target_device()
            self._setup_per_neuron_state(N, device=dev_target)

        # métrica de reutilización acumulada
        Rn = torch.sigmoid(self._R)  # (N,)
        rho = self.cfg.beta - float(self._sim_min) + self.cfg.bias

        # NUEVO: si target_active_frac está definido, ajusta umbral por cuantil de Rn
        thr = rho
        taf = self.cfg.target_active_frac
        if isinstance(taf, float) and 0.0 < taf < 1.0:
            try:
                q = torch.quantile(Rn.detach(), 1.0 - taf)
                thr = max(float(q.item()), rho)  # respeta límite inferior basado en sim_min
            except Exception:
                thr = rho

        if self.cfg.soft_mask_temp > 0.0:
            m = torch.sigmoid((Rn - thr) / max(1e-6, self.cfg.soft_mask_temp)).view(1, 1, N)
        else:
            m = (Rn > thr).to(yTBN.dtype).view(1, 1, N)

        if m.device != yTBN.device:
            m = m.to(yTBN.device, non_blocking=True)

        yTBN = (yTBN * m).contiguous()
        yTBN = torch.nan_to_num(yTBN, nan=0.0, posinf=0.0, neginf=0.0)

        # --- LOG CONTROLADO ---
        if bool(getattr(self, "verbose", self.cfg.verbose)):
            self._step += 1
            every = max(10, int(getattr(self, "log_every", self.cfg.log_every)))
            if (self._step % every) == 0:
                try:
                    active = 100.0 * float(m.mean().item())
                    print(f"[SCA] step={self._step} | {self._attached_name} | sim_min={self._sim_min:.3f} "
                          f"| thr={thr:.3f} | act≈{active:.1f}%")
                except Exception:
                    pass

        return _from_TBN(yTBN, orig, need_back, self.cfg.flatten_spatial)

    # -------- grad hook: acumular por neurona --------
    def _weight_grad_hook(self, grad: torch.Tensor):
        if not torch.is_tensor(grad):
            return
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
        dev = grad.device
        n_out = grad.shape[0]
        with torch.no_grad():
            if (self._Gacc is None) or (self._Gacc.numel() != n_out):
                self._Gacc = torch.zeros(n_out, dtype=torch.float32, device=dev)
            elif self._Gacc.device != dev:
                self._Gacc = self._Gacc.to(dev, non_blocking=True)
            if (self._R is not None) and (self._R.device != dev):
                self._R = self._R.to(dev, non_blocking=True)
            g_row = grad.pow(2).sum(dim=1).sqrt()  # (N_out,)
            g_row = torch.nan_to_num(g_row, nan=0.0, posinf=0.0, neginf=0.0)
            self._Gacc.add_(g_row)

    # -------- helpers de estado --------
    def _setup_per_neuron_state(self, N: int, device):
        self._N = int(N)
        self._R = torch.zeros(N, dtype=torch.float32, device=device)
        self._Gacc = torch.zeros(N, dtype=torch.float32, device=device)

    def _ensure_target_device(self) -> torch.device:
        if self._target is not None and hasattr(self._target, "weight") and isinstance(self._target.weight, torch.Tensor):
            return self._target.weight.device
        if self._target is not None:
            try:
                return next(self._target.parameters()).device
            except StopIteration:
                pass
        return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    # -------- anchors (por tarea) --------
    @torch.no_grad()
    def _collect_anchors_for_task(self, model: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
        """Calcula medias de activación por bin (num_bins, N) en la capa objetivo."""
        assert self._target is not None and self._N is not None

        def _run_collect(active_loader: DataLoader) -> torch.Tensor:
            num_bins = int(self.cfg.num_bins)
            lo, hi = float(self.cfg.bin_lo), float(self.cfg.bin_hi)
            max_batches = int(self.cfg.anchor_batches)

            # Hook temporal: acumula TODAS las salidas de la capa objetivo a lo largo de T
            grab: Dict[str, List[torch.Tensor]] = {"outs": []}
            def _cap_hook(_m, _inp, out):
                if torch.is_tensor(out):
                    grab["outs"].append(out.detach())

            tmp_handle = self._target.register_forward_hook(_cap_hook, with_kwargs=False)
            cnt_per_bin = [0 for _ in range(num_bins)]
            sum_per_bin = [torch.zeros(self._N, dtype=torch.float32, device=device) for _ in range(num_bins)]

            was_training = model.training
            model.eval()
            try:
                batches_seen = 0
                phase_hint = {"val": None}
                for x, y in active_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True).view(-1)
                    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32).contiguous()
                    B = int(y.shape[0])

                    # Forward con orientación consistente, sin AMP
                    _ = _forward_with_cached_orientation(
                        model=model, x=x, y=y,
                        device=device, use_amp=False,
                        phase_hint=phase_hint, phase="val"
                    )

                    # Extrae y apila las salidas capturadas (T llamadas al hook)
                    outs = grab["outs"]
                    grab["outs"] = []
                    if len(outs) == 0:
                        continue
                    if len(outs) == 1:
                        raw = outs[0]
                    else:
                        try:
                            raw = torch.stack(outs, dim=0).contiguous()  # (T,B,N...) esperado
                        except Exception:
                            raw = outs[-1]

                    # Normaliza a (B,N) promediando T si procede
                    F = self._norm_BN(raw, B=B)  # (B,N)
                    if F.device != device:
                        F = F.to(device, non_blocking=True)
                    F = torch.nan_to_num(F.float(), nan=0.0, posinf=0.0, neginf=0.0).contiguous()

                    # Binning y acumulación
                    bidx = _bin_index(y, lo, hi, num_bins)  # (B,)
                    for b in range(num_bins):
                        mask = (bidx == b)
                        if mask.any():
                            take = F[mask]
                            room = int(self.cfg.max_per_bin) - cnt_per_bin[b]
                            if room <= 0:
                                continue
                            if take.shape[0] > room:
                                take = take[:room]
                            sum_per_bin[b].add_(torch.nan_to_num(take.sum(dim=0), nan=0.0, posinf=0.0, neginf=0.0))
                            cnt_per_bin[b] += int(take.shape[0])

                    batches_seen += 1
                    if batches_seen >= max_batches:
                        break
            finally:
                try:
                    tmp_handle.remove()
                except Exception:
                    pass
                if was_training:
                    model.train()

            anchors = torch.stack([
                (sum_per_bin[b] / max(1, cnt_per_bin[b])) for b in range(num_bins)
            ], dim=0).contiguous()  # (num_bins, N)
            anchors = torch.nan_to_num(anchors, nan=0.0, posinf=0.0, neginf=0.0)
            anchors = _row_normalize_pos(anchors)
            return anchors

        try:
            return _run_collect(loader)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}".lower()
            triggers = ("bus error", "unexpected bus error", "shared memory", "shm", "dataloader worker", "worker exited", "multiprocessing")
            if any(t in msg for t in triggers):
                print("[SCA] WARN: caída de worker durante anchors. Reintentando con num_workers=0…")
                safe_loader = _to_single_worker_loader_like(loader)
                return _run_collect(safe_loader)
            raise

    @torch.no_grad()
    def _estimate_similarity_min(self, anchors_curr: torch.Tensor) -> float:
        if not self._anchors_prev:
            return 0.0
        S_vals: List[float] = []
        for Aprev in self._anchors_prev:
            kl = _kl_symmetric(anchors_curr, Aprev)
            S = _similarity_from_kl(kl)
            S_vals.append(S)
        return float(min(S_vals)) if S_vals else 0.0

    # ---------------- API BaseMethod ----------------
    def penalty(self) -> torch.Tensor:
        # Device robusto
        if self._Gacc is not None and isinstance(self._Gacc, torch.Tensor):
            dev = self._Gacc.device
        elif self._R is not None and isinstance(self._R, torch.Tensor):
            dev = self._R.device
        else:
            dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

        # Alinea _R al device de _Gacc si difieren
        if self._R is not None and self._Gacc is not None and self._R.device != self._Gacc.device:
            self._R = self._R.to(self._Gacc.device, non_blocking=True)
            dev = self._Gacc.device

        if self._R is not None and self._Gacc is not None:
            with torch.no_grad():
                self._R.mul_(self.cfg.habit_decay)
                g = torch.nan_to_num(self._Gacc, nan=0.0, posinf=0.0, neginf=0.0)
                try:
                    q = torch.quantile(g, 0.95) if g.numel() else torch.tensor(1.0, device=g.device)
                except Exception:
                    q = torch.tensor(1.0, device=g.device if isinstance(g, torch.Tensor) else dev)
                if float(q) <= 1e-9:
                    q = torch.tensor(1.0, device=g.device if isinstance(g, torch.Tensor) else dev)
                self._R.add_(g / q)
                self._Gacc.zero_()
        return torch.zeros((), dtype=torch.float32, device=dev)

    def before_task(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
        if self._model_pre_hook_handle is None:
            self._model_pre_hook_handle = model.register_forward_pre_hook(
                self._model_pre_forward_hook, with_kwargs=False
            )

        target = None
        if self.cfg.attach_to:
            target = _get_by_path(model, self.cfg.attach_to)
            if target is None and self.cfg.verbose:
                print(f"[SCA] No encuentro '{self.cfg.attach_to}'. Usaré primera Linear.")
        if target is None:
            target = _find_first_linear(model)
        if target is None:
            print("[SCA] WARNING: no se encontró nn.Linear; SCA queda como no-op.")
            self._target = None
            return

        self._target = target
        self._attached_name = getattr(target, "__class__", type(target)).__name__
        dev_target = self._ensure_target_device()

        # Probe para dimensionar N (con orientación correcta y sin AMP)
        with torch.no_grad():
            try:
                xb, yb = next(iter(train_loader))
            except StopIteration:
                raise RuntimeError("[SCA] El train_loader está vacío; no puedo inicializar SCA.")
            xb = xb.to(dev_target, non_blocking=True)
            yb = yb.to(dev_target, non_blocking=True)

            def _probe(_m, _i, out):
                yTBN, _, _ = _to_TBN(out, self._T_seq, self.cfg.flatten_spatial)
                self._setup_per_neuron_state(yTBN.shape[-1], device=dev_target)

            h = target.register_forward_hook(_probe, with_kwargs=False)
            _ = _forward_with_cached_orientation(
                model=model, x=xb, y=yb,
                device=dev_target, use_amp=False,
                phase_hint={"val": None}, phase="val"
            )
            h.remove()

        # Recoge anclas del task SIN aplicar máscara
        self._suspend_mask = True
        anchors_curr = self._collect_anchors_for_task(model, train_loader, dev_target)
        self._suspend_mask = False
        self._sim_min = self._estimate_similarity_min(anchors_curr)

        # --- LOG CONTROLADO ---
        if bool(getattr(self, "verbose", self.cfg.verbose)):
            print(f"[SCA] start | attach={self.cfg.attach_to or 'auto'} -> {self._attached_name} | "
                  f"bins={self.cfg.num_bins} | sim_min={self._sim_min:.3f}")
            print(f"[SCA] probe | N={self._N} | anchor_batches={self.cfg.anchor_batches} | "
                  f"beta={self.cfg.beta} | bias={self.cfg.bias} | soft_temp={self.cfg.soft_mask_temp} | "
                  f"habit_decay={self.cfg.habit_decay} | target_active_frac={self.cfg.target_active_frac}")

        # Activa el hook de gating
        if self._hook_handle is None:
            self._hook_handle = target.register_forward_hook(self._forward_hook, with_kwargs=False)
        if self._need_grad_hook and hasattr(target, "weight") and isinstance(target.weight, torch.Tensor):
            self._w_grad_hook = target.weight.register_hook(self._weight_grad_hook)
            self._need_grad_hook = False

    def after_task(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
        try:
            _ = self.penalty()  # decay + reset Gacc
        except Exception:
            pass

        if self._target is None or self._N is None:
            return
        dev_target = self._ensure_target_device()

        # Recoge y guarda anclas del task SIN aplicar máscara
        self._suspend_mask = True
        anchors = self._collect_anchors_for_task(model, train_loader, dev_target)
        self._suspend_mask = False
        self._anchors_prev.append(anchors.detach().clone())

        # --- LOG CONTROLADO ---
        if bool(getattr(self, "verbose", self.cfg.verbose)):
            if self._R is not None and self._N:
                rho = self.cfg.beta - self._sim_min + self.cfg.bias
                Rn = torch.sigmoid(self._R)
                act_frac = float((Rn > rho).float().mean().item())
                print(f"[SCA] after_task: act≈{100.0*act_frac:.1f}% | rho={rho:.3f} | sim_min={self._sim_min:.3f}")
            print(f"[SCA] after_task: anchors guardadas. N={self._N}")

    def detach(self) -> None:
        if self._hook_handle is not None:
            try: self._hook_handle.remove()
            except Exception: pass
            self._hook_handle = None
        if self._model_pre_hook_handle is not None:
            try: self._model_pre_hook_handle.remove()
            except Exception: pass
            self._model_pre_hook_handle = None
        if self._w_grad_hook is not None:
            try: self._w_grad_hook.remove()
            except Exception: pass
            self._w_grad_hook = None
        self._target = None
        self._N = None
        self._R = None
        self._Gacc = None
        self._suspend_mask = False  # reset de seguridad
        self._warned_shape_once = False

    # introspección
    def get_state(self) -> Dict[str, float]:
        out = {
            "N": int(self._N or 0),
            "sim_min": float(self._sim_min),
            "R_mean": (float(self._R.mean().item()) if isinstance(self._R, torch.Tensor) and self._R.numel() else 0.0),
            "R_p95": (float(self._R.quantile(0.95).item()) if isinstance(self._R, torch.Tensor) and self._R.numel() else 0.0),
            "Gacc_mean": (float(self._Gacc.mean().item()) if isinstance(self._Gacc, torch.Tensor) and self._Gacc.numel() else 0.0),
            "beta": float(self.cfg.beta),
            "bias": float(self.cfg.bias),
            "soft_mask_temp": float(self.cfg.soft_mask_temp),
            "habit_decay": float(self.cfg.habit_decay),
            "num_bins": int(self.cfg.num_bins),
            "anchor_batches": int(self.cfg.anchor_batches),
            "target_active_frac": (None if self.cfg.target_active_frac is None else float(self.cfg.target_active_frac)),
        }
        return out

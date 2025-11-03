# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

# ------------------------------------------------------------
# Configuración SCA-lite
# ------------------------------------------------------------
@dataclass
class SCAConfig:
    attach_to: Optional[str] = None   # p.ej. "f6" en PilotNetSNN; None -> primera Linear
    flatten_spatial: bool = True

    # Similaridad (anchors)
    num_bins: int = 21
    bin_lo: float = -1.0
    bin_hi: float =  1.0
    anchor_batches: int = 8
    max_per_bin: int = 1024

    # Selective reuse (gating)
    beta: float = 0.5
    bias: float = 0.0
    habit_decay: float = 0.99
    soft_mask_temp: float = 0.0

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
    orig = y.shape
    if y.ndim == 2:  # (B,N) ó (T,N), lo trataremos fuera si es ambiguo
        return y.unsqueeze(0), orig, True
    if y.ndim == 3:  # (T,B,N) o (B,T,N) o (B,N,T)
        TBN = y
        if T_hint and y.shape[-1] == T_hint:   # (B,N,T) -> (T,B,N)
            TBN = y.permute(2, 0, 1).contiguous()
            return TBN, orig, True
        if T_hint and y.shape[1] == T_hint:    # (B,T,N) -> (T,B,N)
            TBN = y.permute(1, 0, 2).contiguous()
            return TBN, orig, True
        return TBN, orig, False
    if y.ndim == 5:  # (T,B,C,H,W) o (B,T,C,H,W)
        if T_hint and y.shape[0] != T_hint:
            y = y.permute(1, 0, 2, 3, 4).contiguous()
        T, B, C, H, W = y.shape
        if flatten_spatial:
            return y.reshape(T, B, C * H * W), orig, (T_hint is not None)
        return y.flatten(3).mean(3), orig, (T_hint is not None)
    return y, orig, False

def _from_TBN(yTBN: torch.Tensor, orig_shape: tuple[int, ...], need_back: bool, flatten_spatial: bool) -> torch.Tensor:
    if len(orig_shape) == 2:
        return yTBN.squeeze(0)
    if len(orig_shape) == 3:
        return yTBN.permute(1, 0, 2).contiguous() if need_back else yTBN
    if len(orig_shape) == 5:
        if need_back:
            return yTBN.permute(1, 0, *range(2, yTBN.ndim)).contiguous()
        return yTBN
    return yTBN

def _bin_index(y: torch.Tensor, lo: float, hi: float, num_bins: int) -> torch.Tensor:
    y = y.clamp(min=lo, max=hi)
    t = (y - lo) / max(1e-8, (hi - lo))
    idx = torch.floor(t * num_bins).long()
    return idx.clamp_(0, num_bins - 1)

def _row_normalize_pos(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = torch.relu(x)
    s = x.sum(dim=1, keepdim=True) + eps
    return x / s

def _kl_symmetric(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = (p + eps) / (p.sum(dim=1, keepdim=True) + eps)
    q = (q + eps) / (q.sum(dim=1, keepdim=True) + eps)
    kl_pq = (p * (p.add(eps).log() - q.add(eps).log())).sum(dim=1)
    kl_qp = (q * (q.add(eps).log() - p.add(eps).log())).sum(dim=1)
    return 0.5 * (kl_pq + kl_qp)

def _similarity_from_kl(kl_vals: torch.Tensor) -> float:
    s = (1.0 / (1.0 + kl_vals)).clamp(0.0, 1.0)
    return float(s.mean().item())


# ---------------------------- método ----------------------------
class SCA_SNN:
    name = "sca-snn"

    def __init__(self, **kw):
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

    # -------- forward hook: gating --------
    def _forward_hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], out: torch.Tensor):
        if not torch.is_tensor(out) or self._N is None or self._R is None:
            return out

        # Normaliza forma y aplica máscara
        yTBN, orig, need_back = _to_TBN(out, self._T_seq, self.cfg.flatten_spatial)
        T, B, N = yTBN.shape
        if N != self._N:
            dev_target = self._ensure_target_device()
            self._setup_per_neuron_state(N, device=dev_target)

        rho = self.cfg.beta - float(self._sim_min) + self.cfg.bias
        Rn = torch.sigmoid(self._R)  # (N,)

        if self.cfg.soft_mask_temp > 0.0:
            m = torch.sigmoid((Rn - rho) / max(1e-6, self.cfg.soft_mask_temp)).view(1, 1, N)
        else:
            m = (Rn > rho).to(yTBN.dtype).view(1, 1, N)
        if m.device != yTBN.device:
            m = m.to(yTBN.device, non_blocking=True)

        yTBN = yTBN * m
        yTBN = torch.nan_to_num(yTBN)

        if self.cfg.verbose:
            self._step += 1
            every = max(10, int(self.cfg.log_every))
            if (self._step % every) == 0:
                try:
                    active = 100.0 * float(m.mean().item())
                    print(f"[SCA] step={self._step} | {self._attached_name} | sim_min={self._sim_min:.3f} "
                          f"| rho={(self.cfg.beta - float(self._sim_min) + self.cfg.bias):.3f} | act≈{active:.1f}%")
                except Exception:
                    pass

        return _from_TBN(yTBN, orig, need_back, self.cfg.flatten_spatial)

    # -------- grad hook: acumular por neurona --------
    def _weight_grad_hook(self, grad: torch.Tensor):
        if not torch.is_tensor(grad):
            return
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
            self._Gacc.add_(g_row)

    # -------- helpers de estado --------
    def _setup_per_neuron_state(self, N: int, device):
        self._N = N
        self._R = torch.zeros(N, dtype=torch.float32, device=device)
        self._Gacc = torch.zeros(N, dtype=torch.float32, device=device)

    def _ensure_target_device(self) -> torch.device:
        if self._target is not None and hasattr(self._target, "weight") and isinstance(self._target.weight, torch.Tensor):
            return self._target.weight.device
        return next(self._target.parameters()).device if self._target is not None else (
            torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        )

    # -------- anchors (por tarea) --------
    @torch.no_grad()
    def _collect_anchors_for_task(self, model: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
        """Calcula medias de activación por bin (num_bins, N) en la capa objetivo."""
        assert self._target is not None and self._N is not None
        num_bins = int(self.cfg.num_bins)
        lo, hi = float(self.cfg.bin_lo), float(self.cfg.bin_hi)
        max_batches = int(self.cfg.anchor_batches)

        # Hook temporal: guarda out bruto (un solo buffer)
        grab = {"out": None}
        def _cap_hook(_m, _inp, out):
            if torch.is_tensor(out):
                grab["out"] = out.detach()

        tmp_handle = self._target.register_forward_hook(_cap_hook, with_kwargs=False)

        cnt_per_bin = [0 for _ in range(num_bins)]
        sum_per_bin = [torch.zeros(self._N, dtype=torch.float32, device=device) for _ in range(num_bins)]

        was_training = model.training
        model.eval()

        try:
            batches_seen = 0
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).view(-1)  # (B,)
                B = int(y.shape[0])

                # Desactiva autocast SOLO aquí para evitar NaNs en FP16 durante las anclas
                dev_type = "cuda" if torch.cuda.is_available() else "cpu"
                with torch.amp.autocast(device_type=dev_type, enabled=False):
                    _ = model(x)  # dispara hook

                raw = grab["out"]
                grab["out"] = None
                if raw is None:
                    continue

                # Normalizar a (B,N)
                if raw.ndim == 2:
                    if raw.shape[0] == B:
                        F = raw
                    elif raw.shape[1] == B:
                        F = raw.transpose(0, 1).contiguous()
                    elif raw.shape[0] % B == 0:
                        Tguess = raw.shape[0] // B
                        F = raw.view(Tguess, B, -1).mean(dim=0)
                    else:
                        yTBN, _, _ = _to_TBN(raw, self._T_seq, self.cfg.flatten_spatial)
                        if yTBN.ndim == 3 and yTBN.shape[1] == B:
                            F = yTBN.mean(dim=0)
                        else:
                            raise RuntimeError(f"[SCA] No puedo normalizar out con shape {tuple(raw.shape)} a (B,N) con B={B}")
                elif raw.ndim == 3:
                    yTBN, _, _ = _to_TBN(raw, self._T_seq, self.cfg.flatten_spatial)
                    if yTBN.shape[1] != B:
                        if raw.shape[0] == B:
                            F = raw.mean(dim=1)
                        elif raw.shape[1] == B:
                            F = raw.mean(dim=0)
                        elif raw.shape[-1] == B:
                            F = raw.permute(0, 2, 1).contiguous().mean(dim=0)
                        else:
                            raise RuntimeError(f"[SCA] out 3D no compatible con B={B}: {tuple(raw.shape)}")
                    else:
                        F = yTBN.mean(dim=0)
                else:
                    yTBN, _, _ = _to_TBN(raw, self._T_seq, self.cfg.flatten_spatial)
                    if yTBN.ndim == 3 and yTBN.shape[1] == B:
                        F = yTBN.mean(dim=0)
                    else:
                        raise RuntimeError(f"[SCA] out ND no soportado: {tuple(raw.shape)} con B={B}")

                if F.device != device:
                    F = F.to(device, non_blocking=True)
                F = torch.nan_to_num(F.float())  # (B,N), FP32 estable

                # Binning y acumulación
                bidx = _bin_index(y, lo, hi, num_bins)  # (B,)
                for b in range(num_bins):
                    mask = (bidx == b)  # (B,)
                    if mask.any():
                        take = F[mask]
                        room = self.cfg.max_per_bin - cnt_per_bin[b]
                        if room <= 0:
                            continue
                        if take.shape[0] > room:
                            take = take[:room]
                        sum_per_bin[b].add_(take.sum(dim=0))
                        cnt_per_bin[b] += int(take.shape[0])

                batches_seen += 1
                if batches_seen >= max_batches:
                    break
        finally:
            tmp_handle.remove()
            if was_training:
                model.train()

        anchors = torch.stack([
            (sum_per_bin[b] / max(1, cnt_per_bin[b])) for b in range(num_bins)
        ], dim=0)  # (num_bins, N)
        anchors = _row_normalize_pos(anchors)
        return anchors

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

    # ---------------- API ContinualMethod ----------------
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
                g = self._Gacc
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

        with torch.no_grad():
            try:
                xb, _ = next(iter(train_loader))
            except StopIteration:
                raise RuntimeError("[SCA] El train_loader está vacío; no puedo inicializar SCA.")
            xb = xb.to(dev_target, non_blocking=True)

            def _probe(_m, _i, out):
                yTBN, _, _ = _to_TBN(out, self._T_seq, self.cfg.flatten_spatial)
                self._setup_per_neuron_state(yTBN.shape[-1], device=dev_target)

            h = target.register_forward_hook(_probe, with_kwargs=False)
            _ = model(xb)
            h.remove()

        if self._hook_handle is None:
            self._hook_handle = target.register_forward_hook(self._forward_hook, with_kwargs=False)

        if self._need_grad_hook and hasattr(target, "weight") and isinstance(target.weight, torch.Tensor):
            self._w_grad_hook = target.weight.register_hook(self._weight_grad_hook)
            self._need_grad_hook = False

        anchors_curr = self._collect_anchors_for_task(model, train_loader, dev_target)
        self._sim_min = self._estimate_similarity_min(anchors_curr)

        print(f"[SCA] start | attach={self.cfg.attach_to or 'auto'} -> {self._attached_name} | "
              f"bins={self.cfg.num_bins} | sim_min={self._sim_min:.3f}")
        if self.cfg.verbose:
            print(f"[SCA] probe | N={self._N} | anchor_batches={self.cfg.anchor_batches} | "
                  f"beta={self.cfg.beta} | bias={self.cfg.bias} | soft_temp={self.cfg.soft_mask_temp} | "
                  f"habit_decay={self.cfg.habit_decay}")

    def after_task(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
        try:
            _ = self.penalty()  # decay + reset Gacc
        except Exception:
            pass
        if self._target is None or self._N is None:
            return
        dev_target = self._ensure_target_device()
        anchors = self._collect_anchors_for_task(model, train_loader, dev_target)
        self._anchors_prev.append(anchors.detach().clone())

        if self._R is not None and self._N:
            rho = self.cfg.beta - float(self._sim_min) + self.cfg.bias
            act_frac = float((torch.sigmoid(self._R) > rho).float().mean().item())
            print(f"[SCA] after_task: act≈{100.0*act_frac:.1f}% | rho={rho:.3f} | sim_min={self._sim_min:.3f}")

        if self.cfg.verbose:
            used = None
            if self._R is not None:
                thr = (self.cfg.beta - self._sim_min + self.cfg.bias)
                used = int((torch.sigmoid(self._R) > thr).sum().item())
            print(f"[SCA] after_task: anchors guardadas. N={self._N} | activos≈{used if used is not None else '-'}")

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
        }
        return out

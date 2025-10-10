# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class SASNNConfig:
    # K-WTA (traza temporal)
    k: int = 10
    tau: float = 10.0                 # tr[t+1] = tr[t] - tr[t]/tau + S[t+1]

    # Umbral variable (irreversible): Vth = Thmin + (Thmax - Thmin) * clamp(C/p, 0, 1)
    th_min: float = 1.0
    th_max: float = 2.0
    p: int = 2_000_000
    vt_scale: float = 1.0

    # Localización / shape
    attach_to: Optional[str] = None   # "f6" en PilotNetSNN; None -> auto primera Linear
    flatten_spatial: bool = True      # relevante si el tensor del hook tiene (C,H,W)
    assume_binary_spikes: bool = False

    # Ciclo temporal (se inyecta T desde el runner si procede; el pre-hook lo detecta)
    T: Optional[int] = None

    # Reset por tarea
    reset_counters_each_task: bool = False

    # Logging (el runner puede sobreescribir en el objeto con setattr)
    trace_verbose: bool = False
    trace_every: int = 100
    list_candidates_if_auto: bool = True


class SA_SNN:
    """
    SA-SNN (Selective Activation for Continual Learning):
      - Selección K-WTA con traza temporal (EMA).
      - Sesgo por umbral variable Vth (irreversible) para desalentar neuronas muy usadas.
    Se implementa con:
      * Un pre-forward hook en el MODELO para delimitar cada secuencia (batch) y resetear la traza temporal.
      * Un forward hook en la CAPA objetivo (p.ej., Linear f6) para aplicar la máscara y avanzar el cursor temporal.
    """
    name = "sa-snn"

    def __init__(self, **kw):
        self.cfg = SASNNConfig(**{k: v for k, v in kw.items() if k in SASNNConfig.__annotations__})

        # nombre legible
        tag = f"{self.name}_k{self.cfg.k}_tau{self.cfg.tau:g}_th{self.cfg.th_min:g}-{self.cfg.th_max:g}_p{self.cfg.p}"
        self.name = tag

        # Exponer flags de logging como atributos (para que runner pueda setattr)
        self.trace_verbose: bool = bool(self.cfg.trace_verbose)
        self.trace_every: int = int(self.cfg.trace_every)
        self.list_candidates_if_auto: bool = bool(self.cfg.list_candidates_if_auto)

        # hooks y estado de adjunción
        self._model_pre_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._target: Optional[nn.Module] = None
        self._attached_name: str = "(unknown)"

        # info temporal del batch actual (secuencia)
        self._T_seq: int = int(self.cfg.T or 1)    # longitud de la secuencia actual
        self._t_cursor: int = 0                    # índice temporal dentro de la secuencia

        # estado persistente por capa
        self._neurons_N: Optional[int] = None
        self._counters: Optional[torch.Tensor] = None  # [N], acumulador global
        self._last_mask_ratio: float = 0.0

        # traza por muestra para la secuencia actual (se crea en la primera llamada al hook)
        self._tr_seq: Optional[torch.Tensor] = None    # (B,N) en dtype del activación
        self._batch_seen: int = 0                      # solo para logs

    # ---------- utilidades de localización ----------
    @staticmethod
    def _find_first_linear(m: nn.Module) -> Optional[nn.Module]:
        for mod in m.modules():
            if isinstance(mod, nn.Linear):
                return mod
        return None

    @staticmethod
    def _get_by_path(m: nn.Module, path: str) -> Optional[nn.Module]:
        cur = m
        for p in path.split("."):
            if not hasattr(cur, p):
                return None
            cur = getattr(cur, p)
        return cur if isinstance(cur, nn.Module) else None

    # ---------- pre-forward hook del MODELO ----------
    def _model_pre_forward_hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
        """Delimita el inicio de una NUEVA secuencia/batch (resetea traza y cursor)."""
        if not inputs:
            # no tocamos nada
            return
        x = inputs[0]
        # Detecta T en la entrada al modelo:
        if torch.is_tensor(x):
            if x.ndim == 5:          # (T,B,C,H,W)
                self._T_seq = int(x.shape[0])
            elif x.ndim == 4:        # (B,C,H,W)
                self._T_seq = 1
            else:
                self._T_seq = int(self.cfg.T or 1)
        else:
            self._T_seq = int(self.cfg.T or 1)

        # Reset de cursor y traza para la nueva secuencia
        self._t_cursor = 0
        self._tr_seq = None  # se inicializa en el hook de la capa cuando conozcamos (B,N)

    # ---------- normalización / reconstrucción (por si el hook ve tensores 5D) ----------
    def _to_TBN(self, y: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], bool]:
        """Normaliza a (T,B,N). Si y es 2D (B,N), crea T=1 ficticio y marcará volver atrás."""
        orig = y.shape
        if y.ndim == 2:  # (B,N)
            return y.unsqueeze(0), orig, True
        if y.ndim == 3:  # (T,B,N) o (B,T,N) o (B,N,T)
            # intentamos dejar (T,B,N) sin adivinar en exceso
            TBN = y
            if self._T_seq and y.shape[-1] == self._T_seq:  # (B,N,T)
                TBN = y.permute(2, 0, 1).contiguous()
                return TBN, orig, True
            if self._T_seq and y.shape[1] == self._T_seq:   # (B,T,N)
                TBN = y.permute(1, 0, 2).contiguous()
                return TBN, orig, True
            return TBN, orig, False
        if y.ndim == 5:  # (T,B,C,H,W) o (B,T,C,H,W)
            T, B, C, H, W = (y.shape if y.shape[0] == self._T_seq
                             else y.permute(1, 0, 2, 3, 4).contiguous().shape)
            yT = y if y.shape[0] == self._T_seq else y.permute(1, 0, 2, 3, 4).contiguous()
            if self.cfg.flatten_spatial:
                return yT.reshape(T, B, C * H * W), orig, (y.shape[0] != self._T_seq)
            # media espacial por canal
            return yT.flatten(3).mean(3), orig, (y.shape[0] != self._T_seq)
        # casos raros, devolvemos tal cual
        return y, orig, False

    def _from_TBN(self, yTBN: torch.Tensor, orig_shape: tuple[int, ...], need_back: bool) -> torch.Tensor:
        if len(orig_shape) == 2:  # (B,N)
            return yTBN.squeeze(0)
        if len(orig_shape) == 3:
            if need_back:
                return yTBN.permute(1, 2, 0).contiguous()  # (B,N,T)
            return yTBN
        if len(orig_shape) == 5:
            T, B = yTBN.shape[:2]
            if self.cfg.flatten_spatial:
                try:
                    C, H, W = orig_shape[2:]
                    if C * H * W == yTBN.shape[2]:
                        yT = yTBN.reshape(T, B, C, H, W)
                    else:
                        yT = yTBN
                except Exception:
                    yT = yTBN
            else:
                yT = yTBN
            if need_back:
                return yT.permute(1, 0, *range(2, yT.ndim)).contiguous()
            return yT
        return yTBN

    # ---------- forward hook de la CAPA objetivo ----------
    def _forward_hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], out: torch.Tensor):
        if not torch.is_tensor(out):
            return out

        # Normalizamos (capas Linear típicamente salen como (B,N) -> T=1 ficticio)
        yTBN, orig, need_back = self._to_TBN(out)
        device, dtype = yTBN.device, yTBN.dtype
        T, B, N = yTBN.shape  # T será 1 en Linear; usaremos _t_cursor / _T_seq del pre-hook

        # Inicializar contadores persistentes por neurona si cambió N
        if (self._neurons_N is None) or (self._counters is None) or (self._neurons_N != N):
            self._neurons_N = N
            self._counters = torch.zeros(N, dtype=torch.float32, device=device)

        # Inicializar traza de la secuencia actual si no existe o cambia B/N
        if (self._tr_seq is None) or (self._tr_seq.shape != (B, N)):
            self._tr_seq = torch.zeros(B, N, dtype=dtype, device=device)

        # spikes binarios para la traza
        if self.cfg.assume_binary_spikes:
            S = yTBN[0]  # (N,B) con T=1 -> (B,N)
        else:
            S = (yTBN[0] > 0).to(dtype)

        # Umbral variable por neurona (global y persistente)
        C = self._counters                      # (N,)
        v_frac = torch.clamp(C / max(1, self.cfg.p), 0.0, 1.0)
        Vth = self.cfg.th_min + (self.cfg.th_max - self.cfg.th_min) * v_frac  # (N,)
        vt_bias = (self.cfg.vt_scale * (Vth - self.cfg.th_min)).to(dtype).view(1, -1)  # (1,N)

        # Avanzar traza temporal en el cursor de esta secuencia
        tau = float(self.cfg.tau)
        eps = 1e-8
        self._tr_seq = self._tr_seq - self._tr_seq / max(tau, eps) + S  # (B,N)
        score = self._tr_seq - vt_bias

        # Top-K por fila
        kk = min(max(1, int(self.cfg.k)), N)
        _, idx = torch.topk(score, k=kk, dim=1, largest=True, sorted=False)
        m = torch.zeros_like(score)
        m.scatter_(1, idx, 1.0)

        # Acumula aceptados globalmente (por neurona)
        self._counters += (S * m).sum(dim=0)  # (N,)

        # Aplica máscara al output original (no a S)
        yTBN[0] = yTBN[0] * m
        self._last_mask_ratio = float(m.mean().item())

        # Logging temporal con cursor real t/T
        if getattr(self, "trace_verbose", self.cfg.trace_verbose):
            self._batch_seen += 1
            every = max(1, int(getattr(self, "trace_every", self.cfg.trace_every)))
            if (self._batch_seen % every) == 0:
                try:
                    active = 100.0 * float(m.mean().item())
                    target = 100.0 * (kk / max(1, N))
                    print(
                        f"[SA-SNN] {self._attached_name} t={self._t_cursor}/{max(0,self._T_seq-1)} | "
                        f"trace_mean={self._tr_seq.mean().item():.4f} | score_mean={score.mean().item():.4f} | "
                        f"mask≈{active:.1f}% (target≈{target:.1f}%) | "
                        f"k={kk} N={N} | tau={tau:g} | vt={self.cfg.vt_scale:g} | "
                        f"th∈[{self.cfg.th_min:g},{self.cfg.th_max:g}]"
                    )
                except Exception:
                    pass

        # Avanza el cursor temporal; si completó la secuencia, lo resetea (por seguridad)
        self._t_cursor += 1
        if self._t_cursor >= self._T_seq:
            self._t_cursor = 0
            # No reseteamos self._tr_seq aquí: queremos que sea por-secuencia; el pre-hook lo hará.

        # Reconstruye y devuelve con la forma original
        return self._from_TBN(yTBN, orig, need_back)

    # ---------- API ContinualMethod ----------
    def penalty(self) -> torch.Tensor:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.zeros((), dtype=torch.float32, device=dev)

    def before_task(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
        # Engancha pre-forward del MODELO (delimita secuencias)
        if self._model_pre_hook_handle is None:
            self._model_pre_hook_handle = model.register_forward_pre_hook(
                self._model_pre_forward_hook, with_kwargs=False
            )

        # Localiza capa objetivo
        target = None
        if self.cfg.attach_to:
            target = self._get_by_path(model, self.cfg.attach_to)
            if target is None and self.list_candidates_if_auto:
                print(f"[SA-SNN] No encuentro '{self.cfg.attach_to}'. "
                      f"Intenta un nombre válido (p.ej. 'f6' en PilotNetSNN).")
        if target is None:
            target = self._find_first_linear(model)
            if target is None:
                print("[SA-SNN] WARNING: no se encontró nn.Linear; SA-SNN queda como no-op.")
                self._target = None
                return
            else:
                if self.list_candidates_if_auto:
                    print(f"[SA-SNN] attach_to=auto -> hook en primera capa Linear: {target}")

        self._target = target
        self._attached_name = getattr(target, "__class__", type(target)).__name__

        if self._hook_handle is None:
            self._hook_handle = target.register_forward_hook(self._forward_hook, with_kwargs=False)

        if self.cfg.reset_counters_each_task:
            self._neurons_N = None
            self._counters = None

        if getattr(self, "trace_verbose", self.cfg.trace_verbose):
            every = int(getattr(self, "trace_every", self.cfg.trace_every))
            print(f"[SA-SNN] tracing cada {every} pasos sobre {self._attached_name} ({self.cfg.attach_to or 'auto'})")

    def after_task(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
        if getattr(self, "trace_verbose", self.cfg.trace_verbose) and isinstance(self._counters, torch.Tensor):
            used = int((self._counters > 0).sum().item())
            tot = int(self._counters.numel())
            pct = 100.0 * used / max(1, tot)
            print(f"[SA-SNN] resumen tarea: neuronas con actividad aceptada = {used}/{tot} ({pct:.1f}%), "
                  f"mask_avg≈{100.0*self._last_mask_ratio:.1f}%")

    def detach(self) -> None:
        if self._hook_handle is not None:
            try:
                self._hook_handle.remove()
            except Exception:
                pass
            self._hook_handle = None
        if self._model_pre_hook_handle is not None:
            try:
                self._model_pre_hook_handle.remove()
            except Exception:
                pass
            self._model_pre_hook_handle = None
        self._target = None
        self._tr_seq = None

    # ---------- introspección ----------
    def get_state(self) -> dict:
        return {
            "neurons": int(self._neurons_N or 0),
            "last_mask_ratio": float(self._last_mask_ratio),
            "counters_sum": (float(self._counters.sum().item()) if isinstance(self._counters, torch.Tensor) else 0.0),
            "k": int(self.cfg.k),
            "tau": float(self.cfg.tau),
            "th_min": float(self.cfg.th_min),
            "th_max": float(self.cfg.th_max),
            "p": int(self.cfg.p),
            "vt_scale": float(self.cfg.vt_scale),
        }

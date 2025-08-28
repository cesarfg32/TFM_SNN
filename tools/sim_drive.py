# tools/sim_drive.py
from __future__ import annotations

import sys, argparse, base64, json, time
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

import torch
from torch.amp import autocast

import socketio
import eventlet
from flask import Flask

# --- sys.path al root del repo ------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Proyecto -----------------------------------------------------------------
from src.utils import load_preset
from src.models import build_model
from src.datasets import ImageTransform, encode_rate as enc_rate, encode_latency as enc_latency

# ==============================================================================
# Utilidades
# ==============================================================================

def b64_to_bgr(img_b64: str) -> np.ndarray:
    """Decodifica la imagen del simulador (base64 RGB) a BGR uint8."""
    img = Image.open(base64.b64decode(img_b64))
    rgb = np.asarray(img.convert("RGB"))
    bgr = rgb[..., ::-1].copy()
    return bgr

def make_encode_runtimer(encoder: str, T: int, gain: float):
    enc = str(encoder).lower()
    if enc not in {"rate", "latency", "raw", "image"}:
        raise ValueError(f"Encoder no soportado: {encoder}")
    def _fn(x_img: torch.Tensor) -> torch.Tensor:
        # x_img: (C,H,W) float32 [0,1]
        if enc == "image":
            # sin temporal → devolvemos (T=1,C,H,W)
            return x_img.unsqueeze(0)
        elif enc == "rate":
            return enc_rate(x_img, T=T, gain=gain)          # (T,C,H,W) o (T,H,W)
        elif enc == "latency":
            return enc_latency(x_img, T=T)                  # (T,C,H,W) o (T,H,W)
        elif enc == "raw":
            if x_img.dim() == 2:
                x_img = x_img.unsqueeze(0)
            return x_img.unsqueeze(0).expand(T, *x_img.shape).contiguous()
    return _fn

class PID:
    def __init__(self, kp=0.2, ki=0.001, kd=0.02, out_min=0.0, out_max=1.0):
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.out_min, self.out_max = float(out_min), float(out_max)
        self._prev = None
        self._integ = 0.0
    def reset(self):
        self._prev = None
        self._integ = 0.0
    def __call__(self, error: float, dt: float = 0.05) -> float:
        de = 0.0 if self._prev is None else (error - self._prev) / max(dt, 1e-3)
        self._integ += error * dt
        out = self.kp * error + self.ki * self._integ + self.kd * de
        self._prev = error
        return float(np.clip(out, self.out_min, self.out_max))

# ==============================================================================
# Servidor
# ==============================================================================

def main():
    ap = argparse.ArgumentParser("Servidor de inferencia para el Udacity Simulator")
    ap.add_argument("--ckpt", required=True, type=Path, help="Ruta al .pt (state_dict) — p. ej., model_best.pt")
    ap.add_argument("--model-name", default="pilotnet_snn",
                    choices=["pilotnet_snn","pilotnet_ann","snn_vision"],
                    help="Arquitectura a construir (debe coincidir con el entrenamiento)")
    ap.add_argument("--preset", default="fast", choices=["fast","std","accurate"],
                    help="Preset de configs/presets.yaml a usar como referencia")
    ap.add_argument("--config", default=str(ROOT / "configs" / "presets.yaml"))
    ap.add_argument("--crop-top", type=int, default=0, help="Recorte superior en píxeles antes del resize")
    # overrides opcionales
    ap.add_argument("--encoder", default=None, choices=[None,"rate","latency","raw","image"])
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--img-w", type=int, default=None)
    ap.add_argument("--img-h", type=int, default=None)
    ap.add_argument("--rgb", action="store_true", help="Fuerza color (equivale a to_gray=False)")
    # PID / velocidad
    ap.add_argument("--target-speed", type=float, default=22.0, help="Velocidad objetivo (mph del simulador)")
    ap.add_argument("--kp", type=float, default=0.25)
    ap.add_argument("--ki", type=float, default=0.0005)
    ap.add_argument("--kd", type=float, default=0.02)
    # red
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=4567)

    args = ap.parse_args()

    # --- Carga preset ---------------------------------------------------------
    cfg = load_preset(Path(args.config), args.preset)
    DATA  = cfg["data"]
    MODEL = cfg["model"]

    encoder = args.encoder if args.encoder else DATA["encoder"]
    T       = int(args.T if args.T is not None else DATA["T"])
    gain    = float(args.gain if args.gain is not None else DATA["gain"])

    W = int(args.img_w if args.img_w is not None else MODEL["img_w"])
    H = int(args.img_h if args.img_h is not None else MODEL["img_h"])
    to_gray = not args.rgb if args.rgb else bool(MODEL["to_gray"])

    # --- Modelo / transform ---------------------------------------------------
    tfm = ImageTransform(W, H, to_gray=to_gray, crop_top=args.crop_top)
    model = build_model(args.model_name, tfm, beta=0.9, threshold=0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Carga state_dict
    state = torch.load(args.ckpt, map_location=device)
    # permite CKPT con dict {'state_dict': ...}
    if isinstance(state, dict) and "state_dict" in state and hasattr(model, "load_state_dict"):
        model.load_state_dict(state["state_dict"])
    elif isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        # parece un state_dict directo
        model.load_state_dict(state)
    else:
        raise RuntimeError(f"Formato de checkpoint no reconocido: {args.ckpt}")

    print(f"[sim] dispositivo={device} | AMP={'ON' if torch.cuda.is_available() else 'OFF'}")
    print(f"[sim] modelo={args.model_name} | {W}x{H} gray={to_gray} | enc={encoder} T={T} gain={gain}")
    if args.crop_top:
        print(f"[sim] crop_top={args.crop_top} px")

    # Encoder temporal runtime (CPU→GPU en inferencia)
    encode_runtime = make_encode_runtimer(encoder, T, gain)

    # --- PID para throttle ----------------------------------------------------
    pid = PID(kp=args.kp, ki=args.ki, kd=args.kd, out_min=0.0, out_max=1.0)
    target_speed = float(args.target_speed)

    # --- SocketIO / Flask -----------------------------------------------------
    sio = socketio.Server()
    app = Flask(__name__)

    # Estado simple para FPS/tiempos
    last_t = [time.perf_counter()]

    @sio.on("connect")
    def connect(sid, environ):
        print("[sim] conectado:", sid)
        pid.reset()
        sio.emit("steer", data={"steering_angle": "0", "throttle": "0.0"}, to=sid)

    @sio.on("telemetry")
    def telemetry(sid, data):
        if data is None:
            return
        try:
            img_b64 = data["image"]
            speed = float(data.get("speed", 0.0))
        except Exception:
            return

        # 1) Decode & transform
        bgr = b64_to_bgr(img_b64)
        if args.crop_top and args.crop_top > 0:
            bgr = bgr[args.crop_top:, :, :]
        x_img = tfm(bgr).float()  # (C,H,W) en [0,1]
        # 2) Encode temporal
        xT = encode_runtime(x_img)  # (T,C,H,W) o (1,C,H,W) o (T,H,W)
        if xT.dim() == 3:  # (T,H,W) -> (T,1,H,W)
            xT = xT.unsqueeze(1)
        # 3) Añadir batch -> (T,1,C,H,W)
        if xT.dim() == 4:
            x5d = xT.unsqueeze(1)
        else:
            x5d = xT  # ya 5D

        # 4) Inferencia
        with torch.no_grad():
            x5d_dev = x5d.to(device, non_blocking=True)
            use_amp = torch.cuda.is_available()
            with autocast("cuda", enabled=use_amp):
                y = model(x5d_dev)        # (B=1, 1)
            steer = float(y.squeeze().detach().cpu().item())
            steer = float(np.clip(steer, -1.0, 1.0))

        # 5) Control de velocidad (PID sencillo)
        err = target_speed - speed
        throttle = pid(err)
        throttle = float(np.clip(throttle, 0.0, 1.0))

        # 6) Enviar al simulador
        sio.emit("steer", data={
            "steering_angle": f"{steer:.6f}",
            "throttle": f"{throttle:.6f}"
        }, to=sid)

        # 7) Log micro (FPS)
        t = time.perf_counter()
        dt = t - last_t[0]
        last_t[0] = t
        if dt > 0:
            fps = 1.0 / dt
            print(f"[sim] steer={steer:+.3f} thr={throttle:.2f} | speed={speed:.1f} | {fps:.1f} FPS", end="\r")

    # Lanzar servidor
    app = socketio.Middleware(sio, app)
    print(f"[sim] escuchando en http://{args.host}:{args.port}")
    eventlet.wsgi.server(eventlet.listen((args.host, args.port)), app)

if __name__ == "__main__":
    main()

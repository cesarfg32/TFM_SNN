# tools/sim_drive.py
from __future__ import annotations
import sys
import argparse
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import cv2  # noqa: F401  # por si quieres debug visual
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
from src.config import load_preset
from src.models import build_model
from src.datasets import ImageTransform, encode_rate as enc_rate, encode_latency as enc_latency

# Si tus labels están en [-1,1] mapeados a ±25 grados aprox.
DEG_RANGE = 25.0


# ==============================================================================
# Utilidades
# ==============================================================================
def b64_to_bgr(img_b64: str) -> np.ndarray:
    """Convierte una imagen JPEG en base64 (string) a un array BGR (para OpenCV)."""
    # Por si viniera con encabezado tipo 'data:image/jpeg;base64,...'
    if "," in img_b64:
        img_b64 = img_b64.split(",", 1)[1]

    # Decodificar base64 → bytes
    img_bytes = base64.b64decode(img_b64)

    # PIL necesita un file-like, no bytes crudos
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert("RGB")  # aseguramos 3 canales

    rgb = np.asarray(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
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
            return enc_rate(x_img, T=T, gain=gain)  # (T,C,H,W) o (T,H,W)
        elif enc == "latency":
            return enc_latency(x_img, T=T)          # (T,C,H,W) o (T,H,W)
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
    ap.add_argument(
        "--ckpt",
        required=True,
        type=Path,
        help="Ruta al .pt (state_dict) — p. ej., model_best.pth",
    )
    ap.add_argument(
        "--model-name",
        default="pilotnet_snn",
        choices=["pilotnet_snn", "pilotnet_ann", "snn_vision"],
        help="Arquitectura a construir (debe coincidir con el entrenamiento)",
    )
    ap.add_argument(
        "--preset",
        default="fast",
        choices=["fast", "std", "accurate"],
        help="Preset de configs/presets.yaml a usar como referencia",
    )
    ap.add_argument("--config", default=str(ROOT / "configs" / "presets.yaml"))

    ap.add_argument(
        "--crop-top",
        type=int,
        default=None,
        help="Recorte superior en píxeles antes del resize (por defecto, el del preset)",
    )

    # overrides opcionales
    ap.add_argument(
        "--encoder",
        default=None,
        choices=[None, "rate", "latency", "raw", "image"],
    )
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--img-w", type=int, default=None)
    ap.add_argument("--img-h", type=int, default=None)
    ap.add_argument(
        "--crop-bottom",
        type=int,
        default=None,
        help="Recorte inferior en píxeles antes del resize (por defecto, el del preset)",
    )
    ap.add_argument(
        "--rgb",
        action="store_true",
        help="Fuerza color (equivale a to_gray=False)",
    )

    # PID / velocidad
    ap.add_argument(
        "--target-speed",
        type=float,
        default=22.0,
        help="Velocidad objetivo (mph del simulador)",
    )
    ap.add_argument("--kp", type=float, default=0.25)
    ap.add_argument("--ki", type=float, default=0.0005)
    ap.add_argument("--kd", type=float, default=0.02)

    # red
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=4567)

    args = ap.parse_args()

    # --- Carga preset ---------------------------------------------------------
    cfg = load_preset(Path(args.config), args.preset)
    DATA = cfg["data"]
    MODEL = cfg["model"]

    encoder = args.encoder if args.encoder else DATA["encoder"]
    T = int(args.T if args.T is not None else DATA["T"])
    gain = float(args.gain if args.gain is not None else DATA["gain"])

    W = int(args.img_w if args.img_w is not None else MODEL["img_w"])
    H = int(args.img_h if args.img_h is not None else MODEL["img_h"])
    to_gray = not args.rgb if args.rgb else bool(MODEL["to_gray"])

    preset_crop_top = int(MODEL.get("crop_top", 0) or 0)
    preset_crop_bottom = int(MODEL.get("crop_bottom", 0) or 0)

    crop_top = args.crop_top if args.crop_top is not None else preset_crop_top
    crop_bottom = args.crop_bottom if args.crop_bottom is not None else preset_crop_bottom

    # --- Modelo / transform ---------------------------------------------------
    tfm = ImageTransform(
        W, H,
        to_gray=to_gray,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
    )
    model = build_model(args.model_name, tfm, beta=0.9, threshold=0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Carga state_dict
    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and hasattr(model, "load_state_dict"):
        model.load_state_dict(state["state_dict"])
    elif isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        # parece un state_dict directo
        model.load_state_dict(state)
    else:
        raise RuntimeError(f"Formato de checkpoint no reconocido: {args.ckpt}")

    print(f"[sim] dispositivo={device} | AMP={'ON' if torch.cuda.is_available() else 'OFF'}")
    print(f"[sim] modelo={args.model_name} | {W}x{H} gray={to_gray} | enc={encoder} T={T} gain={gain}")
    print(f"[sim] crop_top={crop_top} px | crop_bottom={crop_bottom} px")
    if args.crop_top:
        print(f"[sim] crop_top={args.crop_top} px")
        

    # Encoder temporal runtime
    encode_runtime = make_encode_runtimer(encoder, T, gain)

    # --- PID para throttle ----------------------------------------------------
    pid = PID(kp=args.kp, ki=args.ki, kd=args.kd, out_min=0.0, out_max=1.0)
    target_speed = float(args.target_speed)

    # --- SocketIO / Flask -----------------------------------------------------
    # logger=False / engineio_logger=False para no imprimir los JSON con la imagen
    sio = socketio.Server(
        async_mode="eventlet",
        cors_allowed_origins="*",
        logger=False,
        engineio_logger=False,
    )
    app = Flask(__name__)

    # Estado simple para FPS/tiempos
    last_t = [time.perf_counter()]
    frame_counter = [0]  # para logs de debug legibles

    @sio.on("connect")
    def connect(sid, environ):
        print("[sim] CONNECT recibido, sid=", sid)
        pid.reset()
        # Enviamos algo por defecto
        sio.emit("steer", data={"steering_angle": "0", "throttle": "0.0"}, to=sid)

    @sio.on("disconnect")
    def disconnect(sid):
        print("[sim] DISCONNECT:", sid)

    @sio.on("telemetry")
    def telemetry(sid, data):
        # A veces el simulador manda data=None para iniciar/terminar
        if data is None:
            print("[sim] telemetry: data=None (probando cambio de modo)")
            return

        try:
            img_b64 = data["image"]
            speed = float(data.get("speed", 0.0))
        except Exception as ex:
            print("[sim] ERROR parseando telemetry:", ex)
            return

        # 1) Decode & transform
        try:
            bgr = b64_to_bgr(img_b64)
        except Exception as e:
            print("[sim] ERROR decodificando imagen:", repr(e))
            return  # ignoramos este frame

        x_img = tfm(bgr).float()  # (C,H,W) en [0,1]

        # 2) Encode temporal
        xT = encode_runtime(x_img)  # (T,C,H,W) o (1,C,H,W) o (T,H,W)
        if xT.dim() == 3:           # (T,H,W) -> (T,1,H,W)
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
                y = model(x5d_dev)  # (B=1, 1)
            steer = float(y.squeeze().detach().cpu().item())
            steer = float(np.clip(steer, -1.0, 1.0))

        # 5) Control de velocidad (PID sencillo)
        err = target_speed - speed
        throttle = pid(err)
        throttle = float(np.clip(throttle, 0.0, 1.0))

        # 6) Enviar al simulador
        sio.emit(
            "steer",
            data={
                "steering_angle": f"{steer:.6f}",
                "throttle": f"{throttle:.6f}",
            },
            to=sid,
        )

        # 6-bis) Debug legible (primeros frames)
        frame_counter[0] += 1
        if frame_counter[0] <= 200 and frame_counter[0] % 10 == 0:
            steer_deg = steer * DEG_RANGE
            print(
                f"\n[debug] frame={frame_counter[0]} "
                f"steer_norm={steer:+.3f} ({steer_deg:+.1f} deg) "
                f"| speed={speed:.1f} | thr={throttle:.2f}"
            )

        # 7) Log micro (FPS) en la misma línea
        t = time.perf_counter()
        dt = t - last_t[0]
        last_t[0] = t
        if dt > 0:
            fps = 1.0 / dt
            print(
                f"[sim] steer={steer:+.3f} thr={throttle:.2f} "
                f"| speed={speed:.1f} | {fps:.1f} FPS       ",
                end="\r",
            )

    # Lanzar servidor (log_output=False para no ver cada petición)
    app = socketio.Middleware(sio, app)
    print(f"[sim] escuchando en http://{args.host}:{args.port}")
    eventlet.wsgi.server(
        eventlet.listen((args.host, args.port)),
        app,
        log_output=False,
    )


if __name__ == "__main__":
    main()

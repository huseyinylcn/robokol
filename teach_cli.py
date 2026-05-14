"""
Terminalden motor komutu ver → kamera hareketi ölç → motor_map ata.

Kullanım:
  python teach_cli.py
  python teach_cli.py --config app_config.json --scene-top 1.0

Örnek komutlar (satır yazıp Enter):
  12 i 0.45     → GET .../12i, 0.45 sn, sonra 12d
  12i           → aynı (varsayılan süre app_config calibration_pulse_sec)
  13 d          → sadece dur
  stop          → motor_ids listesindeki tüm motorlara d
  map base_left 12
  show
  commit        → motor_map.json yazar (5 rol dolu olmalı)
  help | quit
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import requests

from calibration import MotionSignature, _classify_label, measure_motion
from config import CMD_BACKWARD, CMD_FORWARD, CMD_STOP, load_app_config, save_motor_map
from robot_controller import MotorMap
from vision import make_camera_stream

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROLES = ("base_left", "base_right", "arm_up", "arm_down", "claw")


def raw_get(base_url: str, motor_id: int, cmd: str, timeout: float) -> bool:
    url = f"{base_url.rstrip('/')}/{motor_id}{cmd}"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code >= 400:
            logger.warning("HTTP %s %s", r.status_code, url)
            return False
        return True
    except requests.RequestException as e:
        logger.error("%s — %s", url, e)
        return False


def stop_all(base_url: str, motor_ids: list[int], timeout: float) -> None:
    for mid in motor_ids:
        raw_get(base_url, mid, CMD_STOP, timeout)
        time.sleep(0.03)


def parse_motor_line(
    line: str, default_pulse: float
) -> Optional[Tuple[str, int, str, float]]:
    """
    Dönüş: ('pulse', motor_id, 'i'|'g'|'d', seconds) veya ('stop_one', motor_id, 'd', 0)
    """
    line = line.strip()
    if not line:
        return None
    m = re.match(r"^(\d+)([igd])\s*([\d.]+)?$", line, re.I)
    if m:
        mid = int(m.group(1))
        c = m.group(2).lower()
        dur_s = float(m.group(3)) if m.group(3) else default_pulse
        if c == "d":
            return ("pulse", mid, "d", 0.0)
        return ("pulse", mid, c, dur_s)

    parts = line.split()
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].lower() in ("i", "g", "d"):
        mid = int(parts[0])
        c = parts[1].lower()
        dur_s = float(parts[2]) if len(parts) > 2 else default_pulse
        if c == "d":
            return ("pulse", mid, "d", 0.0)
        return ("pulse", mid, c, dur_s)
    return None


def run() -> None:
    ap = argparse.ArgumentParser(description="Terminal motor öğretimi + kamera akış ölçümü")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument(
        "--scene-top",
        type=float,
        default=1.0,
        help="1.0 = tüm karede optik akış (sabit kamera + hareket eden kol için). 0.75 = üst %%75.",
    )
    ap.add_argument("--preview", action="store_true", help="Önce/sonra fark penceresi")
    args = ap.parse_args()

    cfg = load_app_config(Path(args.config) if args.config else None)
    stream = make_camera_stream(
        cfg.camera_url,
        cfg.camera_url_2,
        stitch_horizontal=cfg.camera_stitch_horizontal,
    )
    stream.start()
    if not stream.wait_until_frame(timeout_sec=25.0):
        stream.stop()
        raise SystemExit("Kamera açılmadı veya kare gelmiyor.")

    def grab():
        for _ in range(50):
            f = stream.read()
            if f is not None and f.size > 0:
                return f
            time.sleep(0.04)
        raise RuntimeError("Kamera karesi yok")

    staging: Dict[str, int] = {}
    default_pulse = cfg.calibration_pulse_sec
    settle = cfg.calibration_settle_sec
    base = cfg.robot_base_url
    mids = list(cfg.motor_ids)
    timeout = 3.5

    print("\n=== teach_cli — komut ver, kamera hareketi ölçsün ===")
    print("Örnek:  12 i 0.5   |   12i   |   13 d   |   stop")
    print("Harita: map base_left 12   →   show   →   commit")
    print(f"scene_top_ratio={args.scene_top} (1.0 tüm kare önerilir)\n")

    if args.preview:
        cv2.namedWindow("teach_diff", cv2.WINDOW_NORMAL)

    try:
        while True:
            try:
                line = input("teach> ").strip()
            except EOFError:
                break
            if not line:
                continue
            low = line.lower()
            if low in ("q", "quit", "exit"):
                break
            if low == "help" or low == "?":
                print(__doc__)
                continue
            if low == "stop":
                stop_all(base, mids, timeout)
                print("Tüm motorlar durduruldu (d).")
                continue

            if low.startswith("map "):
                rest = line[4:].strip().split()
                if len(rest) != 2 or rest[0] not in ROLES or not rest[1].isdigit():
                    print("Kullanım: map <rol> <motor_id>  örn: map base_left 12")
                    continue
                staging[rest[0]] = int(rest[1])
                print("Güncel:", staging)
                continue

            if low == "show":
                print("Harita taslağı:", staging)
                continue

            if low == "commit":
                missing = [r for r in ROLES if r not in staging]
                if missing:
                    print("Eksik roller:", ", ".join(missing))
                    continue
                mm = MotorMap(
                    base_left=staging["base_left"],
                    base_right=staging["base_right"],
                    arm_up=staging["arm_up"],
                    arm_down=staging["arm_down"],
                    claw=staging["claw"],
                    claw_open_cmd=CMD_FORWARD,
                    claw_close_cmd=CMD_BACKWARD,
                )
                save_motor_map(mm.to_json(), path=Path(cfg.motor_map_path))
                print(f"Kaydedildi: {cfg.motor_map_path}")
                continue

            parsed = parse_motor_line(line, default_pulse)
            if not parsed:
                print("Anlaşılamadı. Örnek: 12 i 0.5 veya help")
                continue

            _, motor_id, cmd, dur = parsed
            f0 = grab()
            time.sleep(settle * 0.3)

            if cmd == "d":
                raw_get(base, motor_id, CMD_STOP, timeout)
                print(f"→ {motor_id}{CMD_STOP}")
                time.sleep(settle * 0.3)
                f1 = grab()
            else:
                ok = raw_get(base, motor_id, cmd, timeout)
                print(f"→ {motor_id}{cmd} ({dur:.2f}s) ok={ok}")
                time.sleep(max(0.05, dur))
                raw_get(base, motor_id, CMD_STOP, timeout)
                time.sleep(settle)
                f1 = grab()

            fx, fy, mag, gdiff = measure_motion(f0, f1, args.scene_top)
            sig = MotionSignature(
                motor_id=motor_id,
                mean_flow_x=fx,
                mean_flow_y=fy,
                flow_mag=mag,
                gray_diff_mean=gdiff,
                responded=mag >= 0.08 or gdiff >= 2.5,
            )
            label, conf = _classify_label(sig)
            print(
                f"  kamera: flow=({fx:+.3f},{fy:+.3f}) |mag|={mag:.3f} diff={gdiff:.2f}  "
                f"tahmin={label} güven={conf:.2f}"
            )
            if args.preview:
                g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
                g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
                if g0.shape != g1.shape:
                    g1 = cv2.resize(g1, (g0.shape[1], g0.shape[0]))
                d = cv2.absdiff(g0, g1)
                d3 = cv2.applyColorMap(
                    cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(cv2.uint8),
                    cv2.COLORMAP_JET,
                )
                cv2.putText(
                    d3,
                    f"m{motor_id}{cmd} mag={mag:.2f}",
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow("teach_diff", d3)
                cv2.waitKey(1)
    finally:
        stop_all(base, mids, timeout)
        stream.stop()
        if args.preview:
            cv2.destroyWindow("teach_diff")
        print("Çıkıldı.")


if __name__ == "__main__":
    run()

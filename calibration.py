"""
Motor keşfi: üst sahne ROI + optik akış, ileri/geri çift yönlü doğrulama, HTTP tekrar.
Robot gövdesi altta kaldığı için akış üst şeritte ölçülür.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import CMD_BACKWARD, CMD_FORWARD, save_motor_map
from robot_controller import MotorMap, RobotController

logger = logging.getLogger(__name__)


@dataclass
class MotionSignature:
    motor_id: int
    mean_flow_x: float
    mean_flow_y: float
    flow_mag: float
    gray_diff_mean: float
    reverse_mean_flow_x: float = 0.0
    reverse_mean_flow_y: float = 0.0
    reverse_flow_mag: float = 0.0
    bidirectional_ok: bool = False
    responded: bool = True
    http_forward_ok: bool = True
    http_backward_ok: bool = True


def _scene_crop(frame: np.ndarray, scene_top_ratio: float) -> np.ndarray:
    h = frame.shape[0]
    y2 = max(24, int(float(h) * scene_top_ratio))
    return frame[0:y2, :]


def _optical_flow_stats(
    prev_gray: np.ndarray, gray: np.ndarray
) -> Tuple[float, float, float]:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=4,
        winsize=21,
        iterations=4,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    fx = float(np.mean(flow[..., 0]))
    fy = float(np.mean(flow[..., 1]))
    mag = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
    return fx, fy, mag


def _capture_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def measure_motion(
    frame_before: np.ndarray,
    frame_after: np.ndarray,
    scene_top_ratio: float,
) -> Tuple[float, float, float, float]:
    g0 = _capture_gray(_scene_crop(frame_before, scene_top_ratio))
    g1 = _capture_gray(_scene_crop(frame_after, scene_top_ratio))
    if g0.shape != g1.shape:
        g1 = cv2.resize(g1, (g0.shape[1], g0.shape[0]), interpolation=cv2.INTER_AREA)
    fx, fy, mag = _optical_flow_stats(g0, g1)
    diff = cv2.absdiff(g0, g1)
    return fx, fy, mag, float(np.mean(diff))


def _best_post_pulse_frame(
    baseline: np.ndarray,
    grab_fn: Callable[[], np.ndarray],
    scene_top_ratio: float,
    samples: int = 5,
    delay_sec: float = 0.05,
) -> np.ndarray:
    """Pulse sonrası birkaç kare al; taban çizgisine göre en çok hareket eden çifti seç."""
    g0 = _capture_gray(_scene_crop(baseline, scene_top_ratio))
    best_mag = -1.0
    best: Optional[np.ndarray] = None
    for _ in range(samples):
        time.sleep(delay_sec)
        fr = grab_fn()
        g1 = _capture_gray(_scene_crop(fr, scene_top_ratio))
        if g1.shape != g0.shape:
            g1 = cv2.resize(g1, (g0.shape[1], g0.shape[0]), interpolation=cv2.INTER_AREA)
        _, _, mag = _optical_flow_stats(g0, g1)
        if mag > best_mag:
            best_mag = mag
            best = fr
    return best if best is not None else grab_fn()


def _bidirectional_consistent(
    fx1: float, fy1: float, mag1: float, fx2: float, fy2: float, mag2: float
) -> bool:
    if mag1 < 0.2 or mag2 < 0.15:
        return False
    v1 = np.array([fx1, fy1], dtype=np.float64)
    v2 = np.array([fx2, fy2], dtype=np.float64)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return False
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return cos < -0.15


def _classify_label(sig: MotionSignature) -> Tuple[str, float]:
    if not sig.responded:
        return "unknown", 0.12
    ax, ay = abs(sig.mean_flow_x), abs(sig.mean_flow_y)
    mag = sig.flow_mag
    boost = 1.12 if sig.bidirectional_ok else 1.0
    if mag < 0.18 and sig.gray_diff_mean < 3.5:
        return "unknown", 0.18
    if ax > ay * 1.25:
        if sig.mean_flow_x > 0:
            return "pan_right", min(1.0, (0.42 + mag * 0.22) * boost)
        return "pan_left", min(1.0, (0.42 + mag * 0.22) * boost)
    if ay > ax * 1.25:
        if sig.mean_flow_y > 0:
            return "down", min(1.0, (0.42 + mag * 0.22) * boost)
        return "up", min(1.0, (0.42 + mag * 0.22) * boost)
    if sig.gray_diff_mean > 7.0 and mag < 0.55:
        return "claw_like", min(1.0, (0.38 + sig.gray_diff_mean * 0.018) * boost)
    return "mixed", min(1.0, 0.48 * boost)


def _assign_roles(
    signatures: List[MotionSignature],
) -> Tuple[Dict[str, int], Dict[str, float], Dict[int, str]]:
    labeled: List[Tuple[MotionSignature, str, float]] = []
    for s in signatures:
        label, conf = _classify_label(s)
        labeled.append((s, label, conf))

    pans = [(s, lab, c) for s, lab, c in labeled if lab in ("pan_left", "pan_right")]
    verts = [(s, lab, c) for s, lab, c in labeled if lab in ("up", "down")]
    claws = [(s, lab, c) for s, lab, c in labeled if lab == "claw_like"]

    pans.sort(key=lambda x: x[2], reverse=True)
    verts.sort(key=lambda x: x[2], reverse=True)

    used: set = set()
    mapping: Dict[str, int] = {}
    confidence: Dict[str, float] = {}

    def take_pan_left_right() -> None:
        left_m = right_m = None
        left_c = right_c = 0.0
        for s, lab, c in pans:
            if s.motor_id in used:
                continue
            if lab == "pan_left" and left_m is None:
                left_m, left_c = s.motor_id, c
            elif lab == "pan_right" and right_m is None:
                right_m, right_c = s.motor_id, c
        if left_m is None or right_m is None:
            horiz = [
                (s, c)
                for s, lab, c in labeled
                if lab in ("pan_left", "pan_right", "mixed") and s.motor_id not in used
            ]
            horiz.sort(key=lambda x: abs(x[0].mean_flow_x), reverse=True)
            picked: List[Tuple[MotionSignature, float]] = []
            for s, c in horiz:
                if s.motor_id in used:
                    continue
                picked.append((s, c))
                if len(picked) >= 2:
                    break
            if len(picked) >= 2:
                (s1, c1), (s2, c2) = picked[0], picked[1]
                if s1.mean_flow_x <= s2.mean_flow_x:
                    left_m, left_c = s1.motor_id, c1
                    right_m, right_c = s2.motor_id, c2
                else:
                    left_m, left_c = s2.motor_id, c2
                    right_m, right_c = s1.motor_id, c1
        if left_m is not None:
            mapping["base_left"] = left_m
            confidence["base_left"] = left_c
            used.add(left_m)
        if right_m is not None:
            mapping["base_right"] = right_m
            confidence["base_right"] = right_c
            used.add(right_m)

    def take_vert() -> None:
        up_m = down_m = None
        up_c = down_c = 0.0
        for s, lab, c in verts:
            if s.motor_id in used:
                continue
            if lab == "up" and up_m is None:
                up_m, up_c = s.motor_id, c
            elif lab == "down" and down_m is None:
                down_m, down_c = s.motor_id, c
        if up_m is None or down_m is None:
            vert_candidates = [
                (s, c)
                for s, lab, c in labeled
                if lab in ("up", "down", "mixed") and s.motor_id not in used
            ]
            vert_candidates.sort(key=lambda x: abs(x[0].mean_flow_y), reverse=True)
            picked: List[Tuple[MotionSignature, float]] = []
            for s, c in vert_candidates:
                if s.motor_id in used:
                    continue
                picked.append((s, c))
                if len(picked) >= 2:
                    break
            if len(picked) >= 2:
                (s1, c1), (s2, c2) = picked[0], picked[1]
                if s1.mean_flow_y <= s2.mean_flow_y:
                    up_m, up_c = s1.motor_id, c1
                    down_m, down_c = s2.motor_id, c2
                else:
                    up_m, up_c = s2.motor_id, c2
                    down_m, down_c = s1.motor_id, c1
        if up_m is not None:
            mapping["arm_up"] = up_m
            confidence["arm_up"] = up_c
            used.add(up_m)
        if down_m is not None:
            mapping["arm_down"] = down_m
            confidence["arm_down"] = down_c
            used.add(down_m)

    take_pan_left_right()
    take_vert()

    claw_id = None
    claw_conf = 0.0
    for s, lab, c in sorted(claws, key=lambda x: x[2], reverse=True):
        if s.motor_id not in used:
            claw_id, claw_conf = s.motor_id, c
            break
    mids = [s.motor_id for s in signatures]
    if claw_id is None:
        remaining = [mid for mid in mids if mid not in used]
        if len(remaining) == 1:
            claw_id, claw_conf = remaining[0], 0.5
        elif remaining:
            rem_sigs = [s for s in signatures if s.motor_id in remaining]
            rem_sigs.sort(key=lambda s: s.flow_mag)
            if rem_sigs:
                claw_id, claw_conf = rem_sigs[0].motor_id, 0.4
    if claw_id is not None:
        mapping["claw"] = claw_id
        confidence["claw"] = claw_conf
        used.add(claw_id)

    remaining = [mid for mid in mids if mid not in used]
    for key in ("base_left", "base_right", "arm_up", "arm_down", "claw"):
        if key not in mapping and remaining:
            mapping[key] = remaining.pop(0)
            confidence[key] = 0.35

    per_motor: Dict[int, str] = {}
    for k, v in mapping.items():
        per_motor[v] = k

    return mapping, confidence, per_motor


def _preview_pair(
    before: np.ndarray,
    after: np.ndarray,
    motor_id: int,
    tag: str,
    scene_top_ratio: float,
) -> np.ndarray:
    h, w = before.shape[:2]
    line_y = int(h * scene_top_ratio)
    b, a = before.copy(), after.copy()
    cv2.line(b, (0, line_y), (w, line_y), (0, 180, 255), 2)
    cv2.line(a, (0, line_y), (w, line_y), (0, 180, 255), 2)
    gb = _capture_gray(_scene_crop(before, scene_top_ratio))
    ga = _capture_gray(_scene_crop(after, scene_top_ratio))
    if gb.shape != ga.shape:
        ga = cv2.resize(ga, (gb.shape[1], gb.shape[0]), interpolation=cv2.INTER_AREA)
    diff = cv2.absdiff(gb, ga)
    diff_bgr = cv2.applyColorMap(
        cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    max_w = 420

    def _sz(img: np.ndarray) -> np.ndarray:
        ih, iw = img.shape[:2]
        if iw > max_w:
            sc = max_w / float(iw)
            return cv2.resize(img, (int(iw * sc), int(ih * sc)), interpolation=cv2.INTER_AREA)
        return img

    b2, a2, d2 = _sz(b), _sz(a), _sz(diff_bgr)
    mh = max(b2.shape[0], a2.shape[0], d2.shape[0])
    pad = lambda im, H: cv2.copyMakeBorder(
        im, 0, H - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40)
    )
    row = np.hstack([pad(b2, mh), pad(a2, mh), pad(d2, mh)])
    cv2.putText(
        row,
        f"m{motor_id} {tag} | once | sonra | fark (ust ROI)",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return row


def run_calibration(
    robot: RobotController,
    grab_frame_fn: Callable[[], np.ndarray],
    motor_ids: List[int],
    pulse_sec: float,
    settle_sec: float,
    confidence_threshold: float,
    scene_top_ratio: float,
    bidirectional: bool,
    min_mean_flow: float,
    http_retries: int,
    save_path: Optional[str] = None,
    show_preview: bool = False,
    preview_window: str = "Kalibrasyon",
) -> MotorMap:
    signatures: List[MotionSignature] = []
    if show_preview:
        cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)

    for motor_id in motor_ids:
        logger.info("--- Motor %s ---", motor_id)
        f0 = grab_frame_fn()
        time.sleep(settle_sec * 0.5)

        ok_i = robot.send_raw(motor_id, CMD_FORWARD, retries=http_retries)
        if not ok_i:
            logger.error(
                "Motor %s ileri komutu HTTP ile gitmedi — URL: %s",
                motor_id,
                robot._url(motor_id, CMD_FORWARD),
            )
        time.sleep(pulse_sec)
        robot.stop_motor(motor_id)
        time.sleep(settle_sec)

        f1 = _best_post_pulse_frame(f0, grab_frame_fn, scene_top_ratio)
        fx, fy, mag, gdiff = measure_motion(f0, f1, scene_top_ratio)

        rfx = rfy = rmag = 0.0
        b_ok = False
        ok_g = True
        if bidirectional:
            time.sleep(settle_sec * 0.4)
            ok_g = robot.send_raw(motor_id, CMD_BACKWARD, retries=http_retries)
            if not ok_g:
                logger.error(
                    "Motor %s geri komutu HTTP ile gitmedi — %s",
                    motor_id,
                    robot._url(motor_id, CMD_BACKWARD),
                )
            time.sleep(pulse_sec)
            robot.stop_motor(motor_id)
            time.sleep(settle_sec)
            f2 = _best_post_pulse_frame(f1, grab_frame_fn, scene_top_ratio)
            rfx, rfy, rmag, _ = measure_motion(f1, f2, scene_top_ratio)
            b_ok = _bidirectional_consistent(fx, fy, mag, rfx, rfy, rmag)

        responded = (mag >= min_mean_flow) or (gdiff >= 5.5)
        sig = MotionSignature(
            motor_id=motor_id,
            mean_flow_x=fx,
            mean_flow_y=fy,
            flow_mag=mag,
            gray_diff_mean=gdiff,
            reverse_mean_flow_x=rfx,
            reverse_mean_flow_y=rfy,
            reverse_flow_mag=rmag,
            bidirectional_ok=b_ok,
            responded=responded,
            http_forward_ok=ok_i,
            http_backward_ok=ok_g if bidirectional else True,
        )
        signatures.append(sig)

        logger.info(
            "  ileri: flow=(%.3f, %.3f) |mag|=%.3f parlaklik_fark=%.2f | http_i=%s",
            fx,
            fy,
            mag,
            gdiff,
            ok_i,
        )
        if bidirectional:
            logger.info(
                "  geri: flow=(%.3f, %.3f) |mag|=%.3f | ters_tutarli=%s | http_g=%s",
                rfx,
                rfy,
                rmag,
                b_ok,
                ok_g,
            )
        if not responded and ok_i:
            logger.warning(
                "  Uyarı: Kamera üst ROI'de belirgin hareket yok — mekanik/eksen veya "
                "süre (calibration_pulse_sec) kontrol edin."
            )

        if show_preview:
            p1 = _preview_pair(f0, f1, motor_id, "ileri", scene_top_ratio)
            cv2.imshow(preview_window, p1)
            cv2.waitKey(280)
            if bidirectional:
                p2 = _preview_pair(f1, f2, motor_id, "geri", scene_top_ratio)
                cv2.imshow(preview_window, p2)
                cv2.waitKey(280)

    if show_preview:
        cv2.destroyWindow(preview_window)

    mapping, conf, _ = _assign_roles(signatures)
    low = [k for k, c in conf.items() if c < confidence_threshold]

    print("\n=== Motor keşif sonucu (üst sahne ROI + optik akış) ===")
    for role in ("base_left", "base_right", "arm_up", "arm_down", "claw"):
        print(f"  {role}: motor {mapping[role]} (güven={conf.get(role, 0):.2f})")

    if low:
        print(
            "\nDüşük güvenli roller: %s\n"
            "Onaylıyor musunuz? [y] evet / [n] hayır (manuel düzenleme)" % ", ".join(low)
        )
        ans = input("> ").strip().lower()
        if ans != "y":
            print(
                "Manuel format: base_left,base_right,arm_up,arm_down,claw "
                "(örn: 11,12,13,14,15)"
            )
            raw = input("> ").strip()
            parts = [int(x) for x in raw.replace(" ", "").split(",")]
            if len(parts) == 5:
                keys = ["base_left", "base_right", "arm_up", "arm_down", "claw"]
                mapping = dict(zip(keys, parts))

    mm = MotorMap(
        base_left=mapping["base_left"],
        base_right=mapping["base_right"],
        arm_up=mapping["arm_up"],
        arm_down=mapping["arm_down"],
        claw=mapping["claw"],
        claw_open_cmd=CMD_FORWARD,
        claw_close_cmd="g",
    )
    out = mm.to_json()
    save_motor_map(out, path=Path(save_path) if save_path else None)
    logger.info("Motor haritası kaydedildi.")
    return mm

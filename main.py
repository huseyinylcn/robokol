"""
Çöp tespiti + HTTP robot kol kontrolü — giriş noktası.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2

from calibration import run_calibration
from config import load_app_config, load_motor_map, save_motor_map
from robot_controller import MotorMap, RobotController
from state_machine import PickStateMachine
from vision import TrashDetector, draw_overlay, make_camera_stream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_robot(cfg, motor_map: MotorMap) -> RobotController:
    return RobotController(
        cfg.robot_base_url,
        motor_map,
        stop_motor_ids=list(cfg.motor_ids),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RoboKol — çöp tespiti ve robot kontrolü")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="app_config.json yolu (varsayılan: proje kökü)",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="motor_map.json yeniden üretilir (sabit kamera: motor_ids sırası; hareketli kamera: optik keşif)",
    )
    args = parser.parse_args()

    cfg = load_app_config(Path(args.config) if args.config else None)
    map_path = Path(cfg.motor_map_path)

    stream = make_camera_stream(
        cfg.camera_url,
        cfg.camera_url_2,
        stitch_horizontal=cfg.camera_stitch_horizontal,
    )
    stream.start()
    if not stream.wait_until_frame(timeout_sec=25.0):
        stream.stop()
        raise SystemExit(
            "Kamera karesi gelmiyor. Kontrol listesi:\n"
            "  • Tek kamera: camera_url doğru mu?\n"
            "  • İki kamera: camera_url ve camera_url_2 ikisi de açık ve aynı ağda mı?\n"
            "  • Her URL tarayıcıda canlı görüntü veriyor mu?\n"
            "  • IP Webcam: /video veya /videofeed (MJPEG)."
        )

    def grab() -> "object":
        for _ in range(40):
            f = stream.read()
            if f is not None and f.size > 0:
                return f
            time.sleep(0.05)
        raise RuntimeError("Kamera karesi geçici olarak kesildi; bağlantıyı kontrol edin.")

    motor_map: MotorMap
    if args.recalibrate or not map_path.is_file():
        ids = list(cfg.motor_ids)
        if len(ids) < 5:
            raise SystemExit("app_config.json içinde en az 5 motor_id gerekli.")
        if cfg.fixed_cameras:
            logger.info(
                "Sabit kamera modu: optik keşif yok. motor_map.json şu sırayla yazılıyor — "
                "[base_left, base_right, arm_up, arm_down, claw] = %s",
                ids,
            )
            motor_map = MotorMap(
                base_left=ids[0],
                base_right=ids[1],
                arm_up=ids[2],
                arm_down=ids[3],
                claw=ids[4],
            )
            save_motor_map(motor_map.to_json(), path=map_path)
        else:
            logger.info("Motor keşfi (optik akış) — robot hareket edecek, alan güvenli olsun.")
            tmp_robot = _build_robot(
                cfg,
                MotorMap(
                    base_left=ids[0],
                    base_right=ids[1],
                    arm_up=ids[2],
                    arm_down=ids[3],
                    claw=ids[4],
                ),
            )
            try:
                motor_map = run_calibration(
                    tmp_robot,
                    grab_frame_fn=grab,
                    motor_ids=ids,
                    pulse_sec=cfg.calibration_pulse_sec,
                    settle_sec=cfg.calibration_settle_sec,
                    confidence_threshold=cfg.calibration_confidence_threshold,
                    scene_top_ratio=cfg.calibration_scene_top_ratio,
                    bidirectional=cfg.calibration_bidirectional,
                    min_mean_flow=cfg.calibration_min_mean_flow,
                    http_retries=cfg.calibration_http_retries,
                    save_path=str(map_path),
                    show_preview=cfg.show_calibration_preview,
                )
            finally:
                tmp_robot.stop_all()
    else:
        data = load_motor_map(map_path)
        if not data:
            raise SystemExit(f"Motor haritası okunamadı: {map_path}")
        motor_map = MotorMap.from_json(data)

    robot = _build_robot(cfg, motor_map)
    detector = TrashDetector(
        cfg.yolo_model,
        cfg.trash_classes,
        cfg.optional_trash_classes,
        imgsz=cfg.inference_imgsz,
    )

    sm = PickStateMachine(
        robot=robot,
        close_area_ratio=cfg.close_area_ratio,
        approach_timeout_sec=cfg.approach_timeout_sec,
        grab_cy_norm_min=cfg.grab_cy_norm_min,
        lost_frames_threshold=cfg.lost_frames_threshold,
        grab_timings=cfg.grab_timings,
        bin_nav_sequence=cfg.bin_nav_sequence,
        release_timings=cfg.release_timings,
        invert_base_turn=cfg.invert_base_turn,
    )

    win = "RoboKol"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    fps_smooth = 30.0
    last = time.perf_counter()
    sub = ""

    try:
        while True:
            frame = stream.read()
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.perf_counter()
            dt = now - last
            last = now
            if dt > 1e-6:
                fps_smooth = cfg.target_fps_smooth * fps_smooth + (1.0 - cfg.target_fps_smooth) * (
                    1.0 / dt
                )

            dets = detector.detect(
                frame,
                cfg.zone_left_max,
                cfg.zone_right_min,
                confidence_min=cfg.detection_confidence_min,
                ignore_bottom_ratio=cfg.detection_ignore_bottom_ratio,
            )
            primary = detector.primary_detection(dets)

            st, sub = sm.step(primary, now)

            motor_status = (
                f"map L{motor_map.base_left} R{motor_map.base_right} "
                f"U{motor_map.arm_up} D{motor_map.arm_down} C{motor_map.claw} | "
                f"{robot.status_line()}"
            )

            vis = draw_overlay(
                frame,
                dets,
                fps_smooth,
                f"{st.name} / {sub}",
                motor_status,
                cfg.zone_left_max,
                cfg.zone_right_min,
                ignore_bottom_ratio=cfg.detection_ignore_bottom_ratio,
                stitch_guide_xs=getattr(stream, "split_xs", None),
                stitch_guide_ys=getattr(stream, "split_ys", None),
            )
            cv2.imshow(win, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                sm.reset()
                logger.info("Durum makinesi sıfırlandı.")
    finally:
        robot.stop_all()
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

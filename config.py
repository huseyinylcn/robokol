"""
Uygulama yapılandırması: sabitler, JSON yükleme/kaydetme, varsayılanlar.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --- Varsayılan yollar ---
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "app_config.json"
DEFAULT_MOTOR_MAP_PATH = PROJECT_ROOT / "motor_map.json"

# --- Robot HTTP ---
DEFAULT_ROBOT_BASE_URL = "http://10.63.160.65"
MOTOR_IDS: Tuple[int, ...] = (11, 12, 13, 14, 15)
CMD_FORWARD = "i"
CMD_BACKWARD = "g"
CMD_STOP = "d"

# --- Kamera (IP Webcam / DroidCam MJPEG önerisi) ---
# IP Webcam: http://IP:8080/video
# DroidCam: genelde benzer MJPEG veya http stream
DEFAULT_CAMERA_URL = "http://192.168.1.100:8080/video"
# İkinci kamera (boş = tek kamera); görüntüler birleştirilerek YOLO + durum makinesi tek kare üzerinde çalışır
DEFAULT_CAMERA_URL_2 = ""
DEFAULT_CAMERA_STITCH_HORIZONTAL = True

# --- Görüntü bölgeleri (normalize x: sol | orta | sağ) ---
ZONE_LEFT_MAX = 1.0 / 3.0
ZONE_RIGHT_MIN = 2.0 / 3.0

# --- YOLO: COCO sınıfları (çöp benzeri) ---
TRASH_CLASS_NAMES: Tuple[str, ...] = (
    "bottle",
    "cup",
    "wine glass",
    "bowl",
)

# handbag sırt çantasına yakın; ince poşet COCO'da yok — opsiyonel genişletme
OPTIONAL_TRASH_ALIASES: Tuple[str, ...] = ("handbag",)

# --- Yaklaşma eşiği: bbox alanı / görüntü alanı ---
DEFAULT_CLOSE_AREA_RATIO = 0.08
# Ortada yaklaşırken bu süre sonunda hâlâ hedef varsa tutmaya zorla (0 = kapalı)
DEFAULT_APPROACH_TIMEOUT_SEC = 8.0
# bbox merkezi görüntüde bu kadar aşağıdaysa (cy_norm >=) ve orta bölgedeyse "yakın" say
DEFAULT_GRAB_CY_NORM_MIN = 0.36

# --- Nesne kaybı ---
LOST_FRAMES_THRESHOLD = 12

# --- Kalibrasyon ---
CALIBRATION_PULSE_SEC = 0.45
CALIBRATION_SETTLE_SEC = 0.35
CALIBRATION_CONFIDENCE_THRESHOLD = 0.55
# Sahne: robot genelde altta; akış sadece üst şeritte ölçülür (0.75 = üst %75)
CALIBRATION_SCENE_TOP_RATIO = 0.75
CALIBRATION_BIDIRECTIONAL = True
# Ortalama akış büyüklüğü (px) bu altında "motor tepki vermedi" sayılır
CALIBRATION_MIN_MEAN_FLOW = 0.4
CALIBRATION_HTTP_RETRIES = 3

# --- Tespit (kendini çöp sayma / FP azaltma) ---
DETECTION_CONFIDENCE_MIN = 0.42
# Alt %35 genelde kol/gövde — bu şeritteki tespitler yok sayılır
DETECTION_IGNORE_BOTTOM_RATIO = 0.35

# --- Sabit kamera (tripod / duvarda): optik akış kalibrasyonu anlamsız ---
DEFAULT_FIXED_CAMERAS = False
# Görüntüde nesne solda ama robot ters yöne dönüyorsa true yap
DEFAULT_INVERT_BASE_TURN = False

# --- Durum makinesi zamanlamaları (saniye; sahaya göre config'ten ayarlanır) ---
DEFAULT_GRAB_TIMINGS = {
    "claw_open": 0.5,
    "arm_down": 0.6,
    "claw_close": 0.55,
    "arm_up": 0.65,
}
# Kutuya gidiş: tespit yok; sabit sekans (README'de ayar önerisi)
DEFAULT_BIN_NAV = [
    ("turn_right", 1.2),
    ("forward", 1.8),
    ("stop", 0.1),
]
DEFAULT_RELEASE_TIMINGS = {
    "claw_open": 0.55,
}


@dataclass
class AppConfig:
    robot_base_url: str = DEFAULT_ROBOT_BASE_URL
    camera_url: str = DEFAULT_CAMERA_URL
    camera_url_2: str = DEFAULT_CAMERA_URL_2
    camera_stitch_horizontal: bool = DEFAULT_CAMERA_STITCH_HORIZONTAL
    motor_ids: List[int] = field(default_factory=lambda: list(MOTOR_IDS))
    trash_classes: List[str] = field(default_factory=lambda: list(TRASH_CLASS_NAMES))
    optional_trash_classes: List[str] = field(
        default_factory=lambda: list(OPTIONAL_TRASH_ALIASES)
    )
    zone_left_max: float = ZONE_LEFT_MAX
    zone_right_min: float = ZONE_RIGHT_MIN
    close_area_ratio: float = DEFAULT_CLOSE_AREA_RATIO
    approach_timeout_sec: float = DEFAULT_APPROACH_TIMEOUT_SEC
    grab_cy_norm_min: float = DEFAULT_GRAB_CY_NORM_MIN
    lost_frames_threshold: int = LOST_FRAMES_THRESHOLD
    calibration_pulse_sec: float = CALIBRATION_PULSE_SEC
    calibration_settle_sec: float = CALIBRATION_SETTLE_SEC
    calibration_confidence_threshold: float = CALIBRATION_CONFIDENCE_THRESHOLD
    calibration_scene_top_ratio: float = CALIBRATION_SCENE_TOP_RATIO
    calibration_bidirectional: bool = CALIBRATION_BIDIRECTIONAL
    calibration_min_mean_flow: float = CALIBRATION_MIN_MEAN_FLOW
    calibration_http_retries: int = CALIBRATION_HTTP_RETRIES
    detection_confidence_min: float = DETECTION_CONFIDENCE_MIN
    detection_ignore_bottom_ratio: float = DETECTION_IGNORE_BOTTOM_RATIO
    fixed_cameras: bool = DEFAULT_FIXED_CAMERAS
    invert_base_turn: bool = DEFAULT_INVERT_BASE_TURN
    motor_map_path: str = str(DEFAULT_MOTOR_MAP_PATH)
    show_calibration_preview: bool = True
    grab_timings: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_GRAB_TIMINGS)
    )
    bin_nav_sequence: List[List[Any]] = field(
        default_factory=lambda: [list(x) for x in DEFAULT_BIN_NAV]
    )
    release_timings: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_RELEASE_TIMINGS)
    )
    yolo_model: str = "yolov8n.pt"
    inference_imgsz: int = 640
    target_fps_smooth: float = 0.92

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        base = cls()
        for k, v in data.items():
            if hasattr(base, k):
                setattr(base, k, v)
        return base

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_app_config(path: Optional[Path] = None) -> AppConfig:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if not p.is_file():
        logger.info("app_config.json bulunamadı, varsayılanlar kullanılıyor.")
        return AppConfig()
    try:
        with p.open("r", encoding="utf-8") as f:
            return AppConfig.from_dict(json.load(f))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("app_config okunamadı (%s), varsayılanlar.", e)
        return AppConfig()


def save_app_config(cfg: AppConfig, path: Optional[Path] = None) -> None:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with p.open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)


def load_motor_map(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    p = Path(path) if path else DEFAULT_MOTOR_MAP_PATH
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_motor_map(data: Dict[str, Any], path: Optional[Path] = None) -> None:
    p = Path(path) if path else DEFAULT_MOTOR_MAP_PATH
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

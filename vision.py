"""
Kamera akışı (thread), YOLOv8 çöp tespiti, bölge ve yakınlık hesapları.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def _open_video_capture(url: str) -> cv2.VideoCapture:
    """HTTP(S) MJPEG için FFMPEG arka ucunu dene; bazı ortamlarda varsayılan backend boş kare döndürür."""
    url = url.strip()
    if url.startswith("http://") or url.startswith("https://"):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            return cap
        cap.release()
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
    return cap


@dataclass
class Detection:
    xyxy: Tuple[float, float, float, float]
    conf: float
    cls_id: int
    name: str
    zone: str  # "left" | "center" | "right"
    area_ratio: float
    cx_norm: float
    cy_norm: float  # bbox merkezinin y/h (büyük = görüntüde aşağı / genelde daha yakın)


class CameraStream:
    """IP Webcam / DroidCam için arka planda en son kareyi tutar."""

    def __init__(self, url: str):
        self.url = url
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._cap = _open_video_capture(self.url)
        if not self._cap.isOpened():
            raise RuntimeError(f"Kamera açılamadı: {self.url}")
        # MJPEG / ağ gecikmesi: ilk okumalar sık sık başarısız; ana iş parçacığında ısıt.
        for attempt in range(60):
            ok, frame = self._cap.read()
            if ok and frame is not None and frame.size > 0:
                with self._lock:
                    self._frame = frame
                logger.info("Kamera akışı hazır (%s deneme).", attempt + 1)
                break
            time.sleep(0.05)
        else:
            logger.warning(
                "Kamera açıldı ama ilk kare okunamadı; arka plan döngüsü denemeye devam edecek."
            )
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def wait_until_frame(
        self, timeout_sec: float = 20.0, poll_sec: float = 0.05
    ) -> bool:
        """İlk geçerli kare gelene kadar bekle (kalibrasyon / ana döngü öncesi)."""
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None and self._frame.size > 0:
                    return True
            time.sleep(poll_sec)
        return False

    def _loop(self) -> None:
        assert self._cap is not None
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.02)

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def _resize_to_height(frame: np.ndarray, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == target_h:
        return frame
    scale = target_h / float(h)
    return cv2.resize(frame, (max(1, int(w * scale)), target_h), interpolation=cv2.INTER_AREA)


def _resize_to_width(frame: np.ndarray, target_w: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == target_w:
        return frame
    scale = target_w / float(w)
    return cv2.resize(frame, (target_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def _stitch_frames(
    frames: List[np.ndarray], horizontal: bool
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Birden fazla kareyi birleştir. Dönüş: (görüntü, dikey dikiş x'leri, yatay dikiş y'leri).
    """
    if not frames:
        raise ValueError("frames boş")
    if len(frames) == 1:
        h, w = frames[0].shape[:2]
        return frames[0].copy(), [], []

    if horizontal:
        ref_h = int(frames[0].shape[0])
        resized = [_resize_to_height(f, ref_h) for f in frames]
        split_xs: List[int] = []
        acc = 0
        for f in resized[:-1]:
            acc += f.shape[1]
            split_xs.append(acc)
        return np.hstack(resized), split_xs, []

    ref_w = int(frames[0].shape[1])
    resized = [_resize_to_width(f, ref_w) for f in frames]
    split_ys: List[int] = []
    acc = 0
    for f in resized[:-1]:
        acc += f.shape[0]
        split_ys.append(acc)
    return np.vstack(resized), [], split_ys


class StitchedCameraStream:
    """İki (veya daha fazla) IP kamera akışını tek karede birleştirir (yatay veya dikey)."""

    def __init__(self, urls: List[str], stitch_horizontal: bool = True):
        urls = [u.strip() for u in urls if u and u.strip()]
        if len(urls) < 2:
            raise ValueError("StitchedCameraStream en az 2 URL gerektirir")
        self.urls = urls
        self.stitch_horizontal = stitch_horizontal
        self._streams = [CameraStream(u) for u in urls]
        self.split_xs: List[int] = []
        self.split_ys: List[int] = []

    def start(self) -> None:
        logger.info(
            "%d kamera akışı başlatılıyor (%s dikiş).",
            len(self._streams),
            "yatay" if self.stitch_horizontal else "dikey",
        )
        for s in self._streams:
            s.start()

    def wait_until_frame(
        self, timeout_sec: float = 25.0, poll_sec: float = 0.05
    ) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.read() is not None:
                return True
            time.sleep(poll_sec)
        return False

    def read(self) -> Optional[np.ndarray]:
        frames = [s.read() for s in self._streams]
        if any(f is None or f.size == 0 for f in frames):
            return None
        stitched, xs, ys = _stitch_frames(frames, self.stitch_horizontal)
        self.split_xs = xs
        self.split_ys = ys
        return stitched

    def stop(self) -> None:
        for s in self._streams:
            s.stop()


def make_camera_stream(
    camera_url: str,
    camera_url_2: str = "",
    stitch_horizontal: bool = True,
):
    """Tek veya çift (birleştirilmiş) kamera nesnesi üret."""
    u1 = (camera_url or "").strip()
    u2 = (camera_url_2 or "").strip()
    if not u1:
        raise ValueError("camera_url boş olamaz")
    if u2:
        return StitchedCameraStream([u1, u2], stitch_horizontal=stitch_horizontal)
    return CameraStream(u1)


class TrashDetector:
    def __init__(
        self,
        model_name: str,
        trash_names: List[str],
        optional_names: List[str],
        imgsz: int = 640,
        device: Optional[str] = None,
    ):
        self.model = YOLO(model_name)
        self.imgsz = imgsz
        self.device = device
        all_names = [n.lower() for n in trash_names] + [n.lower() for n in optional_names]
        self._allowed = set(all_names)
        # cls_id -> is allowed
        self._class_ids: List[int] = []
        names_dict = self.model.names
        for cid, n in names_dict.items():
            if str(n).lower() in self._allowed:
                self._class_ids.append(int(cid))

    def zone_for_cx(self, cx_norm: float, zl: float, zr: float) -> str:
        if cx_norm < zl:
            return "left"
        if cx_norm > zr:
            return "right"
        return "center"

    def detect(
        self,
        frame: np.ndarray,
        zone_left_max: float,
        zone_right_min: float,
        confidence_min: float = 0.25,
        ignore_bottom_ratio: float = 0.0,
    ) -> List[Detection]:
        """
        ignore_bottom_ratio > 0 ise YOLO yalnızca üst (1 - ratio) şeritte çalışır;
        robot kolu / gövde genelde altta kaldığı için yanlış 'çöp' tespiti azalır.
        """
        h, w = frame.shape[:2]
        area_img = float(h * w)
        if ignore_bottom_ratio > 0.0:
            y_end = max(32, int(h * (1.0 - ignore_bottom_ratio)))
            roi = frame[0:y_end, :]
        else:
            y_end = h
            roi = frame

        kwargs = {"imgsz": self.imgsz, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        res = self.model.predict(roi, **kwargs)
        dets: List[Detection] = []
        if not res:
            return dets
        r0 = res[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return dets
        boxes = r0.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            name = str(self.model.names.get(cls_id, str(cls_id)))
            if cls_id not in self._class_ids:
                continue
            conf = float(boxes.conf[i].item())
            if conf < confidence_min:
                continue
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            x1, y1, x2, y2 = xyxy
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            # ROI üstte hizalı; y tam kare ile aynı. Ek güvenlik: merkez alt bantta ise at
            if ignore_bottom_ratio > 0.0 and cy >= h * (1.0 - ignore_bottom_ratio):
                continue
            cx_norm = cx / float(w)
            cy_norm = cy / float(h)
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            area_ratio = (bw * bh) / area_img
            zone = self.zone_for_cx(cx_norm, zone_left_max, zone_right_min)
            dets.append(
                Detection(
                    xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    conf=conf,
                    cls_id=cls_id,
                    name=name,
                    zone=zone,
                    area_ratio=area_ratio,
                    cx_norm=cx_norm,
                    cy_norm=cy_norm,
                )
            )
        dets.sort(key=lambda d: (d.conf, d.area_ratio), reverse=True)
        return dets

    def primary_detection(self, dets: List[Detection]) -> Optional[Detection]:
        return dets[0] if dets else None


def draw_overlay(
    frame: np.ndarray,
    dets: List[Detection],
    fps: float,
    state_name: str,
    motor_status: str,
    zone_left_max: float,
    zone_right_min: float,
    ignore_bottom_ratio: float = 0.0,
    stitch_guide_xs: Optional[List[int]] = None,
    stitch_guide_ys: Optional[List[int]] = None,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    x1 = int(zone_left_max * w)
    x2 = int(zone_right_min * w)
    cv2.line(out, (x1, 0), (x1, h), (80, 80, 80), 1)
    cv2.line(out, (x2, 0), (x2, h), (80, 80, 80), 1)
    for sx in stitch_guide_xs or []:
        if 0 < sx < w:
            cv2.line(out, (sx, 0), (sx, h), (255, 128, 0), 2)
    for sy in stitch_guide_ys or []:
        if 0 < sy < h:
            cv2.line(out, (0, sy), (w, sy), (255, 128, 0), 2)
    if ignore_bottom_ratio > 0.0:
        y_roi = int(h * (1.0 - ignore_bottom_ratio))
        cv2.line(out, (0, y_roi), (w, y_roi), (50, 50, 255), 2)
        cv2.putText(
            out,
            "tespit ROI (alt yok)",
            (8, max(20, y_roi - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (50, 50, 255),
            1,
            cv2.LINE_AA,
        )

    for d in dets:
        bx1, by1, bx2, by2 = [int(round(v)) for v in d.xyxy]
        color = (0, 200, 255) if d.zone == "center" else (0, 165, 255)
        cv2.rectangle(out, (bx1, by1), (bx2, by2), color, 2)
        label = f"{d.name} {d.conf:.2f} [{d.zone}]"
        cv2.putText(
            out,
            label,
            (bx1, max(20, by1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        out,
        f"FPS: {fps:.1f}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"State: {state_name}",
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (180, 255, 180),
        2,
        cv2.LINE_AA,
    )
    y0 = 82
    for i, line in enumerate(motor_status.split("|")):
        cv2.putText(
            out,
            line.strip(),
            (10, y0 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 255),
            1,
            cv2.LINE_AA,
        )
    return out

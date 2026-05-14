"""
Robot HTTP GET kontrolü: motor haritasına göre anlamsal hareketler.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from config import (
    CMD_BACKWARD,
    CMD_FORWARD,
    CMD_STOP,
    MOTOR_IDS,
)

logger = logging.getLogger(__name__)


@dataclass
class MotorMap:
    base_left: int
    base_right: int
    arm_up: int
    arm_down: int
    claw: int
    claw_open_cmd: str = CMD_FORWARD
    claw_close_cmd: str = CMD_BACKWARD

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "MotorMap":
        m = data.get("motor_map", data)
        return cls(
            base_left=int(m["base_left"]),
            base_right=int(m["base_right"]),
            arm_up=int(m["arm_up"]),
            arm_down=int(m["arm_down"]),
            claw=int(m["claw"]),
            claw_open_cmd=str(data.get("claw_open_cmd", CMD_FORWARD)),
            claw_close_cmd=str(data.get("claw_close_cmd", CMD_BACKWARD)),
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "motor_map": {
                "base_left": self.base_left,
                "base_right": self.base_right,
                "arm_up": self.arm_up,
                "arm_down": self.arm_down,
                "claw": self.claw,
            },
            "claw_open_cmd": self.claw_open_cmd,
            "claw_close_cmd": self.claw_close_cmd,
        }

    def all_motor_ids(self) -> List[int]:
        return sorted(
            {
                self.base_left,
                self.base_right,
                self.arm_up,
                self.arm_down,
                self.claw,
            }
        )


class RobotController:
    def __init__(
        self,
        base_url: str,
        motor_map: MotorMap,
        stop_motor_ids: Optional[List[int]] = None,
        timeout: float = 3.5,
        stop_command_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.motor_map = motor_map
        self.timeout = timeout
        self._last_cmd: str = ""
        self._stop_motor_ids: List[int] = (
            list(stop_motor_ids) if stop_motor_ids else list(MOTOR_IDS)
        )
        self._stop_command_retries = max(1, stop_command_retries)

    def _url(self, motor_id: int, cmd: str) -> str:
        return f"{self.base_url}/{motor_id}{cmd}"

    def send_raw(self, motor_id: int, cmd: str, retries: int = 1) -> bool:
        url = self._url(motor_id, cmd)
        last_err: Optional[Exception] = None
        for attempt in range(max(1, retries)):
            try:
                r = requests.get(url, timeout=self.timeout)
                self._last_cmd = url
                if r.status_code >= 400:
                    logger.warning(
                        "HTTP %s: %s (deneme %s/%s)",
                        r.status_code,
                        url,
                        attempt + 1,
                        retries,
                    )
                    time.sleep(0.12)
                    continue
                return True
            except requests.RequestException as e:
                last_err = e
                logger.warning(
                    "İstek hatası %s (deneme %s/%s): %s",
                    url,
                    attempt + 1,
                    retries,
                    e,
                )
                time.sleep(0.15)
        if last_err:
            logger.error("İstek başarısız: %s — %s", url, last_err)
        return False

    def stop_motor(self, motor_id: int) -> None:
        self.send_raw(motor_id, CMD_STOP, retries=self._stop_command_retries)

    def stop_all(self) -> None:
        """Her motora ayrı ayrı 'd' (dur) gönder — 11d, 12d, ... şeklinde."""
        ids = sorted({*self._stop_motor_ids, *self.motor_map.all_motor_ids()})
        for mid in ids:
            self.send_raw(mid, CMD_STOP, retries=self._stop_command_retries)

    def move_left(self) -> None:
        self.send_raw(self.motor_map.base_left, CMD_FORWARD)

    def move_right(self) -> None:
        self.send_raw(self.motor_map.base_right, CMD_FORWARD)

    def move_forward(self) -> None:
        # İleri sürüş: kullanıcı donanımına göre base_right + base_left birlikte veya tek motor.
        # Haritada "ileri" için her iki tabanı da ileri komutla kısa süreli paralel sürüş.
        self.send_raw(self.motor_map.base_left, CMD_FORWARD)
        self.send_raw(self.motor_map.base_right, CMD_FORWARD)

    def move_backward(self) -> None:
        self.send_raw(self.motor_map.base_left, CMD_BACKWARD)
        self.send_raw(self.motor_map.base_right, CMD_BACKWARD)

    def arm_raise(self) -> None:
        self.send_raw(self.motor_map.arm_up, CMD_FORWARD)

    def arm_lower(self) -> None:
        self.send_raw(self.motor_map.arm_down, CMD_FORWARD)

    def claw_open(self) -> None:
        self.send_raw(self.motor_map.claw, self.motor_map.claw_open_cmd)

    def claw_close(self) -> None:
        self.send_raw(self.motor_map.claw, self.motor_map.claw_close_cmd)

    def stop_drive(self) -> None:
        self.stop_motor(self.motor_map.base_left)
        self.stop_motor(self.motor_map.base_right)

    def stop_arm(self) -> None:
        self.stop_motor(self.motor_map.arm_up)
        self.stop_motor(self.motor_map.arm_down)

    def stop_claw(self) -> None:
        self.stop_motor(self.motor_map.claw)

    def pulse(
        self,
        motor_id: int,
        cmd: str,
        duration_sec: float,
        stop_after: bool = True,
    ) -> None:
        self.send_raw(motor_id, cmd)
        time.sleep(duration_sec)
        if stop_after:
            self.stop_motor(motor_id)

    def status_line(self) -> str:
        return f"last_http: {self._last_cmd}"

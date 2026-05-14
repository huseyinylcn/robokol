"""
Çöp toplama davranışı: arama, hizalama, yaklaşma, tutma, kutuya gitme, bırakma.
"""
from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from robot_controller import RobotController
from vision import Detection

logger = logging.getLogger(__name__)


class State(Enum):
    SEARCHING = auto()
    ALIGNING = auto()
    APPROACHING = auto()
    GRABBING = auto()
    MOVING_TO_BIN = auto()
    RELEASING = auto()


class GrabPhase(Enum):
    OPEN = auto()
    ARM_DOWN = auto()
    CLOSE = auto()
    ARM_UP = auto()
    DONE = auto()


class PickStateMachine:
    def __init__(
        self,
        robot: RobotController,
        close_area_ratio: float,
        approach_timeout_sec: float,
        grab_cy_norm_min: float,
        lost_frames_threshold: int,
        grab_timings: Dict[str, float],
        bin_nav_sequence: List[List[Any]],
        release_timings: Dict[str, float],
        invert_base_turn: bool = False,
    ):
        self.robot = robot
        self.close_area_ratio = close_area_ratio
        self.approach_timeout_sec = approach_timeout_sec
        self.grab_cy_norm_min = grab_cy_norm_min
        self.lost_frames_threshold = lost_frames_threshold
        self.grab_timings = grab_timings
        self.bin_nav_sequence = bin_nav_sequence
        self.release_timings = release_timings
        self.invert_base_turn = invert_base_turn

        self.state = State.SEARCHING
        self._lost_streak = 0
        self._grab_phase = GrabPhase.OPEN
        self._phase_start = 0.0
        self._nav_index = 0
        self._nav_phase_start = 0.0
        self._approach_enter_time = 0.0

    def reset(self) -> None:
        self.robot.stop_all()
        self.state = State.SEARCHING
        self._lost_streak = 0
        self._grab_phase = GrabPhase.OPEN
        self._nav_index = 0
        self._approach_enter_time = 0.0

    def _mark_lost(self) -> None:
        self._lost_streak += 1
        if self._lost_streak >= self.lost_frames_threshold:
            logger.warning("Hedef kayboldu — motorlar durduruluyor.")
            self.robot.stop_all()
            self.state = State.SEARCHING
            self._lost_streak = 0

    def _clear_lost(self) -> None:
        self._lost_streak = 0

    def _should_start_grab(self, primary: Detection, now: float) -> bool:
        if primary.zone != "center":
            return False
        if primary.area_ratio >= self.close_area_ratio:
            return True
        if primary.cy_norm >= self.grab_cy_norm_min:
            return True
        if self.approach_timeout_sec > 0.0:
            if (now - self._approach_enter_time) >= self.approach_timeout_sec:
                logger.info(
                    "Yaklaşma zaman aşımı — ortada hedef varken tutma başlatılıyor."
                )
                return True
        return False

    def _turn_left_base(self) -> None:
        if self.invert_base_turn:
            self.robot.move_right()
        else:
            self.robot.move_left()

    def _turn_right_base(self) -> None:
        if self.invert_base_turn:
            self.robot.move_left()
        else:
            self.robot.move_right()

    def step(
        self,
        primary: Optional[Detection],
        now: float,
    ) -> Tuple[State, str]:
        """
        Her karede çağrılır. primary: seçilen çöp tespiti veya None.
        Dönüş: (state, kısa durum metni)
        """
        if primary is None:
            if self.state == State.SEARCHING:
                self._lost_streak = 0
                return self.state, "scanning"
            if self.state in (State.ALIGNING, State.APPROACHING):
                self._mark_lost()
                return self.state, "lost_target"
            if self.state == State.GRABBING:
                return self._step_grabbing(now)
            if self.state == State.MOVING_TO_BIN:
                return self._step_bin_nav(now)
            if self.state == State.RELEASING:
                return self._step_releasing(now)

        if self.state in (State.ALIGNING, State.APPROACHING):
            self._clear_lost()

        if self.state == State.SEARCHING:
            self.state = State.ALIGNING
            return self.state, "found"

        if self.state == State.ALIGNING:
            self.robot.stop_arm()
            self.robot.stop_claw()
            if primary.zone == "left":
                self._turn_left_base()
                return self.state, "align_left"
            if primary.zone == "right":
                self._turn_right_base()
                return self.state, "align_right"
            self.robot.stop_drive()
            self.state = State.APPROACHING
            self._approach_enter_time = now
            return self.state, "aligned"

        if self.state == State.APPROACHING:
            if self._should_start_grab(primary, now):
                self.robot.stop_drive()
                self.robot.stop_arm()
                self.state = State.GRABBING
                self._grab_phase = GrabPhase.OPEN
                self._phase_start = now
                self.robot.claw_open()
                return self.state, "close_enough_grab"
            if primary.zone != "center":
                self.state = State.ALIGNING
                self.robot.stop_drive()
                return self.state, "realign"
            self.robot.move_forward()
            return self.state, "approach"

        if self.state == State.GRABBING:
            return self._step_grabbing(now)

        if self.state == State.MOVING_TO_BIN:
            return self._step_bin_nav(now)

        if self.state == State.RELEASING:
            return self._step_releasing(now)

        return self.state, "idle"

    def _step_grabbing(self, now: float) -> Tuple[State, str]:
        self.robot.stop_drive()
        dt = now - self._phase_start
        if self._grab_phase == GrabPhase.OPEN:
            self.robot.claw_open()
            if dt >= self.grab_timings.get("claw_open", 0.5):
                self.robot.stop_claw()
                self.robot.arm_lower()
                self._grab_phase = GrabPhase.ARM_DOWN
                self._phase_start = now
            return self.state, "grab_open"

        if self._grab_phase == GrabPhase.ARM_DOWN:
            self.robot.arm_lower()
            if dt >= self.grab_timings.get("arm_down", 0.6):
                self.robot.stop_arm()
                self.robot.claw_close()
                self._grab_phase = GrabPhase.CLOSE
                self._phase_start = now
            return self.state, "grab_lower"

        if self._grab_phase == GrabPhase.CLOSE:
            self.robot.claw_close()
            if dt >= self.grab_timings.get("claw_close", 0.55):
                self.robot.stop_claw()
                self.robot.arm_raise()
                self._grab_phase = GrabPhase.ARM_UP
                self._phase_start = now
            return self.state, "grab_close"

        if self._grab_phase == GrabPhase.ARM_UP:
            self.robot.arm_raise()
            if dt >= self.grab_timings.get("arm_up", 0.65):
                self.robot.stop_arm()
                self._grab_phase = GrabPhase.DONE
                self.state = State.MOVING_TO_BIN
                self._nav_index = 0
                self._nav_phase_start = now
            return self.state, "grab_raise"

        return self.state, "grab_done"

    def _step_bin_nav(self, now: float) -> Tuple[State, str]:
        if self._nav_index >= len(self.bin_nav_sequence):
            self.robot.stop_all()
            self.state = State.RELEASING
            self._phase_start = now
            self.robot.claw_open()
            return self.state, "at_bin"

        action, duration = self.bin_nav_sequence[self._nav_index]
        if isinstance(duration, (int, float)):
            dur = float(duration)
        else:
            dur = 0.5

        elapsed = now - self._nav_phase_start
        if elapsed >= dur:
            self.robot.stop_drive()
            self.robot.stop_arm()
            self._nav_index += 1
            self._nav_phase_start = now
            if self._nav_index >= len(self.bin_nav_sequence):
                self.robot.stop_all()
                self.state = State.RELEASING
                self._phase_start = now
                self.robot.claw_open()
                return self.state, "at_bin"
            action, duration = self.bin_nav_sequence[self._nav_index]
            if isinstance(duration, (int, float)):
                dur = float(duration)
            else:
                dur = 0.5

        act = str(action).lower()
        if act == "turn_left":
            self._turn_left_base()
        elif act == "turn_right":
            self._turn_right_base()
        elif act == "forward":
            self.robot.move_forward()
        elif act == "backward":
            self.robot.move_backward()
        elif act == "stop":
            self.robot.stop_all()
        else:
            logger.debug("Bilinmeyen nav aksiyonu: %s", action)
            self.robot.stop_drive()

        return self.state, f"bin_nav:{act}"

    def _step_releasing(self, now: float) -> Tuple[State, str]:
        dt = now - self._phase_start
        if dt >= self.release_timings.get("claw_open", 0.55):
            self.robot.stop_claw()
            self.robot.stop_all()
            self.state = State.SEARCHING
            return self.state, "released"
        return self.state, "releasing"

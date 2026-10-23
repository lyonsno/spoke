"""Lifecycle decisions for House optical surfaces.

This module owns small, AppKit-free transition decisions that consumers should
share instead of copying ``CommandOverlay`` timer and compositor internals.
"""

from __future__ import annotations

from typing import NamedTuple
from dataclasses import dataclass


OPTICAL_BODY_READY_PROGRESS = 0.55
OPTICAL_MAG_SEED_PROGRESS = 0.04
OPTICAL_SLIT_REENTRY_PROGRESS = 0.24


class RetargetDecision(NamedTuple):
    should_retarget: bool
    start_progress: float


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def retarget_progress_for_dismiss(progress: float) -> RetargetDecision:
    """Map dismiss-local body progress onto a lawful summon re-entry point."""
    p = _clamp01(progress)
    if p <= 0.0:
        return RetargetDecision(should_retarget=True, start_progress=0.0)
    if p < OPTICAL_BODY_READY_PROGRESS:
        start_progress = min(p, OPTICAL_MAG_SEED_PROGRESS)
    else:
        # Dismiss is not the inverse of summon: by the time a close has entered
        # the body-ready scalar band, the visible text/body path may already be
        # collapsing through a slit. Re-enter below text release so timer lag
        # cannot flash a full readable body during hammered reversal.
        start_progress = min(p, OPTICAL_SLIT_REENTRY_PROGRESS)
    return RetargetDecision(should_retarget=True, start_progress=start_progress)


@dataclass(frozen=True)
class HouseTransitionFrame:
    phase: str
    progress: float
    tail_progress: float
    configs: dict[str, dict]
    body: dict[str, float]
    text: dict[str, float]


class HouseTransition:
    """Wall-clock House choreography, independent of timers and native carriers."""

    def __init__(self):
        self.phase = "hidden"
        self.started_at = 0.0
        self.start_progress = 0.0
        self._dismiss_text_ceiling = None

    def _position(self, now):
        from . import house_optical_primitive as house

        elapsed = max(0.0, now - self.started_at)
        if self.phase == "summon":
            progress = _clamp01(self.start_progress + elapsed / house.OPTICAL_MATERIALIZATION_S)
            return ("rest" if progress >= 1 else "summon", progress, 0.0)
        if self.phase == "dismiss":
            remaining = self.start_progress * house.OPTICAL_MATERIALIZATION_DISMISS_S - elapsed
            if remaining > 0:
                return "dismiss", _clamp01(remaining / house.OPTICAL_MATERIALIZATION_DISMISS_S), 0.0
            tail = house.OPTICAL_MATERIALIZATION_PUCKER_PREARM_TAIL_PROGRESS - remaining / house.OPTICAL_MATERIALIZATION_PUCKER_TAIL_S
            return ("hidden" if tail >= 1 else "tail", 0.0, min(tail, 1.0))
        return self.phase, (1.0 if self.phase == "rest" else 0.0), 0.0

    def summon(self, now):
        phase, progress, _ = self._position(now)
        self.start_progress = (
            retarget_progress_for_dismiss(progress).start_progress
            if phase == "dismiss" else progress if phase == "summon" else 0.0
        )
        self.phase, self.started_at = "summon", now
        self._dismiss_text_ceiling = None

    def dismiss(self, now, *, exposed_text=None):
        """Close from the carrier's last applied text state, never a future sample."""
        phase, progress, _ = self._position(now)
        if phase in {"hidden", "tail", "dismiss"}:
            return
        if self.phase == "summon":
            # A computed future frame is not evidence that the carrier exposed it.
            self._dismiss_text_ceiling = dict(exposed_text) if exposed_text is not None else dict(
                width_frac=0.0, height_frac=0.0, alpha=0.0)
        else:
            self._dismiss_text_ceiling = None
        self.phase, self.started_at, self.start_progress = "dismiss", now, progress

    def rest(self):
        """Acknowledge that the carrier applied the complete resting frame."""
        self.phase = "rest"

    def frame(self, final, now):
        from . import house_optical_primitive as house

        phase, progress, tail = self._position(now)
        client_id = final["client_id"]
        main = house.materialized_house_optical_shell_config(final, progress)
        seam = house.dismiss_seam_latch_house_shell_config(final, progress)
        radial_progress = tail if phase == "tail" else house.dismiss_pucker_tail_progress_for_close_progress(progress)
        radial = house.dismiss_radial_pucker_house_shell_config(final, radial_progress)
        seam["visible"] = phase == "dismiss" and progress <= house.OPTICAL_MATERIALIZATION_SEAM_OVERLAP_START_PROGRESS
        radial["visible"] = phase == "tail" or (phase == "dismiss" and progress <= house.OPTICAL_MATERIALIZATION_PUCKER_PREARM_START_PROGRESS)
        if phase in {"hidden", "tail"}:
            main = house.hidden_dismiss_main_house_shell_config(final)
        if phase == "rest":
            main = dict(final)
        body = (house.house_dismiss_materialization_fill_state(progress) if phase in {"dismiss", "tail", "hidden"}
                else house.house_materialization_fill_state(progress))
        if phase in {"dismiss", "tail", "hidden"}:
            text = house.house_dismiss_text_collapse_state(progress)
            if self._dismiss_text_ceiling is not None:
                text = {key: min(value, self._dismiss_text_ceiling[key]) for key, value in text.items()}
        else:
            text = dict(width_frac=1.0, height_frac=body["height_frac"], alpha=float(house.optical_entrance_text_ready(progress, body["height_frac"])))
        configs = {client_id: main}
        for suffix, config, offset in (("dismiss_seam", seam, 1), ("dismiss_radial_pucker", radial, 2)):
            config.update(client_id=f"{client_id}.{suffix}", role=final.get("role", "assistant"), z_index=final.get("z_index", 10) - offset)
            configs[config["client_id"]] = config
        for config in configs.values():
            if isinstance(config.get("optical_field"), dict):
                config["optical_field"] = dict(config["optical_field"], transition_phase=phase)
        return HouseTransitionFrame(phase, progress, tail, configs, body, text)

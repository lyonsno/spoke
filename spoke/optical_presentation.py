"""Generationed optical presentation-frame receipts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any


def optical_presentation_config_identity(config: dict[str, Any] | None) -> str | None:
    """Return a stable identity for the compositor config fields humans see."""
    if not config:
        return None
    payload = {
        key: config.get(key)
        for key in (
            "client_id",
            "role",
            "presentation_generation",
            "presentation_requested_state",
            "presentation_publisher_state",
            "center_x",
            "center_y",
            "content_width_points",
            "content_height_points",
            "corner_radius_points",
            "cut_radius_points",
            "visible",
            "opacity",
            "text_mask_progress",
            "materialization_progress",
        )
        if key in config
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class OpticalPresentationFrameBundle:
    """Receipt for one human-presentable command-overlay frame generation."""

    generation_id: int
    requested_state: str
    committed_publisher_state: str
    compositor_config_generation: int | None
    compositor_config_identity: str | None
    window_visible: bool
    window_ordered: bool
    window_alpha: float | None
    text_publication_state: str
    body_publication_state: str
    mask_publication_state: str
    presentation_ack_generation: int | None
    presentation_acknowledged: bool

    def to_trace_fields(self) -> dict[str, object]:
        return {
            "presentation_generation": self.generation_id,
            "presentation_requested_state": self.requested_state,
            "presentation_publisher_state": self.committed_publisher_state,
            "presentation_config_generation": self.compositor_config_generation,
            "presentation_config_identity": self.compositor_config_identity,
            "presentation_window_visible": self.window_visible,
            "presentation_window_ordered": self.window_ordered,
            "presentation_window_alpha": self.window_alpha,
            "presentation_text_state": self.text_publication_state,
            "presentation_body_state": self.body_publication_state,
            "presentation_mask_state": self.mask_publication_state,
            "presentation_ack_generation": self.presentation_ack_generation,
            "presentation_acknowledged": self.presentation_acknowledged,
        }

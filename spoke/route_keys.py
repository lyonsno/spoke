"""Route key destination selection during recording.

Route keys replace timing-based destination selection with explicit, visible,
toggleable selection made during recording. During spacebar-held or latched
recording, tapping `]` or a number row key selects a destination. Tapping the
same key deselects. Only one route key is active at a time. Route key state
resets after each recording.

Keycodes (US keyboard layout):
    ] = 30
    6 = 22, 7 = 26, 8 = 28, 9 = 25, 0 = 29, - = 27, = = 24
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

# ── Keycodes (US keyboard layout) ────────────────────────────

BRACKET_RIGHT_KEYCODE = 30  # `]`

# Number row keycodes for 6-0, -, = (right side, reachable without hand move)
NUMBER_ROW_KEYCODES: tuple[int, ...] = (
    22,  # 6
    26,  # 7
    28,  # 8
    25,  # 9
    29,  # 0
    27,  # -
    24,  # =
)

ALL_ROUTE_KEYCODES: frozenset[int] = frozenset(
    {BRACKET_RIGHT_KEYCODE} | set(NUMBER_ROW_KEYCODES)
)

# ── Key labels for ghost indicators ──────────────────────────

_KEY_LABELS: dict[int, str] = {
    BRACKET_RIGHT_KEYCODE: "]",
    22: "6",
    26: "7",
    28: "8",
    25: "9",
    29: "0",
    27: "-",
    24: "=",
}


def default_bindings() -> dict[int, dict]:
    """Return the default route key bindings.

    Each binding maps a keycode to a dict with:
        destination: str — routing target name
        label: str — display label for ghost indicator
        flavor: str — one of "persistent", "contingent", "one-shot"
    """
    return {
        BRACKET_RIGHT_KEYCODE: {
            "destination": "assistant",
            "label": "]",
            "flavor": "one-shot",
        },
        22: {"destination": "route_6", "label": "6", "flavor": "one-shot"},
        26: {"destination": "route_7", "label": "7", "flavor": "one-shot"},
        28: {"destination": "route_8", "label": "8", "flavor": "one-shot"},
        25: {"destination": "route_9", "label": "9", "flavor": "one-shot"},
        29: {"destination": "route_0", "label": "0", "flavor": "one-shot"},
        27: {"destination": "route_minus", "label": "-", "flavor": "one-shot"},
        24: {"destination": "route_equals", "label": "=", "flavor": "one-shot"},
    }


class RouteKeySelector:
    """Tap-to-toggle route key selection state machine.

    Only one route key is active at a time. Tapping the active key deselects
    it. Tapping a different key switches to it. reset() clears everything.

    Parameters
    ----------
    bindings : dict, optional
        Keycode-to-binding map. Defaults to default_bindings().
    on_change : callable, optional
        Called with (active_keycode, active_destination) whenever the
        selection changes. Both are None when deselected.
    """

    def __init__(
        self,
        bindings: dict[int, dict] | None = None,
        on_change: Callable[[int | None, str | None], None] | None = None,
    ):
        self._bindings = bindings if bindings is not None else default_bindings()
        self._on_change = on_change
        self._active_keycode: int | None = None

    @property
    def active_keycode(self) -> int | None:
        """The currently selected route key keycode, or None."""
        return self._active_keycode

    @property
    def active_destination(self) -> str | None:
        """The destination name of the currently selected route key, or None."""
        if self._active_keycode is None:
            return None
        entry = self._bindings.get(self._active_keycode)
        return entry["destination"] if entry else None

    @property
    def active_binding(self) -> dict | None:
        """The full binding dict of the currently selected route key, or None."""
        if self._active_keycode is None:
            return None
        return self._bindings.get(self._active_keycode)

    def tap(self, keycode: int) -> None:
        """Handle a route key tap. Toggles selection."""
        if keycode not in self._bindings:
            return  # unknown key, ignore

        if self._active_keycode == keycode:
            # Deselect
            self._active_keycode = None
            logger.info("Route key deselected: keycode=%d", keycode)
        else:
            # Select (deactivates any previous)
            self._active_keycode = keycode
            entry = self._bindings[keycode]
            logger.info(
                "Route key selected: keycode=%d dest=%s",
                keycode,
                entry["destination"],
            )

        if self._on_change is not None:
            self._on_change(self._active_keycode, self.active_destination)

    def reset(self) -> None:
        """Clear any active selection. Called at recording end."""
        if self._active_keycode is not None:
            self._active_keycode = None
            logger.info("Route key selection reset")
            if self._on_change is not None:
                self._on_change(None, None)

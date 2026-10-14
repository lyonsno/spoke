"""Native-independent key vocabulary and monitor gesture recognition.

The recognizer owns presses and returns actions. Only the native adapter may
change recording state, schedule timers, or call application callbacks.
"""

from dataclasses import dataclass
from enum import Enum, auto


class KeyKind(Enum):
    DOWN = auto()
    UP = auto()
    MODIFIERS = auto()


@dataclass(frozen=True)
class KeyEvent:
    kind: KeyKind
    keycode: int
    shifted: bool = False
    modified: bool = False
    timestamp_ns: int | None = None


class MonitorAction(Enum):
    BEGIN_HOLD = auto()
    CANCEL_HOLD = auto()
    LEGACY_CHORD = auto()
    LEGACY_ENTER = auto()
    SPACE = auto()
    FORWARD = auto()
    BACKWARD = auto()
    FLEET = auto()
    GREENROOM = auto()


@dataclass(frozen=True)
class GestureResult:
    consumed: bool = False
    actions: tuple[MonitorAction, ...] = ()


class MonitorGesture:
    def __init__(self):
        self.space_owner: str | None = None
        self.space_used = False
        self.space_shifted = False
        self.shift_owner: str | None = None
        self.shift_used = False
        self.pending_space_owner: str | None = None

    @property
    def space_down(self):
        return self.space_owner is not None

    def release_to_dictation(self):
        self.space_owner = None
        self.shift_owner = None
        self.pending_space_owner = None

    def recognize(self, event: KeyEvent, context: str | None, *,
                  capture_state: str, awaiting_release: bool,
                  enter_commands: bool, enter_held: bool = False) -> GestureResult:
        actions = self._settle_pending_space(event, context)
        result = self._recognize(event, context, capture_state=capture_state,
                                 awaiting_release=awaiting_release,
                                 enter_commands=enter_commands, enter_held=enter_held)
        return GestureResult(result.consumed, actions + result.actions)

    def _settle_pending_space(self, event: KeyEvent, context: str | None):
        if self.pending_space_owner is None:
            return ()
        if context != self.pending_space_owner:
            self.pending_space_owner = self.shift_owner = None
        elif event.kind == KeyKind.MODIFIERS and not event.shifted:
            self.pending_space_owner = None
        elif event.kind == KeyKind.DOWN:
            self.pending_space_owner = None
            self.shift_used = True
            if event.keycode == 49 and not event.modified:
                return (MonitorAction.BACKWARD,)
            if not event.modified and event.keycode < 51 and event.keycode not in {36, 48}:
                return (MonitorAction.SPACE,)
        return ()

    def _recognize(self, event: KeyEvent, context: str | None, *, capture_state: str,
                   awaiting_release: bool, enter_commands: bool, enter_held: bool):
        down, up = event.kind == KeyKind.DOWN, event.kind == KeyKind.UP
        held = self.space_down
        if awaiting_release and (context or held) and event.keycode == 49 and (down or up):
            if up:
                self.space_owner = None
            return GestureResult(True)
        if awaiting_release and down and event.keycode in {39, 41}:
            # Completed legacy chords retain their existing suppression path.
            return GestureResult()
        if held and up and event.keycode == 49:
            owner = self.space_owner
            self.space_owner = None
            if capture_state in {'RECORDING', 'LATCHED'}:
                return GestureResult()
            if capture_state == 'WAITING' and enter_commands and enter_held:
                return GestureResult()
            actions = [MonitorAction.CANCEL_HOLD]
            if context == owner and not self.space_used:
                if not self.space_shifted and self.shift_owner == context and not self.shift_used:
                    # A letter can still resolve this as capitalization, not rocking.
                    self.pending_space_owner = context
                else:
                    actions.append(MonitorAction.FORWARD if self.space_shifted else MonitorAction.SPACE)
            return GestureResult(True, tuple(actions))
        eligible = capture_state == 'IDLE' or (capture_state == 'WAITING' and held)
        if not context or not eligible:
            self.shift_owner = None
            return GestureResult(held and down and event.keycode == 49)
        # A press belongs to its original monitor, even if another host becomes key.
        if held and context != self.space_owner:
            self.space_used = True
            self.shift_owner = None
            return GestureResult(down and event.keycode == 49, (MonitorAction.CANCEL_HOLD,))
        if down and event.keycode == 49 and not held and enter_commands and enter_held:
            self.release_to_dictation()
            return GestureResult()
        if event.kind == KeyKind.MODIFIERS:
            if event.shifted and self.shift_owner != context:
                self.shift_owner = context
                self.shift_used = False
                if held:
                    return GestureResult(True, (MonitorAction.CANCEL_HOLD,))
            elif not event.shifted and self.shift_owner is not None:
                owner = self.shift_owner
                self.shift_owner = None
                if owner == context and not self.shift_used:
                    if held:
                        self.space_used = True
                    return GestureResult(True, (MonitorAction.BACKWARD,))
            return GestureResult(True)
        if down and not event.modified:
            if held and event.keycode in {36, 76} and enter_commands:
                was_used = self.space_used
                self.space_used = self.shift_used = True
                if capture_state == 'IDLE' and not was_used:
                    return GestureResult(False, (MonitorAction.LEGACY_ENTER,))
                return GestureResult()
            if held and event.keycode in {39, 41}:
                self.space_used = self.shift_used = True
                return GestureResult(True, (MonitorAction.LEGACY_CHORD,))
            if event.keycode == 49:
                actions = ()
                if not held:
                    self.space_owner = context
                    self.space_used = False
                    self.space_shifted = event.shifted
                    actions = (MonitorAction.BEGIN_HOLD,)
                self.shift_used = True
                return GestureResult(True, actions)
            if held and event.keycode in {44, 47}:
                self.space_used = self.shift_used = True
                action = MonitorAction.FLEET if event.keycode == 44 else MonitorAction.GREENROOM
                return GestureResult(True, (MonitorAction.CANCEL_HOLD, action))
        if down:
            self.shift_used = True
            if held:
                actions = [MonitorAction.CANCEL_HOLD]
                if (not self.space_used and not event.modified and event.keycode < 51
                        and event.keycode not in {36, 48}):
                    actions.append(MonitorAction.SPACE)
                self.space_used = True
                return GestureResult(False, tuple(actions))
        return GestureResult()

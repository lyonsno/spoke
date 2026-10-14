"""Complete monitor traces without Quartz, AppKit, timers, or callbacks."""

import pytest

from spoke.input_lifecycle import KeyEvent, KeyKind, MonitorAction, MonitorGesture


def replay(events, context='fleet'):
    gesture = MonitorGesture()
    state = 'IDLE'
    actions = []
    for event in events:
        result = gesture.recognize(event, context, capture_state=state,
                                   awaiting_release=False, enter_commands=True)
        actions.extend(result.actions)
        if MonitorAction.BEGIN_HOLD in result.actions:
            state = 'WAITING'
        if MonitorAction.CANCEL_HOLD in result.actions:
            state = 'IDLE'
    return gesture, actions


def test_plain_space_owns_press_and_returns_hold_then_space_actions():
    gesture, actions = replay([KeyEvent(KeyKind.DOWN, 49), KeyEvent(KeyKind.UP, 49)])
    assert actions == [MonitorAction.BEGIN_HOLD, MonitorAction.CANCEL_HOLD, MonitorAction.SPACE]
    assert not gesture.space_down
    assert not hasattr(gesture, '_state')
    assert not hasattr(gesture, '_hold_timer')


@pytest.mark.parametrize('key,action', [(44, MonitorAction.FLEET), (47, MonitorAction.GREENROOM)])
def test_summon_chord_does_not_also_supply_a_space(key, action):
    _, actions = replay([KeyEvent(KeyKind.DOWN, 49), KeyEvent(KeyKind.DOWN, key),
                         KeyEvent(KeyKind.UP, key), KeyEvent(KeyKind.UP, 49)])
    assert actions == [MonitorAction.BEGIN_HOLD, MonitorAction.CANCEL_HOLD, action,
                       MonitorAction.CANCEL_HOLD]


def test_rocking_both_directions_has_no_extra_space():
    _, actions = replay([
        KeyEvent(KeyKind.MODIFIERS, 56, shifted=True),
        KeyEvent(KeyKind.DOWN, 49, shifted=True), KeyEvent(KeyKind.UP, 49, shifted=True),
        KeyEvent(KeyKind.MODIFIERS, 56),
        KeyEvent(KeyKind.DOWN, 49), KeyEvent(KeyKind.MODIFIERS, 56, shifted=True),
        KeyEvent(KeyKind.MODIFIERS, 56), KeyEvent(KeyKind.UP, 49),
    ])
    assert [a for a in actions if a in {MonitorAction.FORWARD, MonitorAction.BACKWARD,
                                       MonitorAction.SPACE}] == [MonitorAction.FORWARD, MonitorAction.BACKWARD]


def test_typing_rollover_supplies_one_space_without_consuming_the_letter():
    gesture = MonitorGesture()
    gesture.recognize(KeyEvent(KeyKind.DOWN, 49), 'fleet', capture_state='IDLE',
                      awaiting_release=False, enter_commands=True)
    result = gesture.recognize(KeyEvent(KeyKind.DOWN, 0), 'fleet', capture_state='WAITING',
                               awaiting_release=False, enter_commands=True)
    assert not result.consumed
    assert result.actions == (MonitorAction.CANCEL_HOLD, MonitorAction.SPACE)
    result = gesture.recognize(KeyEvent(KeyKind.UP, 49), 'fleet', capture_state='IDLE',
                               awaiting_release=False, enter_commands=True)
    assert result.actions == (MonitorAction.CANCEL_HOLD,)


def test_context_change_irrevocably_consumes_old_monitor_action():
    gesture = MonitorGesture()
    def send(kind, key, context, state):
        return gesture.recognize(KeyEvent(kind, key), context, capture_state=state,
                                 awaiting_release=False, enter_commands=True)
    send(KeyKind.DOWN, 49, 'fleet', 'IDLE')
    result = send(KeyKind.DOWN, 0, 'greenroom', 'WAITING')
    assert not result.consumed
    assert result.actions == (MonitorAction.CANCEL_HOLD,)
    result = send(KeyKind.UP, 49, 'fleet', 'IDLE')
    assert result.consumed
    assert result.actions == (MonitorAction.CANCEL_HOLD,)


def test_recording_ownership_leaves_monitor_and_preserves_release_for_dictation():
    gesture = MonitorGesture()
    gesture.recognize(KeyEvent(KeyKind.DOWN, 49), 'fleet', capture_state='IDLE',
                      awaiting_release=False, enter_commands=True)
    gesture.release_to_dictation()
    result = gesture.recognize(KeyEvent(KeyKind.UP, 49), 'fleet', capture_state='RECORDING',
                               awaiting_release=False, enter_commands=True)
    assert not result.consumed
    assert not result.actions


@pytest.mark.parametrize('next_context', [None, 'greenroom'])
def test_pending_space_does_not_escape_its_monitor(next_context):
    gesture, actions = replay([KeyEvent(KeyKind.DOWN, 49),
                               KeyEvent(KeyKind.MODIFIERS, 56, shifted=True),
                               KeyEvent(KeyKind.UP, 49, shifted=True)])
    assert MonitorAction.SPACE not in actions
    result = gesture.recognize(KeyEvent(KeyKind.DOWN, 0, shifted=True), next_context,
                               capture_state='IDLE', awaiting_release=False, enter_commands=True)
    assert not result.consumed and not result.actions
    assert gesture.pending_space_owner is None and gesture.shift_owner is None


def test_new_shift_space_resolves_old_pending_rock_then_navigates_forward():
    gesture, actions = replay([
        KeyEvent(KeyKind.DOWN, 49), KeyEvent(KeyKind.MODIFIERS, 56, shifted=True),
        KeyEvent(KeyKind.UP, 49, shifted=True), KeyEvent(KeyKind.DOWN, 49, shifted=True),
        KeyEvent(KeyKind.UP, 49, shifted=True), KeyEvent(KeyKind.MODIFIERS, 56),
    ])
    assert [a for a in actions if a in {MonitorAction.FORWARD, MonitorAction.BACKWARD,
                                       MonitorAction.SPACE}] == [MonitorAction.BACKWARD, MonitorAction.FORWARD]
    assert gesture.pending_space_owner is None and not gesture.space_down

"""Tests for the tray — a speech-native stacked clipboard.

The tray is entered by releasing spacebar while shift is held during
recording. It holds recent transcriptions for review, insertion, sending
to the assistant, or later retrieval.

See docs/keyboard-grammar.md "The tray" for the full spec.
"""

import time
from unittest.mock import MagicMock, call, patch

import pytest


def _make_delegate(main_module, monkeypatch, *, command_client=False):
    """Create a SpokeAppDelegate with mocked sub-components."""
    monkeypatch.setenv("SPOKE_WHISPER_URL", "http://test:8000")

    delegate = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
    delegate._capture = MagicMock()
    delegate._client = MagicMock(supports_streaming=False)
    delegate._detector = MagicMock()
    delegate._menubar = MagicMock()
    delegate._glow = MagicMock()
    delegate._overlay = MagicMock()
    delegate._transcribing = False
    delegate._transcription_token = 0
    delegate._preview_active = False
    delegate._preview_thread = None
    delegate._preview_client = MagicMock()
    delegate._local_mode = False
    delegate._record_start_time = 0.0
    delegate._cap_fired = False
    delegate._transcribe_start = time.monotonic()
    delegate._last_preview_text = ""
    delegate._models_ready = True
    delegate._mic_ready = True
    delegate._preview_session_token = 0
    delegate._preview_done = MagicMock()
    delegate._preview_done.set = MagicMock()
    delegate._local_inference_lock = MagicMock()
    # Tray state
    delegate._tray_stack = []
    delegate._tray_index = 0
    delegate._tray_active = False
    delegate._tray_deck = "text"
    # Typed coordination surface stack
    from spoke.coordination_surfaces import CoordinationStack, build_default_registry
    delegate._surface_registry = build_default_registry()
    delegate._coordination_stack = CoordinationStack(registry=delegate._surface_registry)
    # Recovery state (implementation detail of tray)
    delegate._pre_paste_clipboard = None
    delegate._verify_paste_text = None
    delegate._verify_paste_attempt = 0
    delegate._recovery_saved_clipboard = None
    delegate._recovery_text = None
    delegate._recovery_clipboard_state = "idle"
    delegate._recovery_pending_insert = None
    delegate._recovery_hold_active = False
    delegate._recovery_retry_pending = False
    delegate.performSelectorOnMainThread_withObject_waitUntilDone_ = MagicMock()

    if command_client:
        delegate._command_client = MagicMock()
        delegate._command_client.history = []
        delegate._command_overlay = MagicMock()
        delegate._command_overlay._visible = False
    else:
        delegate._command_client = None
        delegate._command_overlay = None

    return delegate


def _run_main_thread_selector(delegate):
    def _dispatch(selector, payload, wait):
        getattr(delegate, selector.replace(":", "_"))(payload)
    delegate.performSelectorOnMainThread_withObject_waitUntilDone_.side_effect = _dispatch


class TestTrayEntry:
    """Shift+release during recording enters the tray."""

    def test_shift_release_spawns_tray_transcription(self, main_module, monkeypatch):
        """Shift+release during recording should spawn tray transcription thread."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._capture.stop.return_value = b"audio"
        d._record_start_time = time.monotonic() - 2.0

        with patch.object(main_module.threading, "Thread") as MockThread:
            MockThread.return_value = MagicMock()
            d._on_hold_end(shift_held=True)

        # Tray transcription thread should be spawned
        MockThread.assert_called_once()
        # tray_active is deferred until transcription completes
        assert d._transcribing is True

    def test_shift_release_does_not_send_to_command(self, main_module, monkeypatch):
        """Tray must intercept — text should not go to assistant."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._send_text_as_command = MagicMock()
        d._capture.stop.return_value = b"audio"
        d._record_start_time = time.monotonic() - 2.0

        with patch.object(main_module.threading, "Thread"):
            d._on_hold_end(shift_held=True)

        d._send_text_as_command.assert_not_called()

    def test_enter_held_sends_to_command(self, main_module, monkeypatch):
        """Enter held at release should send directly to assistant (fast path)."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._capture.stop.return_value = b"audio"
        d._record_start_time = time.monotonic() - 2.0

        with patch.object(main_module.threading, "Thread") as MockThread:
            MockThread.return_value = MagicMock()
            d._on_hold_end(shift_held=False, enter_held=True)

        # Should route to command pathway, not tray
        assert d._tray_active is not True

    def test_tray_empty_transcription_does_not_promote_preview_text(
        self, main_module, monkeypatch
    ):
        """An empty final tray transcript should not enter tray with preview text."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._transcription_token = 7
        d._transcribing = True
        d._last_preview_text = "partial tray preview"

        d.trayTranscriptionComplete_({"token": 7, "text": ""})

        assert d._tray_stack == []
        assert d._tray_active is False
        d._menubar.set_status_text.assert_called_with("Ready — hold spacebar")

    def test_tray_failed_transcription_does_not_promote_preview_text(
        self, main_module, monkeypatch
    ):
        """A failed final tray transcript should not enter tray with preview text."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._transcription_token = 7
        d._transcribing = True
        d._last_preview_text = "partial tray preview"

        d.trayTranscriptionFailed_({"token": 7, "error": "tray final failed"})

        assert d._tray_stack == []
        assert d._tray_active is False
        d._menubar.set_status_text.assert_called_with("tray final failed")


class TestTrayStack:
    """Tray stack lifecycle — push, navigate, consume, delete."""

    def test_enter_tray_records_user_owned_entry(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch, command_client=True)

        d._enter_tray("hello world")

        entry = d._tray_stack[0]
        assert isinstance(entry, main_module.TrayEntry)
        assert entry.text == "hello world"
        assert entry.owner == "user"
        assert entry.acknowledged is True

    def test_enter_tray_pushes_to_stack(self, main_module, monkeypatch):
        """Entering the tray should push text onto the stack."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)

        d._enter_tray("hello world")

        assert d._tray_stack == ["hello world"]
        assert d._tray_index == 0
        assert d._tray_active is True
        d._glow.show_tray_dim.assert_called_once()

    def test_dismiss_tray_hides_tray_dim(self, main_module, monkeypatch):
        """Leaving the tray should release the half-strength dimmer."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_active = True

        d._dismiss_tray()

        d._glow.hide.assert_called_once()

    def test_multiple_entries_stack(self, main_module, monkeypatch):
        """Multiple tray entries should stack with most recent on top."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)

        d._enter_tray("first")
        d._dismiss_tray()
        d._enter_tray("second")

        assert d._tray_stack == ["first", "second"]
        assert d._tray_index == 1  # viewing the most recent

    def test_navigate_down(self, main_module, monkeypatch):
        """Navigate down should show older entries."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = ["old", "middle", "newest"]
        d._tray_index = 2
        d._tray_active = True

        d._tray_navigate_down()

        assert d._tray_index == 1

    def test_navigate_up(self, main_module, monkeypatch):
        """Navigate up should show more recent entries."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = ["old", "middle", "newest"]
        d._tray_index = 1
        d._tray_active = True

        d._tray_navigate_up()

        assert d._tray_index == 2

    def test_navigate_up_at_top_stays_on_current_entry(self, main_module, monkeypatch):
        """Navigating up at the most recent entry should clamp at the top."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = ["old", "newest"]
        d._tray_index = 1
        d._tray_active = True

        d._tray_navigate_up()

        assert d._tray_active is True
        assert d._tray_index == 1
        d._overlay.show_tray.assert_not_called()

    def test_navigate_down_stops_at_bottom(self, main_module, monkeypatch):
        """Navigating down past the oldest entry should stop (no wrap)."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = ["oldest", "newest"]
        d._tray_index = 0
        d._tray_active = True

        d._tray_navigate_down()

        assert d._tray_index == 0  # didn't move

    def test_delete_entry(self, main_module, monkeypatch):
        """Delete should remove the current entry from the stack."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = ["keep", "delete me", "also keep"]
        d._tray_index = 1
        d._tray_active = True

        d._tray_delete_current()

        assert "delete me" not in d._tray_stack
        assert len(d._tray_stack) == 2

    def test_delete_last_entry_dismisses(self, main_module, monkeypatch):
        """Deleting the only entry should dismiss the tray."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = ["only one"]
        d._tray_index = 0
        d._tray_active = True

        d._tray_delete_current()

        assert d._tray_stack == []
        assert d._tray_active is False

    def test_insert_consumes_entry(self, main_module, monkeypatch):
        """Inserting text should remove it from the stack."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = ["first", "insert this"]
        d._tray_index = 1
        d._tray_active = True
        d._recovery_text = "insert this"

        with patch("spoke.__main__.has_focused_text_input", return_value=True), \
             patch("spoke.__main__.inject_text"):
            d._tray_insert_current()

        assert "insert this" not in d._tray_stack

    def test_dismiss_preserves_stack(self, main_module, monkeypatch):
        """Dismissing the tray should preserve the stack for re-entry."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = ["preserved"]
        d._tray_index = 0
        d._tray_active = True

        d._dismiss_tray()

        assert d._tray_active is False
        assert d._tray_stack == ["preserved"]

    def test_reentry_shows_top(self, main_module, monkeypatch):
        """Re-entering the tray should show the top of the stack."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = ["old", "recent"]
        d._tray_index = 0  # was viewing old item

        d._enter_tray("newest")

        assert d._tray_index == 2  # viewing the newest

    def test_switch_deck_moves_between_text_and_coordination_entries(
        self, main_module, monkeypatch
    ):
        """The tray can rock between ordinary text and coordination decks."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)

        d._add_tray_entry("ordinary one", activate=True)
        d._add_diaulos_card_to_stack(
            {
                "diaulos": "chairside-sparkwright",
                "diaulos_id": "dia-test",
                "display_name": "Chairside Sparkwright",
                "topos": "projects/spoke/topoi/test.md",
            },
            activate=True,
        )
        d._add_tray_entry("ordinary two", activate=False)

        assert d._tray_deck == "coordination"
        assert d._tray_index == 1

        d._tray_switch_deck()

        assert d._tray_deck == "text"
        assert d._tray_index == 2
        d._overlay.show_tray.assert_called_with("ordinary two", owner="user")

        d._tray_switch_deck()

        assert d._tray_deck == "coordination"
        assert d._tray_index == 1
        assert "Chairside Sparkwright" in d._overlay.show_tray.call_args.args[0]

    def test_navigation_skips_entries_outside_active_deck(
        self, main_module, monkeypatch
    ):
        """Up/down only walks entries from the active deck."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = [
            main_module.TrayEntry("old text"),
            main_module.TrayEntry("diaulos card", kind="diaulos_card"),
            main_module.TrayEntry("new text"),
        ]
        d._tray_active = True
        d._detector.tray_active = True
        d._tray_deck = "text"
        d._tray_index = 2

        d._tray_navigate_down()

        assert d._tray_index == 0

        d._tray_deck = "coordination"
        d._tray_index = 1

        d._tray_navigate_up()
        d._tray_navigate_down()

        assert d._tray_index == 1

    def test_coordination_entry_space_does_not_insert_or_consume(
        self, main_module, monkeypatch
    ):
        """Stack cards are readable surfaces, not pasteable tray text."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = [
            main_module.TrayEntry("Diaulos card body", kind="diaulos_card"),
        ]
        d._tray_active = True
        d._detector.tray_active = True
        d._tray_deck = "coordination"
        d._tray_index = 0

        with patch("spoke.__main__.inject_text") as inject:
            d._tray_insert_current()

        inject.assert_not_called()
        assert len(d._tray_stack) == 1
        assert d._tray_active is True
        assert d._detector.tray_active is True
        d._overlay.hide.assert_not_called()
        d._menubar.set_status_text.assert_called_with("Stack entry is read-only")

    def test_coordination_entry_send_does_not_route_or_consume(
        self, main_module, monkeypatch
    ):
        """Stack cards do not become assistant prompts through the tray send path."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = [
            main_module.TrayEntry("Diaulos card body", kind="diaulos_card"),
        ]
        d._tray_active = True
        d._detector.tray_active = True
        d._tray_deck = "coordination"
        d._tray_index = 0
        d._send_text_as_command = MagicMock()

        d._tray_send_current()

        d._send_text_as_command.assert_not_called()
        assert len(d._tray_stack) == 1
        assert d._tray_active is True
        assert d._detector.tray_active is True
        d._menubar.set_status_text.assert_called_with("Stack entry is read-only")

    def test_assistant_add_to_closed_tray_stays_silent(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        _run_main_thread_selector(d)

        result = d._add_assistant_content_to_tray("saved result")

        assert d._tray_active is False
        d._overlay.show_tray.assert_not_called()
        entry = d._tray_stack[0]
        assert isinstance(entry, main_module.TrayEntry)
        assert entry.owner == "assistant"
        assert entry.acknowledged is False
        assert result["tray_visible"] is False

    def test_assistant_add_to_open_tray_surfaces_immediately(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        _run_main_thread_selector(d)
        d._tray_stack = [main_module.TrayEntry("existing")]
        d._tray_index = 0
        d._tray_active = True
        d._detector.tray_active = True

        result = d._add_assistant_content_to_tray("new result")

        assert d._tray_active is True
        assert d._tray_index == 1
        d._overlay.show_tray.assert_called_with("new result", owner="assistant")
        entry = d._tray_stack[1]
        assert entry.owner == "assistant"
        assert entry.acknowledged is False
        assert result["tray_visible"] is True

    def test_bottom_insert_stays_silent_and_preserves_active_entry(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = [
            main_module.TrayEntry("older"),
            main_module.TrayEntry("current"),
        ]
        d._tray_index = 1
        d._tray_active = True
        d._detector.tray_active = True

        entry = d._add_tray_entry("saved fallback", activate=False, position="bottom")

        assert entry.text == "saved fallback"
        assert [item.text for item in d._tray_stack] == ["saved fallback", "older", "current"]
        assert d._tray_index == 2
        d._overlay.show_tray.assert_not_called()

    def test_recalling_assistant_entry_acknowledges_it(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = [
            main_module.TrayEntry("from assistant", owner="assistant", acknowledged=False)
        ]
        d._capture.stop.return_value = b"audio"
        d._record_start_time = time.monotonic() - 0.5

        d._on_hold_end(shift_held=True)

        entry = d._tray_stack[0]
        assert entry.acknowledged is True
        d._overlay.show_tray.assert_called_with("from assistant", owner="user")

    def test_acknowledged_assistant_entry_renders_as_user_on_revisit(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = [
            main_module.TrayEntry("user text"),
            main_module.TrayEntry("from assistant", owner="assistant", acknowledged=True),
        ]
        d._tray_index = 0
        d._tray_active = True
        d._detector.tray_active = True

        # Navigate up to the acknowledged assistant entry
        d._on_tray_navigate_up()

        entry = d._tray_stack[1]
        assert entry.owner == "assistant"
        assert entry.acknowledged is True
        d._overlay.show_tray.assert_called_with("from assistant", owner="user")

    def test_tray_stack_assertions_use_entry_not_string(
        self, main_module, monkeypatch
    ):
        """Verify that TrayEntry equality with strings doesn't mask ownership."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        _run_main_thread_selector(d)

        d._add_assistant_content_to_tray("same text")
        d._enter_tray("same text")

        # Both entries have the same .text but different ownership
        assert len(d._tray_stack) == 2
        assert d._tray_stack[0].owner == "assistant"
        assert d._tray_stack[1].owner == "user"
        # String equality hides the difference — this is the hazard
        assert d._tray_stack[0] == "same text"
        assert d._tray_stack[1] == "same text"
        # But direct TrayEntry comparison distinguishes them
        assert d._tray_stack[0] != d._tray_stack[1]

    def test_dual_write_focus_matches_legacy_tray_on_active_top_insert(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch, command_client=True)

        d._add_tray_entry("old", activate=True)
        d._add_tray_entry("new", activate=True, position="top")

        assert d._tray_stack[d._tray_index].text == "new"
        assert d._coordination_stack.primary is not None
        assert d._coordination_stack.primary.payload["text"] == "new"

    def test_dual_write_focus_tracks_legacy_tray_navigation(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch, command_client=True)

        d._add_tray_entry("old", activate=True)
        d._add_tray_entry("new", activate=True)
        d._tray_navigate_down()

        assert d._tray_stack[d._tray_index].text == "old"
        assert d._coordination_stack.primary is not None
        assert d._coordination_stack.primary.payload["text"] == "old"

    def test_dual_write_acknowledges_matching_surface_entry(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        entry = d._add_tray_entry("from assistant", owner="assistant", activate=False)
        surface = d._coordination_stack.find_by_id(entry.coordination_surface_id)
        assert surface is not None
        assert surface.acknowledged is False

        d._tray_active = True
        d._tray_index = 0
        d._show_tray_current(acknowledge=True)

        assert entry.acknowledged is True
        assert surface.acknowledged is True

    def test_dual_write_removes_matching_surface_entry_on_delete(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        keep = d._add_tray_entry("keep", activate=True)
        delete_me = d._add_tray_entry("delete me", activate=True)
        assert d._coordination_stack.find_by_id(keep.coordination_surface_id) is not None
        assert d._coordination_stack.find_by_id(delete_me.coordination_surface_id) is not None

        d._tray_delete_current()

        assert d._coordination_stack.find_by_id(delete_me.coordination_surface_id) is None
        assert d._coordination_stack.find_by_id(keep.coordination_surface_id) is not None


class TestDirectiveInboxStackPressure:
    """Source-signed directive inbox entries become stack pressure, not truth."""

    def test_inbox_json_entry_maps_to_handle_first_tray_pressure(self, main_module):
        payload = [
            {
                "path": "metadosis/upstream-directives/route-smoke.md",
                "target_path": "projects/epistaxis/topoi/badgestall.md",
                "target_diaulos": "badgestall-proctocolips-cartographer",
                "target_diaulos_id": None,
                "packet_exists": True,
                "source": {
                    "kind": "diaulos",
                    "sign": "opus-miserena-id-cartographer",
                    "diaulos": "opus-miserena-id-cartographer",
                    "diaulos_id": "dia-b715a7f9-ec67-4dcd-80a3-12688844f177",
                    "topos": "projects/epistaxis/topoi/codex-opus.md",
                    "refs": [
                        "metadosis/coordination-packets/perceptasia-spoke_switchblade-reticule-post-office_2026-05-22.md"
                    ],
                },
                "intent_class": "notice",
                "authority_basis": "peer-diaulos pressure; non-binding unless separately authorized",
                "required_rereads": [
                    "metadosis/coordination-packets/perceptasia-spoke_switchblade-reticule-post-office_2026-05-22.md",
                    "metadosis/source-signed-diaulos-switchboard_2026-05-20.md",
                ],
                "scope_refs": [],
                "delivery_state": "pending-durable-badge",
                "disposition_state": "pending",
                "supersedes": None,
            }
        ]

        entries = main_module.directive_inbox_json_to_tray_entries(
            main_module.json.dumps(payload)
        )

        assert len(entries) == 1
        entry = entries[0]
        assert isinstance(entry, main_module.TrayEntry)
        assert entry.owner == "directive"
        assert entry.acknowledged is False
        assert entry.kind == "directive_inbox_pressure"
        assert entry.text.startswith(
            "opus-miserena-id-cartographer -> badgestall-proctocolips-cartographer"
        )
        assert "dia-b715a7f9-ec67-4dcd-80a3-12688844f177" not in entry.text
        assert "notice" in entry.text
        assert "Disposition: pending" in entry.text
        assert "Delivery: pending-durable-badge" in entry.text
        assert "Required rereads:" in entry.text
        assert entry.metadata["source"]["handle"] == "opus-miserena-id-cartographer"
        assert entry.metadata["source"]["id"] == "dia-b715a7f9-ec67-4dcd-80a3-12688844f177"
        assert entry.metadata["target"]["handle"] == "badgestall-proctocolips-cartographer"
        assert entry.metadata["target"]["id"] is None
        assert entry.metadata["disposition_state"] == "pending"
        assert entry.metadata["delivery_state"] == "pending-durable-badge"

    def test_delegate_skips_consumed_inbox_rows_as_active_pressure(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        payload = [
            {
                "path": "metadosis/upstream-directives/route-smoke.md",
                "target_path": "projects/epistaxis/topoi/badgestall.md",
                "target_diaulos": "badgestall-proctocolips-cartographer",
                "target_diaulos_id": "dia-target",
                "packet_exists": True,
                "source": {
                    "kind": "diaulos",
                    "sign": "opus-miserena-id-cartographer",
                    "diaulos": "opus-miserena-id-cartographer",
                    "diaulos_id": "dia-source",
                    "topos": None,
                    "refs": [],
                },
                "intent_class": "directive",
                "authority_basis": "operator sign",
                "required_rereads": [],
                "scope_refs": [],
                "delivery_state": "pending-durable-badge",
                "disposition_state": "consumed",
                "supersedes": None,
            }
        ]

        with patch.object(main_module.subprocess, "run") as run:
            result = d._add_directive_inbox_json_to_stack(main_module.json.dumps(payload))

        run.assert_not_called()
        assert result == {"status": "added", "stack_size": 0, "entry_count": 0}
        assert d._tray_stack == []
        assert all(
            getattr(entry, "kind", None) != "directive_inbox_pressure"
            for entry in d._tray_stack
        )

    def test_directive_inbox_accepts_spool_envelope_rows(self, main_module):
        payload = {
            "schema": "epistaxis.directive_inbox_spool.v1",
            "generated_at": "2026-05-23T00:00:00Z",
            "rows": [
                {
                    "path": "metadosis/upstream-directives/pending.md",
                    "target_path": "projects/epistaxis/topoi/badgestall.md",
                    "target_diaulos": "badgestall-proctocolips-cartographer",
                    "target_diaulos_id": None,
                    "packet_exists": True,
                    "source": {
                        "kind": "diaulos",
                        "sign": "opus-miserena-id-cartographer",
                        "diaulos": "opus-miserena-id-cartographer",
                        "diaulos_id": "dia-source",
                        "topos": None,
                        "refs": [],
                    },
                    "intent_class": "notice",
                    "authority_basis": "peer-diaulos pressure; non-binding unless separately authorized",
                    "required_rereads": ["metadosis/source-signed-diaulos-switchboard_2026-05-20.md"],
                    "scope_refs": [],
                    "delivery_state": "pending-durable-badge",
                    "disposition_state": "pending",
                    "supersedes": None,
                },
                {
                    "path": "metadosis/upstream-directives/consumed.md",
                    "target_path": "projects/epistaxis/topoi/badgestall.md",
                    "target_diaulos": "badgestall-proctocolips-cartographer",
                    "target_diaulos_id": None,
                    "packet_exists": True,
                    "source": {
                        "kind": "diaulos",
                        "sign": "opus-miserena-id-cartographer",
                        "diaulos": "opus-miserena-id-cartographer",
                        "diaulos_id": "dia-source",
                        "topos": None,
                        "refs": [],
                    },
                    "intent_class": "notice",
                    "authority_basis": "peer-diaulos pressure; non-binding unless separately authorized",
                    "required_rereads": [],
                    "scope_refs": [],
                    "delivery_state": "pending-durable-badge",
                    "disposition_state": "consumed",
                    "supersedes": None,
                },
            ],
        }

        entries = main_module.directive_inbox_json_to_tray_entries(
            main_module.json.dumps(payload)
        )

        assert len(entries) == 1
        assert entries[0].kind == "directive_inbox_pressure"
        assert entries[0].metadata["path"] == "metadosis/upstream-directives/pending.md"
        assert entries[0].metadata["disposition_state"] == "pending"
        assert entries[0].metadata["required_rereads"] == [
            "metadosis/source-signed-diaulos-switchboard_2026-05-20.md"
        ]

    def test_directive_inbox_rejects_non_spool_mapping_payload(self, main_module):
        with pytest.raises(ValueError, match="list of entries or spool envelope"):
            main_module.directive_inbox_json_to_tray_entries(
                main_module.json.dumps({"rows": [], "schema": "wrong.schema"})
            )

    def test_directive_inbox_rejects_spool_envelope_without_list_rows(self, main_module):
        with pytest.raises(ValueError, match="spool envelope rows must be a list"):
            main_module.directive_inbox_json_to_tray_entries(
                main_module.json.dumps({
                    "schema": "epistaxis.directive_inbox_spool.v1",
                    "rows": {"not": "a list"},
                })
            )


class TestTrayGestures:
    """Gestures available while the tray is active."""

    def test_spacebar_hold_from_tray_starts_recording(self, main_module, monkeypatch):
        """Plain spacebar hold from tray should dismiss tray and start recording."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_active = True
        d._tray_stack = ["old text"]
        d._tray_index = 0
        d._recovery_text = "old text"
        d._detector._shift_at_press = False  # plain spacebar, no shift
        d._detector._shift_latched = False

        d._on_hold_start()

        # Should dismiss tray and start recording
        d._capture.start.assert_called_once()
        assert d._tray_active is False

    def test_shift_spacebar_hold_from_tray_navigates(self, main_module, monkeypatch):
        """Shift+spacebar hold from tray should wait for release (navigation)."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_active = True
        d._tray_stack = ["old text"]
        d._tray_index = 0
        d._recovery_text = "old text"
        d._detector._shift_at_press = True  # shift held

        d._on_hold_start()

        # Should NOT start recording — waits for release to navigate
        d._capture.start.assert_not_called()
        assert d._recovery_hold_active is True

    def test_enter_from_tray_does_not_send_to_assistant(self, main_module, monkeypatch):
        """Bare Enter with the tray visible should not route to the assistant."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_active = True
        d._tray_stack = ["send this"]
        d._tray_index = 0
        d._send_text_as_command = MagicMock()

        d._on_tray_enter_pressed()

        d._send_text_as_command.assert_not_called()
        assert d._tray_stack == ["send this"]
        assert d._tray_active is True

    def test_short_shift_hold_recalls_into_tray(self, main_module, monkeypatch):
        """Short shift-hold should enter tray with last tray entry."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = ["previous text"]
        d._capture.stop.return_value = b"audio"
        d._record_start_time = time.monotonic() - 0.5  # < 800ms

        d._on_hold_end(shift_held=True)

        assert d._tray_active is True
        assert d._tray_index == 0  # viewing the existing entry


class TestTrayRecoveryUnification:
    """Paste failure enters the tray automatically."""

    def test_paste_failure_enters_tray(self, main_module, monkeypatch):
        """OCR verification failure should enter tray."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)

        d._enter_tray("failed paste text")

        assert d._tray_active is True
        assert "failed paste text" in d._tray_stack


class TestOperatorPingTokenVisualDelegate:
    def test_tray_summon_repaints_operator_ping_tokens_after_tray_body(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._add_tray_entry("visible tray body", activate=True)
        d._maybe_show_operator_ping_token_smoke = MagicMock()

        sequence = MagicMock()
        sequence.attach_mock(d._overlay.show_tray, "show_tray")
        sequence.attach_mock(
            d._maybe_show_operator_ping_token_smoke,
            "show_operator_ping_tokens",
        )

        d._show_tray_current()

        assert sequence.mock_calls[:2] == [
            call.show_tray("visible tray body", owner="user"),
            call.show_operator_ping_tokens(),
        ]

    def test_backend_ping_events_project_to_overlay_visuals_without_tray_rows(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._overlay = MagicMock()

        visuals = d._show_operator_ping_tokens_from_events(
            [
                {
                    "kind": "operator_ping.created",
                    "event_id": "epistaxis.event.v1:operator_ping.created:spoke:ping-1",
                    "operator_ping": {
                        "ping_id": "ping-1",
                        "created_at": "2026-05-22T12:00:00Z",
                        "diaulos": "chairside-sparkwright",
                        "reason_token": "question",
                    },
                }
            ],
            stack_body_frame=(100.0, 80.0, 360.0, 96.0),
        )

        assert [visual.ping_id for visual in visuals] == ["ping-1"]
        assert d._tray_stack == []
        assert d._coordination_stack.entries == []
        d._overlay.show_operator_ping_token_visuals.assert_called_once_with(visuals)

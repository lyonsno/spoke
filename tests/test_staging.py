"""Tests for staging mode — the intermediate state between recording and commit.

Staging mode is entered by releasing spacebar while shift is held during
recording. The transcribed text stays in the overlay for the user to decide:
insert at cursor, send to assistant, re-record, or dismiss.

See docs/keyboard-grammar.md "Staging mode (planned)" for the full spec.
"""

import time
from unittest.mock import MagicMock, patch


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
    delegate._preview_session_token = 0
    delegate._preview_done = MagicMock()
    delegate._preview_done.set = MagicMock()
    delegate._local_inference_lock = MagicMock()
    # Recovery/staging state
    delegate._pre_paste_clipboard = None
    delegate._verify_paste_text = None
    delegate._verify_paste_attempt = 0
    delegate._recovery_saved_clipboard = None
    delegate._recovery_text = None
    delegate._recovery_clipboard_state = "idle"
    delegate._recovery_pending_insert = None
    delegate._recovery_hold_active = False
    delegate._recovery_retry_pending = False
    # Staging state
    delegate._staging_text = None
    delegate._staging_active = False
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


def _simulate_recording_then_shift_release(delegate, main_module, wav_bytes=b"audio"):
    """Simulate a full recording cycle ending with shift+release."""
    delegate._capture.stop.return_value = wav_bytes
    delegate._record_start_time = time.monotonic() - 2.0  # 2 seconds of recording

    with patch.object(main_module.threading, "Thread") as MockThread:
        mock_thread = MagicMock()
        MockThread.return_value = mock_thread
        delegate._on_hold_end(shift_held=True)
        return MockThread, mock_thread


class TestStagingEntry:
    """Shift+release during recording enters staging instead of command pathway."""

    def test_shift_release_enters_staging(self, main_module, monkeypatch):
        """Shift+release during recording should set staging state, not send to command."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)

        MockThread, mock_thread = _simulate_recording_then_shift_release(d, main_module)

        # Should be in staging, not command pathway
        assert d._staging_active is True
        assert d._staging_text is not None or d._transcribing is True
        # Should NOT have called _send_text_as_command directly
        # The text should be staged, not dispatched to the assistant

    def test_shift_release_does_not_send_to_command(self, main_module, monkeypatch):
        """Staging must intercept the command pathway — text should not be sent yet."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._send_text_as_command = MagicMock()

        _simulate_recording_then_shift_release(d, main_module)

        d._send_text_as_command.assert_not_called()

    def test_staging_shows_overlay(self, main_module, monkeypatch):
        """Entering staging should show the staging overlay with the transcribed text."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._staging_text = None

        # Simulate transcription completing while in staging
        d._staging_active = True
        d._enter_staging_mode("hello world")

        assert d._staging_text == "hello world"
        d._overlay.show_recovery.assert_called()

    def test_staging_without_command_client(self, main_module, monkeypatch):
        """Shift+release without a command client should still enter staging."""
        d = _make_delegate(main_module, monkeypatch, command_client=False)

        _simulate_recording_then_shift_release(d, main_module)

        assert d._staging_active is True


class TestStagingGestures:
    """Gestures available while staging is active."""

    def test_spacebar_hold_from_staging_starts_recording(self, main_module, monkeypatch):
        """Holding spacebar from staging should start a new recording."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._staging_active = True
        d._staging_text = "old text"
        d._recovery_text = "old text"  # recovery state aliased to staging

        # Simulate hold start during staging
        d._on_hold_start()

        # Should start recording (capture.start called)
        d._capture.start.assert_called_once()

    def test_shift_hold_from_staging_sends_to_assistant(self, main_module, monkeypatch):
        """Shift+spacebar from staging should send the staged text to the assistant."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._staging_active = True
        d._staging_text = "send this"
        d._recovery_text = "send this"

        d._on_hold_end(shift_held=True)

        # Should have sent text to command pathway
        # (via _send_text_as_command or equivalent)
        assert d._staging_active is False

    def test_clean_release_from_staging_inserts_text(self, main_module, monkeypatch):
        """Spacebar release without shift from staging should insert text at cursor."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._staging_active = True
        d._staging_text = "insert this"
        d._recovery_text = "insert this"
        d._recovery_hold_active = False

        with patch("spoke.__main__.has_focused_text_input", return_value=True), \
             patch("spoke.__main__.inject_text") as mock_inject:
            d._on_hold_end(shift_held=False)

        # Should have attempted to insert text
        # (either directly or via _recovery_retry_insert)
        assert d._staging_active is False


class TestStagingPersistence:
    """Staged text lifecycle — persists until consumed or replaced."""

    def test_staged_text_persists_after_dismiss(self, main_module, monkeypatch):
        """Dismissing staging should hide overlay but keep text available."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._staging_active = True
        d._staging_text = "remember me"

        d._dismiss_staging()

        assert d._staging_active is False
        # Text should still be available for re-entry
        assert d._staging_text == "remember me"

    def test_successful_dictation_clears_staged_text(self, main_module, monkeypatch):
        """Normal dictation that pastes successfully should clear staged text."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._staging_text = "old staged text"
        d._staging_active = False

        # Simulate a successful text pathway transcription
        with patch("spoke.__main__.inject_text"), \
             patch("spoke.__main__.has_focused_text_input", return_value=True), \
             patch("spoke.__main__.save_pasteboard", return_value=None):
            d._inject_result_text("new dictation", "Pasted!")

        assert d._staging_text is None

    def test_new_recording_replaces_staged_text(self, main_module, monkeypatch):
        """Starting a new recording from staging should replace the old staged text."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._staging_active = True
        d._staging_text = "old text"
        d._recovery_text = "old text"

        # Start a new recording
        d._on_hold_start()

        # After new recording transcription completes, the staged text should be updated
        # (the old text is gone because a new recording was started)
        d._capture.start.assert_called_once()


class TestStagingRecoveryUnification:
    """Recovery mode (paste failure) should enter staging automatically."""

    def test_paste_failure_enters_staging(self, main_module, monkeypatch):
        """When OCR verification fails, should enter staging (not separate recovery)."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)

        d._enter_staging_mode("failed paste text")

        assert d._staging_active is True
        assert d._staging_text == "failed paste text"

    def test_recovery_and_staging_share_overlay(self, main_module, monkeypatch):
        """Both recovery and staging should use the same overlay surface."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)

        d._enter_staging_mode("some text")

        # Should call the same overlay method (show_recovery for now, will be renamed)
        d._overlay.show_recovery.assert_called()


class TestShortShiftHoldRecall:
    """Short shift-hold (< 800ms) enters staging with last Q&A pair."""

    def test_short_shift_hold_with_history_enters_staging(self, main_module, monkeypatch):
        """Short shift-hold with command history should enter staging with last Q&A."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._command_client.history = [("what time is it", "It's 3:14 PM")]
        d._capture.stop.return_value = b"audio"
        d._record_start_time = time.monotonic() - 0.5  # < 800ms

        d._on_hold_end(shift_held=True)

        # Should enter staging with the recalled Q&A, not send to command
        assert d._staging_active is True

    def test_short_shift_hold_without_history_dismisses(self, main_module, monkeypatch):
        """Short shift-hold with no history should dismiss (no staging to show)."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._command_client.history = []
        d._capture.stop.return_value = b"audio"
        d._record_start_time = time.monotonic() - 0.5  # < 800ms

        d._on_hold_end(shift_held=True)

        # Should not be in staging with no content to show
        d._menubar.set_status_text.assert_called_with("Ready — hold spacebar")


class TestHoldThroughFastPath:
    """Hold-through: shift held ~400ms after staging entry auto-sends."""

    def test_holdthrough_sends_after_timer(self, main_module, monkeypatch):
        """Enter staging with shift held, fire commit timer → sends to assistant."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._staging_text = None

        # Enter staging mode
        d._enter_staging_mode("send this")
        assert d._staging_active is True

        # Simulate that shift was held at staging entry — timer should be started
        assert d._staging_commit_timer is not None

        # Fire the timer manually
        d.stagingCommitTimerFired_(None)

        # Should have sent to assistant and left staging
        assert d._staging_active is False

    def test_holdthrough_cancelled_on_shift_release(self, main_module, monkeypatch):
        """Enter staging with shift held, release shift before timer → stay in staging."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)

        d._enter_staging_mode("keep this")
        assert d._staging_commit_timer is not None

        # Simulate shift release
        d._on_staging_shift_released()

        # Timer should be cancelled, still in staging
        assert d._staging_commit_timer is None
        assert d._staging_active is True
        assert d._staging_text == "keep this"

    def test_holdthrough_preauthorization(self, main_module, monkeypatch):
        """Timer fires but transcription not done → sends when transcription completes."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._staging_text = None
        d._transcribing = True  # transcription still in progress

        d._enter_staging_mode(None)  # text not yet available
        d._staging_preauthorized = False  # reset for test

        # Fire timer while text is None
        d.stagingCommitTimerFired_(None)

        # Should set pre-authorization flag
        assert d._staging_preauthorized is True
        # Should still be in staging (waiting for text)
        assert d._staging_active is True

    def test_preauthorized_sends_when_text_arrives(self, main_module, monkeypatch):
        """Pre-authorized staging sends as soon as transcription text arrives."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._staging_active = True
        d._staging_preauthorized = True
        d._transcribing = True
        d._transcription_token = 1
        d._send_text_as_command = MagicMock()

        # Simulate transcription completing
        d.stagingTranscriptionComplete_({"token": 1, "text": "hello"})

        # Should auto-send because pre-authorized
        d._send_text_as_command.assert_called_once_with("hello")

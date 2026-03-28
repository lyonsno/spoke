"""Tests for the command dispatch client."""

import json
import io
from unittest.mock import patch, MagicMock

import pytest


class TestCommandClient:
    """Test CommandClient message assembly and ring buffer."""

    def _make_client(self, **kwargs):
        from spoke.command import CommandClient
        defaults = {
            "base_url": "http://localhost:9999",
            "model": "test-model",
            "api_key": "test-key",
            "max_history": 5,
        }
        defaults.update(kwargs)
        return CommandClient(**defaults)

    def test_build_messages_no_history(self):
        """First command: system prompt + user utterance only."""
        client = self._make_client()
        msgs = client._build_messages("hello world")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1] == {"role": "user", "content": "hello world"}

    def test_build_messages_with_history(self):
        """History pairs are injected between system and current utterance."""
        client = self._make_client()
        client._history = [("first question", "first answer")]
        msgs = client._build_messages("second question")
        assert len(msgs) == 4
        assert msgs[0]["role"] == "system"
        assert msgs[1] == {"role": "user", "content": "first question"}
        assert msgs[2] == {"role": "assistant", "content": "first answer"}
        assert msgs[3] == {"role": "user", "content": "second question"}

    def test_build_messages_preserves_order(self):
        """Oldest history comes first (cache-friendly prefix stability)."""
        client = self._make_client()
        client._history = [
            ("q1", "a1"),
            ("q2", "a2"),
            ("q3", "a3"),
        ]
        msgs = client._build_messages("q4")
        user_msgs = [m["content"] for m in msgs if m["role"] == "user"]
        assert user_msgs == ["q1", "q2", "q3", "q4"]

    def test_ring_buffer_bounded(self):
        """History should not exceed max_history entries."""
        client = self._make_client(max_history=3)
        for i in range(5):
            client._history.append((f"q{i}", f"a{i}"))
            if len(client._history) > client._max_history:
                client._history.pop(0)
        assert len(client._history) == 3
        assert client._history[0] == ("q2", "a2")

    def test_history_property_returns_copy(self):
        """history property should return a copy, not a reference."""
        client = self._make_client()
        client._history.append(("q", "a"))
        h = client.history
        h.clear()
        assert len(client._history) == 1

    def test_clear_history(self):
        """clear_history should empty the ring buffer."""
        client = self._make_client()
        client._history = [("q1", "a1"), ("q2", "a2")]
        client.clear_history()
        assert client._history == []

    def test_config_from_env(self, monkeypatch):
        """Client should read config from environment variables."""
        monkeypatch.setenv("SPOKE_COMMAND_URL", "http://env-host:7777")
        monkeypatch.setenv("SPOKE_COMMAND_MODEL", "env-model")
        monkeypatch.setenv("SPOKE_COMMAND_API_KEY", "env-key")
        monkeypatch.setenv("SPOKE_COMMAND_HISTORY", "20")
        from spoke.command import CommandClient
        client = CommandClient()
        assert client._base_url == "http://env-host:7777"
        assert client._model == "env-model"
        assert client._api_key == "env-key"
        assert client._max_history == 20

    def test_config_kwargs_override_env(self, monkeypatch):
        """Explicit kwargs should take precedence over env vars."""
        monkeypatch.setenv("SPOKE_COMMAND_URL", "http://env-host:7777")
        client = self._make_client(base_url="http://kwarg-host:8888")
        assert client._base_url == "http://kwarg-host:8888"


class TestStreamCommand:
    """Test streaming command dispatch with mocked HTTP."""

    def _make_sse_response(self, chunks):
        """Build a fake HTTP response that yields SSE lines."""
        lines = []
        for chunk in chunks:
            lines.append(f"data: {json.dumps(chunk)}\n\n".encode())
        lines.append(b"data: [DONE]\n\n")
        body = b"".join(lines)
        resp = MagicMock()
        resp.__enter__ = MagicMock(return_value=io.BytesIO(body))
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def _content_chunk(self, token):
        return {
            "choices": [{"index": 0, "delta": {"content": token}}]
        }

    def _reasoning_chunk(self, token):
        return {
            "choices": [{"index": 0, "delta": {"reasoning_content": token}}]
        }

    def _role_chunk(self):
        return {
            "choices": [{"index": 0, "delta": {"role": "assistant"}}]
        }

    def test_stream_yields_content_tokens(self):
        """Content tokens should be yielded to the caller."""
        from spoke.command import CommandClient
        client = CommandClient(
            base_url="http://localhost:9999",
            model="test",
            api_key="key",
        )
        chunks = [
            self._role_chunk(),
            self._content_chunk("Hello"),
            self._content_chunk(" world"),
        ]
        fake_resp = self._make_sse_response(chunks)
        with patch("urllib.request.urlopen", return_value=fake_resp):
            tokens = list(client.stream_command("test"))
        assert tokens == ["Hello", " world"]

    def test_stream_skips_reasoning_tokens(self):
        """reasoning_content tokens should not be yielded."""
        from spoke.command import CommandClient
        client = CommandClient(
            base_url="http://localhost:9999",
            model="test",
            api_key="key",
        )
        chunks = [
            self._role_chunk(),
            self._reasoning_chunk("Let me think..."),
            self._reasoning_chunk("2+2=4"),
            self._content_chunk("The answer is 4."),
        ]
        fake_resp = self._make_sse_response(chunks)
        with patch("urllib.request.urlopen", return_value=fake_resp):
            tokens = list(client.stream_command("what is 2+2"))
        assert tokens == ["The answer is 4."]

    def test_stream_adds_to_history(self):
        """Completed stream should add (utterance, response) to history."""
        from spoke.command import CommandClient
        client = CommandClient(
            base_url="http://localhost:9999",
            model="test",
            api_key="key",
        )
        chunks = [
            self._content_chunk("Hi"),
            self._content_chunk(" there"),
        ]
        fake_resp = self._make_sse_response(chunks)
        with patch("urllib.request.urlopen", return_value=fake_resp):
            list(client.stream_command("hello"))
        assert client._history == [("hello", "Hi there")]

    def test_stream_does_not_store_reasoning_in_history(self):
        """Only content should appear in history, not reasoning."""
        from spoke.command import CommandClient
        client = CommandClient(
            base_url="http://localhost:9999",
            model="test",
            api_key="key",
        )
        chunks = [
            self._reasoning_chunk("thinking hard..."),
            self._content_chunk("done"),
        ]
        fake_resp = self._make_sse_response(chunks)
        with patch("urllib.request.urlopen", return_value=fake_resp):
            list(client.stream_command("do it"))
        assert client._history == [("do it", "done")]

    def test_stream_history_eviction(self):
        """History should evict oldest entry when full."""
        from spoke.command import CommandClient
        client = CommandClient(
            base_url="http://localhost:9999",
            model="test",
            api_key="key",
            max_history=2,
        )
        for i in range(3):
            chunks = [self._content_chunk(f"answer{i}")]
            fake_resp = self._make_sse_response(chunks)
            with patch("urllib.request.urlopen", return_value=fake_resp):
                list(client.stream_command(f"q{i}"))
        assert len(client._history) == 2
        assert client._history[0] == ("q1", "answer1")
        assert client._history[1] == ("q2", "answer2")

    def test_stream_sends_auth_header(self):
        """Request should include Authorization header when API key is set."""
        from spoke.command import CommandClient
        client = CommandClient(
            base_url="http://localhost:9999",
            model="test",
            api_key="secret123",
        )
        chunks = [self._content_chunk("ok")]
        fake_resp = self._make_sse_response(chunks)
        with patch("urllib.request.urlopen", return_value=fake_resp) as mock_open:
            list(client.stream_command("test"))
            req = mock_open.call_args[0][0]
            assert req.get_header("Authorization") == "Bearer secret123"

    def test_stream_sends_correct_payload(self):
        """Request payload should include model, messages, and stream=true."""
        from spoke.command import CommandClient
        client = CommandClient(
            base_url="http://localhost:9999",
            model="my-model",
            api_key="key",
        )
        chunks = [self._content_chunk("ok")]
        fake_resp = self._make_sse_response(chunks)
        with patch("urllib.request.urlopen", return_value=fake_resp) as mock_open:
            list(client.stream_command("do something"))
            req = mock_open.call_args[0][0]
            body = json.loads(req.data)
            assert body["model"] == "my-model"
            assert body["stream"] is True
            assert body["messages"][-1] == {"role": "user", "content": "do something"}

    def test_stream_includes_history_in_payload(self):
        """When history exists, it should appear in the messages array."""
        from spoke.command import CommandClient
        client = CommandClient(
            base_url="http://localhost:9999",
            model="test",
            api_key="key",
        )
        client._history = [("prev question", "prev answer")]
        chunks = [self._content_chunk("ok")]
        fake_resp = self._make_sse_response(chunks)
        with patch("urllib.request.urlopen", return_value=fake_resp) as mock_open:
            list(client.stream_command("new question"))
            req = mock_open.call_args[0][0]
            body = json.loads(req.data)
            msgs = body["messages"]
            assert msgs[1] == {"role": "user", "content": "prev question"}
            assert msgs[2] == {"role": "assistant", "content": "prev answer"}
            assert msgs[3] == {"role": "user", "content": "new question"}


class TestStreamErrorHandling:
    """Test error paths in stream_command."""

    def _make_sse_response(self, chunks):
        lines = []
        for chunk in chunks:
            lines.append(f"data: {json.dumps(chunk)}\n\n".encode())
        lines.append(b"data: [DONE]\n\n")
        body = b"".join(lines)
        resp = MagicMock()
        resp.__enter__ = MagicMock(return_value=io.BytesIO(body))
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def test_stream_connection_error_raises(self):
        """URLError on connection should propagate to caller."""
        import urllib.error
        from spoke.command import CommandClient
        client = CommandClient(
            base_url="http://localhost:9999",
            model="test",
            api_key="key",
        )
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            with pytest.raises(urllib.error.URLError):
                list(client.stream_command("test"))
        # History should not be modified on failure
        assert client._history == []

    def test_stream_malformed_json_skipped(self):
        """Malformed SSE chunks should be skipped, not crash the stream."""
        from spoke.command import CommandClient
        client = CommandClient(
            base_url="http://localhost:9999",
            model="test",
            api_key="key",
        )
        body = b"data: {bad json}\n\ndata: " + json.dumps(
            {"choices": [{"index": 0, "delta": {"content": "ok"}}]}
        ).encode() + b"\n\ndata: [DONE]\n\n"
        resp = MagicMock()
        resp.__enter__ = MagicMock(return_value=io.BytesIO(body))
        resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=resp):
            tokens = list(client.stream_command("test"))
        assert tokens == ["ok"]

    def test_stream_empty_response_stored_in_history(self):
        """If OMLX returns no content tokens, empty string is stored."""
        from spoke.command import CommandClient
        client = CommandClient(
            base_url="http://localhost:9999",
            model="test",
            api_key="key",
        )
        # Only role chunk + done, no content
        chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant"}}]}]
        fake_resp = self._make_sse_response(chunks)
        with patch("urllib.request.urlopen", return_value=fake_resp):
            tokens = list(client.stream_command("hello"))
        assert tokens == []
        assert client._history == [("hello", "")]

    def test_stream_only_reasoning_yields_nothing(self):
        """If OMLX returns only reasoning tokens and no content, yield nothing."""
        from spoke.command import CommandClient
        client = CommandClient(
            base_url="http://localhost:9999",
            model="test",
            api_key="key",
        )
        chunks = [
            {"choices": [{"index": 0, "delta": {"reasoning_content": "thinking..."}}]},
            {"choices": [{"index": 0, "delta": {"reasoning_content": "still thinking..."}}]},
        ]
        fake_resp = self._make_sse_response(chunks)
        with patch("urllib.request.urlopen", return_value=fake_resp):
            tokens = list(client.stream_command("think hard"))
        assert tokens == []
        assert client._history == [("think hard", "")]


class TestShiftReleaseRouting:
    """Test that shift-release is detected and routes to command path."""

    def _make_detector(self, input_tap_module, hold_ms=400):
        mod = input_tap_module
        on_start = MagicMock()
        on_end = MagicMock()
        det = mod.SpacebarHoldDetector.__new__(mod.SpacebarHoldDetector)
        det._on_hold_start = on_start
        det._on_hold_end = on_end
        det._hold_s = hold_ms / 1000.0
        det._state = mod._State.IDLE
        det._hold_timer = None
        det._safety_timer = None
        det._forwarding = False
        det._forwarding_timer = None
        det._tap = None
        det._tap_source = None
        return det, on_start, on_end

    def test_shift_held_at_release_passes_flag(self, input_tap_module):
        """When shift is held during recording release, on_hold_end should
        receive shift_held=True (once the routing is wired)."""
        mod = input_tap_module
        det, on_start, on_end = self._make_detector(input_tap_module)

        # Enter recording state
        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)
        assert det._state == mod._State.RECORDING

        # Release with shift held
        shift_flag = mod.kCGEventFlagMaskShift
        det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=shift_flag)
        assert det._state == mod._State.IDLE
        on_end.assert_called_once_with(shift_held=True)

    def test_normal_release_passes_no_shift(self, input_tap_module):
        """Normal release (no shift) should pass shift_held=False."""
        mod = input_tap_module
        det, on_start, on_end = self._make_detector(input_tap_module)

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)
        assert det._state == mod._State.RECORDING

        det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0)
        assert det._state == mod._State.IDLE
        on_end.assert_called_once_with(shift_held=False)

    def test_shift_during_keydown_starts_recording(self, input_tap_module):
        """Shift+Space on keyDown should start recording normally.
        Shift is only detected at release for command routing."""
        mod = input_tap_module
        det, _, _ = self._make_detector(input_tap_module)

        shift_flag = mod.kCGEventFlagMaskShift
        result = det.handle_key_down(mod.SPACEBAR_KEYCODE, shift_flag)
        assert result is True  # suppressed — recording starts
        assert det._state == mod._State.WAITING


class TestCommandThinking:
    """Test SPOKE_COMMAND_THINKING toggle."""

    def _make_client(self, **kwargs):
        from spoke.command import CommandClient
        defaults = {
            "base_url": "http://localhost:9999",
            "model": "test-model",
            "api_key": "test-key",
            "max_history": 5,
        }
        defaults.update(kwargs)
        return CommandClient(**defaults)

    def test_thinking_enabled_by_default(self):
        """Default: thinking is enabled, no chat_template_kwargs in body."""
        client = self._make_client()
        assert client._enable_thinking is True

    def test_thinking_disabled_with_env(self, monkeypatch):
        """SPOKE_COMMAND_THINKING=0 should disable thinking."""
        monkeypatch.setenv("SPOKE_COMMAND_THINKING", "0")
        client = self._make_client()
        assert client._enable_thinking is False

    def test_thinking_disabled_adds_template_kwargs(self, monkeypatch):
        """When thinking is disabled, stream_command body should include
        chat_template_kwargs with enable_thinking: False."""
        monkeypatch.setenv("SPOKE_COMMAND_THINKING", "0")
        client = self._make_client()

        # Capture the request body by mocking urlopen
        import urllib.request
        captured_body = {}

        def fake_urlopen(req, timeout=None):
            captured_body["data"] = json.loads(req.data)
            # Return a fake empty SSE response
            return MagicMock(
                __enter__=lambda s: MagicMock(
                    __iter__=lambda s: iter([]),
                    read=lambda: b"",
                ),
                __exit__=lambda s, *a: None,
            )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            # Consume the generator
            try:
                for _ in client.stream_command("test"):
                    pass
            except Exception:
                pass

        assert "chat_template_kwargs" in captured_body["data"]
        assert captured_body["data"]["chat_template_kwargs"] == {"enable_thinking": False}

    def test_thinking_enabled_omits_template_kwargs(self):
        """When thinking is enabled (default), body should NOT include
        chat_template_kwargs."""
        client = self._make_client()

        import urllib.request
        captured_body = {}

        def fake_urlopen(req, timeout=None):
            captured_body["data"] = json.loads(req.data)
            return MagicMock(
                __enter__=lambda s: MagicMock(
                    __iter__=lambda s: iter([]),
                    read=lambda: b"",
                ),
                __exit__=lambda s, *a: None,
            )

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            try:
                for _ in client.stream_command("test"):
                    pass
            except Exception:
                pass

        assert "chat_template_kwargs" not in captured_body["data"]

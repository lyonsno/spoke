"""Tests for the command dispatch client."""

import json
import io
from unittest.mock import patch, MagicMock

import pytest


def _make_sse_response(chunks):
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
        assert "add_to_tray" in msgs[0]["content"]
        assert "query_gmail" in msgs[0]["content"]
        assert msgs[1] == {"role": "user", "content": "hello world"}

    def test_system_prompt_explicitly_allows_literal_read_aloud(self):
        """The prompt should tell the assistant to use literal refs for arbitrary speech."""
        from spoke.command import _SYSTEM_PROMPT
        assert "literal:<exact text to speak>" in _SYSTEM_PROMPT
        assert "Do not pretend read_aloud is limited to visible text." in _SYSTEM_PROMPT

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

    def test_system_prompt_does_not_carry_voice_or_terse_question_avoidance_framing(self):
        """The command prompt should not force voice-assistant, terse, or no-questions framing."""
        from spoke.command import _SYSTEM_PROMPT

        assert "voice assistant" not in _SYSTEM_PROMPT.lower()
        assert "be concise" not in _SYSTEM_PROMPT.lower()
        assert "without questioning" not in _SYSTEM_PROMPT.lower()
        assert "spoke this aloud" not in _SYSTEM_PROMPT.lower()

    def test_ring_buffer_bounded_via_stream(self):
        """stream_command should evict the oldest entry when history exceeds max_history."""
        client = self._make_client(max_history=3)
        for i in range(5):
            chunks = [{"choices": [{"index": 0, "delta": {"content": f"a{i}"}}]}]
            fake_resp = _make_sse_response(chunks)
            with patch("urllib.request.urlopen", return_value=fake_resp):
                list(client.stream_command(f"q{i}"))
        assert len(client._history) == 3
        assert client._history[0] == ("q2", "a2")
        assert client._history[2] == ("q4", "a4")

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
        """Client should read model/key config from environment variables."""
        monkeypatch.setenv("SPOKE_COMMAND_MODEL", "env-model")
        monkeypatch.setenv("SPOKE_COMMAND_API_KEY", "env-key")
        monkeypatch.setenv("SPOKE_COMMAND_HISTORY", "20")
        from spoke.command import CommandClient
        client = CommandClient()
        assert client._base_url == "http://localhost:8001"
        assert client._model == "env-model"
        assert client._api_key == "env-key"
        assert client._max_history == 20

    def test_config_kwargs_override_default(self, monkeypatch):
        """Explicit kwargs should take precedence over the built-in default."""
        client = self._make_client(base_url="http://kwarg-host:8888")
        assert client._base_url == "http://kwarg-host:8888"

    def test_list_models_fetches_openai_models_endpoint(self):
        """list_models() should return ids from /v1/models in server order."""
        from spoke.command import CommandClient

        payload = {
            "data": [
                {"id": "qwen3p5-35B-A3B"},
                {"id": "qwen3-14b"},
                {"id": "qwen3-4b"},
            ]
        }
        fake_resp = MagicMock()
        fake_resp.__enter__ = MagicMock(return_value=io.BytesIO(json.dumps(payload).encode()))
        fake_resp.__exit__ = MagicMock(return_value=False)

        client = CommandClient(
            base_url="http://localhost:9999",
            model="test",
            api_key="key",
        )

        with patch("urllib.request.urlopen", return_value=fake_resp) as mock_open:
            assert client.list_models() == ["qwen3p5-35B-A3B", "qwen3-14b", "qwen3-4b"]

        req = mock_open.call_args[0][0]
        assert req.full_url == "http://localhost:9999/v1/models"
        assert req.get_method() == "GET"

    def test_list_models_skips_v1_prefix_for_cloud_urls(self):
        """Cloud OpenAI-compat endpoints already include a version prefix."""
        from spoke.command import CommandClient

        payload = {"data": [{"id": "gemini-2.5-flash"}]}
        fake_resp = MagicMock()
        fake_resp.__enter__ = MagicMock(return_value=io.BytesIO(json.dumps(payload).encode()))
        fake_resp.__exit__ = MagicMock(return_value=False)

        client = CommandClient(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            model="gemini-2.5-flash",
            api_key="key",
        )

        with patch("urllib.request.urlopen", return_value=fake_resp) as mock_open:
            assert client.list_models() == ["gemini-2.5-flash"]

        req = mock_open.call_args[0][0]
        assert req.full_url == "https://generativelanguage.googleapis.com/v1beta/openai/models"

    def test_list_models_strips_models_prefix(self):
        """Gemini returns 'models/gemini-2.5-flash'; list_models strips the prefix."""
        from spoke.command import CommandClient

        payload = {"data": [
            {"id": "models/gemini-2.5-flash"},
            {"id": "models/gemini-2.5-pro"},
            {"id": "qwen3-14b"},
        ]}
        fake_resp = MagicMock()
        fake_resp.__enter__ = MagicMock(return_value=io.BytesIO(json.dumps(payload).encode()))
        fake_resp.__exit__ = MagicMock(return_value=False)

        client = CommandClient(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            model="gemini-2.5-flash",
            api_key="key",
        )

        with patch("urllib.request.urlopen", return_value=fake_resp):
            assert client.list_models() == ["gemini-2.5-flash", "gemini-2.5-pro", "qwen3-14b"]

    def test_list_models_sends_auth_header(self):
        """list_models() should reuse the configured bearer token."""
        from spoke.command import CommandClient

        payload = {"data": [{"id": "qwen3p5-35B-A3B"}]}
        fake_resp = MagicMock()
        fake_resp.__enter__ = MagicMock(return_value=io.BytesIO(json.dumps(payload).encode()))
        fake_resp.__exit__ = MagicMock(return_value=False)

        client = CommandClient(
            base_url="http://localhost:9999",
            model="test",
            api_key="secret123",
        )

        with patch("urllib.request.urlopen", return_value=fake_resp) as mock_open:
            client.list_models()

        req = mock_open.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer secret123"


class TestStreamCommand:
    """Test streaming command dispatch with mocked HTTP."""

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
        fake_resp = _make_sse_response(chunks)
        with patch("urllib.request.urlopen", return_value=fake_resp):
            tokens = list(client.stream_command("test"))
        assert tokens == ["Hello", " world"]

    def test_stream_events_keep_tool_round_tokens_provisional(self):
        """Tool-round deltas may stream, but only the final assistant reply is canonized."""
        from spoke.command import CommandClient
        from spoke.tool_dispatch import get_tool_schemas

        client = CommandClient(
            base_url="http://localhost:9999",
            model="test",
            api_key="key",
        )

        tool_round_chunks = [
            self._content_chunk("Let me check. "),
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "call_1", "type": "function",
                 "function": {"name": "capture_context", "arguments": ""}}
            ]}}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '{"scope":"active_window"}'}}
            ]}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ]
        final_chunks = [self._content_chunk("Done.")]
        first_resp = _make_sse_response(tool_round_chunks)
        second_resp = _make_sse_response(final_chunks)

        call_count = {"n": 0}

        def fake_urlopen(req, timeout=None):
            call_count["n"] += 1
            return first_resp if call_count["n"] == 1 else second_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            events = list(
                client.stream_command_events(
                    "check it",
                    tools=get_tool_schemas(),
                    tool_executor=lambda **kwargs: '{"ok": true}',
                )
            )

        assert call_count["n"] == 2
        assert [
            (event.kind, event.text, event.tool_name)
            for event in events
        ] == [
            ("assistant_delta", "Let me check. ", None),
            ("assistant_delta", "\n[calling capture_context…]\n", None),
            ("tool_call", "", "capture_context"),
            ("assistant_delta", "Done.", None),
            ("assistant_final", "Done.", None),
        ]
        assert client._history == [("check it", "Done.")]

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
        fake_resp = _make_sse_response(chunks)
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
        fake_resp = _make_sse_response(chunks)
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
        fake_resp = _make_sse_response(chunks)
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
            fake_resp = _make_sse_response(chunks)
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
        fake_resp = _make_sse_response(chunks)
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
        fake_resp = _make_sse_response(chunks)
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
        fake_resp = _make_sse_response(chunks)
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
        fake_resp = _make_sse_response(chunks)
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
        fake_resp = _make_sse_response(chunks)
        with patch("urllib.request.urlopen", return_value=fake_resp):
            tokens = list(client.stream_command("think hard"))
        assert tokens == []
        assert client._history == [("think hard", "")]


class TestStreamCleanup:
    """Test HTTP connection cleanup and history invariants on partial consumption."""

    def test_partial_consumption_does_not_append_history(self):
        """If the caller breaks mid-stream, history should remain unchanged."""
        from spoke.command import CommandClient
        client = CommandClient(
            base_url="http://localhost:9999",
            model="test",
            api_key="key",
        )
        chunks = [
            {"choices": [{"index": 0, "delta": {"content": "tok1"}}]},
            {"choices": [{"index": 0, "delta": {"content": "tok2"}}]},
            {"choices": [{"index": 0, "delta": {"content": "tok3"}}]},
        ]
        fake_resp = _make_sse_response(chunks)
        with patch("urllib.request.urlopen", return_value=fake_resp):
            gen = client.stream_command("test")
            first = next(gen)
            assert first == "tok1"
            # Abandon the generator mid-stream
            gen.close()
        assert client._history == []

    def test_partial_consumption_closes_http_response(self):
        """Abandoning a stream mid-flight should exit the HTTP context manager."""
        from spoke.command import CommandClient
        client = CommandClient(
            base_url="http://localhost:9999",
            model="test",
            api_key="key",
        )
        chunks = [
            {"choices": [{"index": 0, "delta": {"content": "tok1"}}]},
            {"choices": [{"index": 0, "delta": {"content": "tok2"}}]},
        ]
        fake_resp = _make_sse_response(chunks)
        with patch("urllib.request.urlopen", return_value=fake_resp):
            gen = client.stream_command("test")
            next(gen)
            gen.close()
        # The mock context manager's __exit__ should have been called
        fake_resp.__exit__.assert_called_once()

    def test_second_stream_after_partial_has_clean_state(self):
        """A new stream after a partial consumption should start with empty response."""
        from spoke.command import CommandClient
        client = CommandClient(
            base_url="http://localhost:9999",
            model="test",
            api_key="key",
        )
        # First stream: partial consumption
        chunks1 = [
            {"choices": [{"index": 0, "delta": {"content": "partial"}}]},
            {"choices": [{"index": 0, "delta": {"content": "-data"}}]},
        ]
        fake_resp1 = _make_sse_response(chunks1)
        with patch("urllib.request.urlopen", return_value=fake_resp1):
            gen = client.stream_command("first")
            next(gen)
            gen.close()

        # Second stream: full consumption
        chunks2 = [
            {"choices": [{"index": 0, "delta": {"content": "clean"}}]},
        ]
        fake_resp2 = _make_sse_response(chunks2)
        with patch("urllib.request.urlopen", return_value=fake_resp2):
            tokens = list(client.stream_command("second"))
        assert tokens == ["clean"]
        # Only the completed stream should be in history
        assert client._history == [("second", "clean")]


class TestToolCallRendering:
    """Test that tool-call deltas are surfaced as visible inline text."""

    def _content_chunk(self, token):
        return {
            "choices": [{"index": 0, "delta": {"content": token}}]
        }

    def _tool_call_chunk(self, name):
        return {
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"name": name},
                    }]
                },
            }]
        }

    def _tool_call_args_chunk(self, args_fragment):
        """Tool-call delta with only arguments (no name) — typical for streamed args."""
        return {
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": args_fragment},
                    }]
                },
            }]
        }

    def test_tool_call_yields_calling_indicator(self):
        """A tool-call delta with a function name should yield '[calling name…]'."""
        from spoke.command import CommandClient
        client = CommandClient(
            base_url="http://localhost:9999",
            model="test",
            api_key="key",
        )
        chunks = [
            self._content_chunk("Let me check. "),
            self._tool_call_chunk("capture_context"),
            self._content_chunk("Done."),
        ]
        fake_resp = _make_sse_response(chunks)
        with patch("urllib.request.urlopen", return_value=fake_resp):
            tokens = list(client.stream_command("read the screen"))
        assert tokens == [
            "Let me check. ",
            "\n[calling capture_context…]\n",
            "Done.",
        ]

    def test_tool_call_included_in_full_response(self):
        """The full_response stored in history should include tool-call text."""
        from spoke.command import CommandClient
        client = CommandClient(
            base_url="http://localhost:9999",
            model="test",
            api_key="key",
        )
        chunks = [
            self._content_chunk("A"),
            self._tool_call_chunk("read_aloud"),
            self._content_chunk("B"),
        ]
        fake_resp = _make_sse_response(chunks)
        with patch("urllib.request.urlopen", return_value=fake_resp):
            list(client.stream_command("test"))
        assert len(client._history) == 1
        _, response = client._history[0]
        assert "A" in response
        assert "[calling read_aloud…]" in response
        assert "B" in response

    def test_tool_call_without_name_is_silent(self):
        """Tool-call deltas that carry only arguments (no name) should not yield text."""
        from spoke.command import CommandClient
        client = CommandClient(
            base_url="http://localhost:9999",
            model="test",
            api_key="key",
        )
        chunks = [
            self._content_chunk("thinking"),
            self._tool_call_args_chunk('{"text": "hello"}'),
        ]
        fake_resp = _make_sse_response(chunks)
        with patch("urllib.request.urlopen", return_value=fake_resp):
            tokens = list(client.stream_command("test"))
        assert tokens == ["thinking"]

    def test_multiple_tool_calls_in_single_delta(self):
        """Multiple tool calls in one delta should each yield an indicator."""
        from spoke.command import CommandClient
        client = CommandClient(
            base_url="http://localhost:9999",
            model="test",
            api_key="key",
        )
        chunks = [
            {
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"name": "capture_context"}},
                            {"index": 1, "function": {"name": "read_aloud"}},
                        ]
                    },
                }]
            },
            self._content_chunk("All done."),
        ]
        fake_resp = _make_sse_response(chunks)
        with patch("urllib.request.urlopen", return_value=fake_resp):
            tokens = list(client.stream_command("do two things"))
        assert "\n[calling capture_context…]\n" in tokens
        assert "\n[calling read_aloud…]\n" in tokens
        assert "All done." in tokens


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
        """Shift-held release should carry the flag through the decision timer."""
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
        assert det._pending_release_active is True

        det.releaseDecisionTimerFired_(None)

        on_end.assert_called_once_with(shift_held=True, enter_held=False)

    def test_normal_release_passes_no_shift(self, input_tap_module):
        """Normal release (no shift) should pass shift_held=False."""
        mod = input_tap_module
        det, on_start, on_end = self._make_detector(input_tap_module)

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)
        assert det._state == mod._State.RECORDING

        det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0)
        assert det._state == mod._State.IDLE
        on_end.assert_called_once_with(shift_held=False, enter_held=False)

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
            for _ in client.stream_command("test"):
                pass

        assert "data" in captured_body, "urlopen was never called — request body not captured"
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
            for _ in client.stream_command("test"):
                pass

        assert "data" in captured_body, "urlopen was never called — request body not captured"
        assert "chat_template_kwargs" not in captured_body["data"]


class TestXMLToolCallFallback:
    """Test spoke-side XML tool call extraction from content stream."""

    def test_bare_function_tags_extracted(self):
        from spoke.command import _extract_xml_tool_calls
        text = '<function=list_directory><parameter=dir_path>/Users/dev</parameter></function>'
        result = _extract_xml_tool_calls(text)
        assert result is not None
        cleaned, calls = result
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "list_directory"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["dir_path"] == "/Users/dev"
        assert cleaned == ""

    def test_wrapped_in_tool_call_tags(self):
        from spoke.command import _extract_xml_tool_calls
        text = '<tool_call><function=read_file><parameter=path>foo.txt</parameter></function></tool_call>'
        result = _extract_xml_tool_calls(text)
        assert result is not None
        _, calls = result
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "read_file"

    def test_trailing_tool_call_close_only(self):
        from spoke.command import _extract_xml_tool_calls
        text = '<function=search><parameter=query>hello</parameter></function></tool_call>'
        result = _extract_xml_tool_calls(text)
        assert result is not None
        _, calls = result
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "search"

    def test_mixed_text_and_xml(self):
        from spoke.command import _extract_xml_tool_calls
        text = 'Let me check:\n<function=ls><parameter=dir>/tmp</parameter></function>\nDone.'
        result = _extract_xml_tool_calls(text)
        assert result is not None
        cleaned, calls = result
        assert len(calls) == 1
        assert "Let me check:" in cleaned
        assert "Done." in cleaned

    def test_no_xml_returns_none(self):
        from spoke.command import _extract_xml_tool_calls
        assert _extract_xml_tool_calls("just plain text") is None

    def test_multiple_calls_extracted(self):
        from spoke.command import _extract_xml_tool_calls
        text = ('<function=a><parameter=x>1</parameter></function>'
                '<function=b><parameter=y>2</parameter></function>')
        result = _extract_xml_tool_calls(text)
        assert result is not None
        _, calls = result
        assert len(calls) == 2
        assert calls[0]["function"]["name"] == "a"
        assert calls[1]["function"]["name"] == "b"

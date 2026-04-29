"""Tests for local-auth operator agent backend sessions."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _DeferredThread:
    """Test thread that starts only when run_now() is called."""

    created: list["_DeferredThread"] = []

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self.daemon = daemon
        self.started = False
        type(self).created.append(self)

    def start(self):
        self.started = True

    def run_now(self):
        self._target(*self._args, **self._kwargs)


class _ImmediateThread:
    """Test thread that executes target synchronously on start()."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self.daemon = daemon

    def start(self):
        self._target(*self._args, **self._kwargs)


class _FakeAgentBackendManager:
    def __init__(self, *, final_state: str = "completed", result: str = "agent result"):
        self.final_state = final_state
        self.result = result
        self.launched: list[dict] = []
        self.cancelled: list[str] = []

    def launch(self, **kwargs):
        self.launched.append(kwargs)
        provider = kwargs["provider"]
        return {
            "id": f"agent-backend-{provider}-1",
            "provider": provider,
            "state": "queued",
            "provider_session_id": None,
            "result": None,
            "error": None,
        }

    def get_session(self, session_id):
        provider = self.launched[-1]["provider"]
        error = "backend failed" if self.final_state == "failed" else None
        return {
            "id": session_id,
            "provider": provider,
            "state": self.final_state,
            "provider_session_id": f"{provider}-provider-session-1",
            "result": self.result,
            "error": error,
        }

    def cancel(self, session_id):
        self.cancelled.append(session_id)
        return {"id": session_id, "state": "cancelled"}


class _StreamingFakeAgentBackendManager:
    def __init__(self):
        self.launched: list[dict] = []
        self.poll_count = 0

    def launch(self, **kwargs):
        self.launched.append(kwargs)
        return {
            "id": "agent-backend-codex-1",
            "provider": "codex",
            "state": "running",
            "provider_session_id": None,
            "backend_events": [],
            "result": None,
            "error": None,
        }

    def get_session(self, session_id):
        self.poll_count += 1
        if self.poll_count == 1:
            return {
                "id": session_id,
                "provider": "codex",
                "state": "running",
                "provider_session_id": "codex-thread-1",
                "backend_events": [
                    {
                        "sequence": 1,
                        "kind": "command_execution",
                        "text": "pytest tests/test_agent_backends.py\nSECRET OUTPUT",
                        "data": {
                            "id": "cmd-1",
                            "type": "command_execution",
                            "command": "pytest tests/test_agent_backends.py",
                            "aggregated_output": "SECRET OUTPUT",
                            "status": "in_progress",
                        },
                    },
                    {
                        "sequence": 2,
                        "kind": "reasoning",
                        "text": "checking the focused test",
                        "data": {
                            "id": "reason-1",
                            "type": "reasoning",
                            "text": "checking the focused test",
                        },
                    },
                ],
                "result": None,
                "error": None,
            }
        return {
            "id": session_id,
            "provider": "codex",
            "state": "completed",
            "provider_session_id": "codex-thread-1",
            "backend_events": [],
            "result": "done",
            "error": None,
        }

    def cancel(self, _session_id):
        return {"state": "cancelled"}


class _QuietRunningAgentBackendManager:
    def __init__(self):
        self.launched: list[dict] = []
        self.poll_count = 0

    def launch(self, **kwargs):
        self.launched.append(kwargs)
        return {
            "id": "agent-backend-codex-quiet",
            "provider": "codex",
            "state": "running",
            "provider_session_id": None,
            "backend_events": [],
            "result": None,
            "error": None,
        }

    def get_session(self, session_id):
        self.poll_count += 1
        state = "running" if self.poll_count == 1 else "completed"
        return {
            "id": session_id,
            "provider": "codex",
            "state": state,
            "provider_session_id": "codex-thread-quiet",
            "backend_events": [],
            "result": "quiet done",
            "error": None,
        }

    def cancel(self, _session_id):
        return {"state": "cancelled"}


def _make_agent_shell_delegate(main_module):
    delegate = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
    delegate._transcription_token = 0
    delegate._transcribing = False
    delegate._command_tool_used_tts = False
    delegate._transcribe_start = 0.0
    delegate._menubar = MagicMock()
    delegate._detector = MagicMock()
    delegate._detector.approval_active = False
    delegate._detector.command_overlay_active = False
    delegate._command_overlay = None
    delegate._command_client = MagicMock()
    delegate._command_client.history = []
    delegate._command_client.stream_command_events.return_value = [
        SimpleNamespace(kind="assistant_final", text="assistant response")
    ]
    delegate._scene_cache = None
    delegate._tool_schemas = []
    delegate._tts_client = None
    delegate._turn_carver = None
    delegate._glow = None
    delegate._overlay = None
    delegate._narrator = None
    delegate._agent_shell_sessions = {}
    delegate._agent_shell_provider = "off"
    delegate._save_preference = MagicMock()
    delegate.performSelectorOnMainThread_withObject_waitUntilDone_ = MagicMock()
    return delegate


class TestAgentBackendManager:
    def test_codex_resolution_checks_homebrew_path_when_gui_path_is_thin(
        self, monkeypatch
    ):
        from spoke.agent_backends import _resolve_codex_path

        monkeypatch.setattr(
            "spoke.agent_backends.shutil.which",
            lambda _name, *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "spoke.agent_backends.os.path.isfile",
            lambda path: path == "/opt/homebrew/bin/codex",
        )
        monkeypatch.setattr("spoke.agent_backends.os.access", lambda _path, _mode: True)

        assert _resolve_codex_path() == "/opt/homebrew/bin/codex"

    def test_codex_backend_env_strips_billing_credentials(self, monkeypatch):
        from spoke.agent_backends import _subscription_only_env

        monkeypatch.setenv("OPENAI_API_KEY", "forbidden")
        monkeypatch.setenv("CODEX_API_KEY", "forbidden")
        monkeypatch.setenv("CODEX_HOME", "/tmp/codex-home")

        env = _subscription_only_env()

        assert "OPENAI_API_KEY" not in env
        assert "CODEX_API_KEY" not in env
        assert env["CODEX_HOME"] == "/tmp/codex-home"

    def test_codex_backend_rejects_non_chatgpt_login_status(self, monkeypatch):
        from spoke.agent_backends import AgentBackendUnavailable, _require_codex_subscription_login

        def fake_status(_codex_path, _env):
            return "Logged in using billing credentials"

        monkeypatch.setattr("spoke.agent_backends._codex_login_status", fake_status)

        with pytest.raises(AgentBackendUnavailable, match="ChatGPT subscription"):
            _require_codex_subscription_login("/usr/local/bin/codex", {})

    def test_codex_json_item_events_preserve_tool_loop_shape(self):
        from spoke.agent_backends import _event_from_codex_item

        event = _event_from_codex_item(
            {
                "id": "cmd-1",
                "type": "command_execution",
                "command": "pytest tests/test_agent_backends.py",
                "aggregated_output": "1 passed",
                "status": "completed",
                "exit_code": 0,
            }
        )

        assert event is not None
        assert event.kind == "command_execution"
        assert "pytest tests/test_agent_backends.py" in event.text
        assert "1 passed" in event.text
        assert event.data["status"] == "completed"

    def test_manager_publishes_backend_events_while_session_is_running(self):
        from spoke.agent_backends import (
            AgentBackendEvent,
            AgentBackendManager,
            AgentBackendRunResult,
        )

        event_added = threading.Event()
        release = threading.Event()

        def fake_runner(provider, prompt, cwd, resume_id, cancel_check, event_sink):
            event_sink(
                AgentBackendEvent(
                    kind="reasoning",
                    text="checking the focused test",
                    data={"id": "reason-1", "type": "reasoning"},
                )
            )
            event_added.set()
            assert release.wait(timeout=2)
            return AgentBackendRunResult(
                provider=provider,
                session_id="codex-thread-123",
                final_response="Plan complete.",
            )

        manager = AgentBackendManager(backend_runner=fake_runner)

        launched = manager.launch(
            provider="codex",
            prompt="inspect the failing tests",
            cwd="/tmp/project",
        )
        assert event_added.wait(timeout=2)
        running = manager.get_session(launched["id"])

        assert running["state"] == "running"
        assert running["backend_events"] == [
            {
                "sequence": 1,
                "kind": "reasoning",
                "text": "checking the focused test",
                "data": {"id": "reason-1", "type": "reasoning"},
            }
        ]

        release.set()
        deadline = time.time() + 2
        while time.time() < deadline:
            completed = manager.get_session(launched["id"])
            if completed["state"] == "completed":
                break
            time.sleep(0.01)
        assert completed["state"] == "completed"

    def test_launch_tracks_provider_cwd_resume_and_result_identity(self, tmp_path):
        from spoke.agent_backends import AgentBackendManager, AgentBackendRunResult

        calls = []
        _DeferredThread.created = []

        def fake_runner(provider, prompt, cwd, resume_id, cancel_check, event_sink):
            calls.append((provider, prompt, cwd, resume_id, cancel_check()))
            return AgentBackendRunResult(
                provider=provider,
                session_id="codex-thread-123",
                final_response="Plan complete.",
            )

        manager = AgentBackendManager(
            backend_runner=fake_runner,
            thread_factory=_DeferredThread,
        )

        launched = manager.launch(
            provider="codex",
            prompt="inspect the failing tests",
            cwd=str(tmp_path),
            resume_id="prior-session",
        )
        assert launched["id"] == "agent-backend-codex-1"
        assert launched["provider"] == "codex"
        assert launched["state"] == "queued"
        assert launched["cwd"] == str(tmp_path)
        assert launched["resume_id"] == "prior-session"

        _DeferredThread.created[-1].run_now()

        result = manager.get_session(launched["id"])
        assert calls == [
            ("codex", "inspect the failing tests", str(tmp_path), "prior-session", False)
        ]
        assert result["state"] == "completed"
        assert result["provider_session_id"] == "codex-thread-123"
        assert result["result"] == "Plan complete."
        assert result["result_preview"] == "Plan complete."

    def test_backend_unavailable_is_visible_without_looking_like_terminal_failure(self):
        from spoke.agent_backends import (
            AgentBackendManager,
            AgentBackendUnavailable,
        )

        _DeferredThread.created = []

        def fake_runner(provider, prompt, cwd, resume_id, cancel_check, event_sink):
            raise AgentBackendUnavailable("Codex CLI is not logged in with ChatGPT")

        manager = AgentBackendManager(
            backend_runner=fake_runner,
            thread_factory=_DeferredThread,
        )

        launched = manager.launch(
            provider="codex",
            prompt="make a patch",
            cwd="/tmp/project",
        )
        _DeferredThread.created[-1].run_now()

        result = manager.get_session(launched["id"])
        assert result["state"] == "failed"
        assert result["backend_unavailable"] is True
        assert "ChatGPT" in result["error"]
        assert result["result"] is None

    @pytest.mark.parametrize("provider", ["", "search", "gpt"])
    def test_rejects_unknown_providers(self, provider):
        from spoke.agent_backends import AgentBackendManager

        manager = AgentBackendManager(
            backend_runner=MagicMock(),
            thread_factory=_DeferredThread,
        )

        with pytest.raises(ValueError, match="Unsupported agent backend"):
            manager.launch(provider=provider, prompt="hello", cwd="/tmp/project")

    def test_rejects_empty_prompt(self):
        from spoke.agent_backends import AgentBackendManager

        manager = AgentBackendManager(
            backend_runner=MagicMock(),
            thread_factory=_DeferredThread,
        )

        with pytest.raises(ValueError, match="prompt must be a non-empty string"):
            manager.launch(provider="codex", prompt="   ", cwd="/tmp/project")

    def test_cancelled_session_does_not_publish_result(self):
        from spoke.agent_backends import AgentBackendManager, AgentBackendRunResult

        _DeferredThread.created = []

        def fake_runner(provider, prompt, cwd, resume_id, cancel_check, event_sink):
            assert cancel_check() is True
            return AgentBackendRunResult(
                provider=provider,
                session_id="codex-thread-123",
                final_response="Should be discarded",
            )

        manager = AgentBackendManager(
            backend_runner=fake_runner,
            thread_factory=_DeferredThread,
        )

        launched = manager.launch(provider="codex", prompt="continue", cwd="/tmp/project")
        cancelled = manager.cancel(launched["id"])
        assert cancelled["state"] == "cancelling"

        _DeferredThread.created[-1].run_now()

        result = manager.get_session(launched["id"])
        assert result["state"] == "cancelled"
        assert result["result"] is None
        assert result["provider_session_id"] is None


class TestAgentBackendToolDispatch:
    def test_generic_tool_schemas_do_not_expose_raw_agent_backend_controls(self):
        from spoke import tool_dispatch

        schemas = tool_dispatch.get_tool_schemas()
        names = {schema["function"]["name"] for schema in schemas}

        assert "launch_agent_session" not in names
        assert "list_agent_sessions" not in names
        assert "get_agent_session_result" not in names
        assert "cancel_agent_session" not in names

    def test_execute_tool_does_not_launch_agent_backend_sessions_from_generic_surface(self):
        from spoke import tool_dispatch

        fake_manager = MagicMock()
        result = tool_dispatch.execute_tool(
            "launch_agent_session",
            {
                "provider": "codex",
                "prompt": "make a plan",
                "cwd": "/tmp/project",
                "resume_id": "thread-1",
            },
            agent_backend_manager=fake_manager,
        )

        assert json.loads(result) == {"error": "Unknown tool: launch_agent_session"}
        fake_manager.launch.assert_not_called()

    def test_operator_prompt_does_not_present_agent_backends_as_generic_tools(self):
        import spoke.command as command

        assert "launch_agent_session" not in command.COMMAND_SYSTEM_PROMPT
        assert "Claude Agent SDK" not in command.COMMAND_SYSTEM_PROMPT
        assert "Codex SDK" not in command.COMMAND_SYSTEM_PROMPT

    def test_dispatch_module_does_not_import_agent_backends_directly(self):
        module_path = Path(__file__).resolve().parents[1] / "spoke" / "tool_dispatch.py"
        text = module_path.read_text(encoding="utf-8")

        assert "claude_agent_sdk" not in text
        assert "codex_app_server" not in text

    def test_backend_module_does_not_offer_api_credit_or_claude_agent_sdk_path(self):
        module_path = Path(__file__).resolve().parents[1] / "spoke" / "agent_backends.py"
        text = module_path.read_text(encoding="utf-8")

        assert "claude_agent_sdk" not in text
        assert "Claude Agent SDK" not in text
        assert "ANTHROPIC_API_KEY" not in text
        assert "OPENAI_API_KEY" not in text
        assert "api key" not in text.casefold()


class TestAgentBackendPresentation:
    def test_presenter_maps_events_to_compact_actions_without_raw_output(self):
        from spoke.agent_backend_presenter import (
            AgentBackendPresentationState,
            present_backend_events,
        )

        state = AgentBackendPresentationState()
        actions = present_backend_events(
            [
                {
                    "sequence": 1,
                    "kind": "command_execution",
                    "text": "pytest tests/test_agent_backends.py\nSECRET OUTPUT",
                    "data": {
                        "id": "cmd-1",
                        "type": "command_execution",
                        "command": "pytest tests/test_agent_backends.py",
                        "aggregated_output": "SECRET OUTPUT",
                        "status": "in_progress",
                    },
                },
                {
                    "sequence": 2,
                    "kind": "reasoning",
                    "text": "checking the focused test",
                    "data": {
                        "id": "reason-1",
                        "type": "reasoning",
                        "text": "checking the focused test",
                    },
                },
            ],
            state,
        )

        assert [(action.kind, action.text) for action in actions] == [
            ("tool_start", ""),
            ("status", "Codex running: pytest tests/test_agent_backends.py"),
            ("narrator_summary", "checking the focused test"),
        ]
        assert all("SECRET OUTPUT" not in action.text for action in actions)

    def test_presenter_streams_only_new_agent_message_text(self):
        from spoke.agent_backend_presenter import (
            AgentBackendPresentationState,
            present_backend_events,
        )

        state = AgentBackendPresentationState()
        first = present_backend_events(
            [
                {
                    "sequence": 1,
                    "kind": "agent_message",
                    "text": "hello",
                    "data": {"id": "msg-1", "type": "agent_message", "text": "hello"},
                }
            ],
            state,
        )
        second = present_backend_events(
            [
                {
                    "sequence": 2,
                    "kind": "agent_message",
                    "text": "hello world",
                    "data": {
                        "id": "msg-1",
                        "type": "agent_message",
                        "text": "hello world",
                    },
                }
            ],
            state,
        )

        assert [(action.kind, action.text) for action in first] == [
            ("response_delta", "hello")
        ]
        assert [(action.kind, action.text) for action in second] == [
            ("response_delta", " world")
        ]

    def test_presenter_exposes_backend_liveness_actions(self):
        from spoke.agent_backend_presenter import present_backend_liveness

        actions = present_backend_liveness("Codex")

        assert [(action.kind, action.text) for action in actions] == [
            ("narrator_summary", "Codex thinking"),
            ("narrator_shimmer", ""),
        ]
        assert actions[-1].active is True


class TestAgentShellRouting:
    def test_active_agent_shell_routes_ordinary_input_to_selected_provider(self):
        from spoke.agent_shell import AgentShellState, route_agent_shell_input

        state = AgentShellState(
            active=True,
            provider="codex",
            spoke_session_id="agent-backend-codex-1",
            provider_session_id="thread-abc",
            cwd="/tmp/project",
        )

        decision = route_agent_shell_input(
            "inspect the failing test and propose the smallest fix",
            state,
        )

        assert decision.kind == "provider_message"
        assert decision.provider == "codex"
        assert decision.spoke_session_id == "agent-backend-codex-1"
        assert decision.provider_session_id == "thread-abc"
        assert decision.cwd == "/tmp/project"
        assert decision.text == "inspect the failing test and propose the smallest fix"

    @pytest.mark.parametrize(
        "utterance",
        [
            "epistaxis zetesis how fares the tyrant state",
            "zetesis is there incoherence between these lanes",
            "how fares the tyrant state",
            "run Epístaxis zetesis with --3.1-flash-lite and ask what's up",
        ],
    )
    def test_active_agent_shell_does_not_intercept_epistaxis_text_without_executor(
        self, utterance
    ):
        from spoke.agent_shell import AgentShellState, route_agent_shell_input

        state = AgentShellState(
            active=True,
            provider="codex",
            spoke_session_id="agent-backend-codex-1",
            provider_session_id="session-xyz",
            cwd="/tmp/project",
        )

        decision = route_agent_shell_input(utterance, state)

        assert decision.kind == "provider_message"
        assert decision.provider == "codex"
        assert decision.provider_session_id == "session-xyz"
        assert decision.text == utterance

    def test_active_agent_shell_routes_provider_switch_as_mode_control(self):
        from spoke.agent_shell import AgentShellState, route_agent_shell_input

        state = AgentShellState(active=True, provider="claude-code", cwd="/tmp/project")

        decision = route_agent_shell_input("switch to codex", state)

        assert decision.kind == "mode_control"
        assert decision.control_action == "switch_provider"
        assert decision.provider == "codex"

    def test_active_agent_shell_recognizes_claude_code_as_distinct_cli_backend(self):
        from spoke.agent_shell import AgentShellState, route_agent_shell_input

        state = AgentShellState(active=True, provider="codex", cwd="/tmp/project")

        decision = route_agent_shell_input("switch to Claude Code", state)

        assert decision.kind == "mode_control"
        assert decision.control_action == "switch_provider"
        assert decision.provider == "claude-code"

    def test_inactive_agent_shell_leaves_input_for_normal_assistant(self):
        from spoke.agent_shell import AgentShellState, route_agent_shell_input

        state = AgentShellState(active=False, provider=None, cwd="/tmp/project")

        decision = route_agent_shell_input("inspect the failing test", state)

        assert decision.kind == "normal_assistant"
        assert decision.text == "inspect the failing test"


class TestAgentShellMenuState:
    def test_delegate_exposes_agent_shell_provider_menu_without_backend_replacement(
        self, monkeypatch
    ):
        import spoke.__main__ as main_module

        delegate = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
        delegate._command_client = MagicMock()
        delegate._command_backend = "local"
        delegate._command_model_id = "qwen-test"
        delegate._command_model_options = [("qwen-test", "qwen-test", True)]
        delegate._command_server_unreachable = False
        delegate._agent_shell_provider = "codex"
        delegate._agent_backend_manager = MagicMock()
        delegate._load_cloud_provider_preference = MagicMock(return_value="google")
        delegate._load_preference = MagicMock(return_value=None)
        delegate._select_model = MagicMock(return_value=[])
        delegate._sanitize_model_ids = MagicMock(side_effect=lambda a, b: (a, b))
        delegate._default_transcription_model = MagicMock(return_value="whisper-base")
        delegate._launch_target_menu_state = MagicMock(return_value=None)
        delegate._local_whisper_controls_available = MagicMock(return_value=False)
        delegate._tts_client = None
        delegate._tts_backend = "local"
        delegate._tts_sidecar_url = ""
        delegate._whisper_backend = "local"
        delegate._preview_backend = "local"
        delegate._whisper_sidecar_url = ""
        delegate._whisper_cloud_url = ""
        delegate._whisper_cloud_api_key = ""

        state = delegate._handle_model_menu_action(None)

        assert state["assistant_backend"]["items"][0][1] == "Local OMLX"
        assert state["agent_shell"] == {
            "title": "Agent Shell",
            "items": [
                ("off", "Off", False, True),
                ("codex", "Codex", True, True),
                ("claude-code", "Claude Code", False, False),
            ],
        }


class TestAgentShellDelegateDispatch:
    def test_send_text_routes_active_agent_shell_to_backend_manager(
        self, main_module, monkeypatch
    ):
        monkeypatch.setattr(main_module.threading, "Thread", _ImmediateThread)
        delegate = _make_agent_shell_delegate(main_module)
        delegate._agent_shell_provider = "codex"
        delegate._agent_backend_manager = _FakeAgentBackendManager(result="Patch looks good.")

        delegate._send_text_as_command("inspect the failing test")

        assert delegate._agent_backend_manager.launched == [
            {
                "provider": "codex",
                "prompt": "inspect the failing test",
                "cwd": str(Path.cwd()),
                "resume_id": None,
            }
        ]
        delegate._command_client.stream_command_events.assert_not_called()
        assert delegate._agent_shell_sessions["codex"] == {
            "spoke_session_id": "agent-backend-codex-1",
            "provider_session_id": "codex-provider-session-1",
        }
        calls = delegate.performSelectorOnMainThread_withObject_waitUntilDone_.call_args_list
        assert calls[0].args[0] == "commandUtteranceReady:"
        assert calls[-1].args[0] == "commandComplete:"
        assert calls[-1].args[1]["response"] == "Patch looks good."

    def test_backend_failure_does_not_claim_session_started_first(
        self, main_module, monkeypatch
    ):
        monkeypatch.setattr(main_module.threading, "Thread", _ImmediateThread)
        delegate = _make_agent_shell_delegate(main_module)
        delegate._agent_shell_provider = "codex"
        delegate._agent_backend_manager = _FakeAgentBackendManager(final_state="failed")

        delegate._send_text_as_command("inspect the failing test")

        calls = delegate.performSelectorOnMainThread_withObject_waitUntilDone_.call_args_list
        token_texts = [
            call.args[1]["text"]
            for call in calls
            if call.args[0] == "commandToken:" and "text" in call.args[1]
        ]
        assert not any("Started Codex session" in text for text in token_texts)
        assert calls[-1].args[0] == "commandFailed:"
        assert calls[-1].args[1]["error"] == "backend failed"

    def test_running_backend_events_are_presented_without_raw_output(
        self, main_module, monkeypatch
    ):
        monkeypatch.setattr(main_module.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(main_module.time, "sleep", lambda _seconds: None)
        delegate = _make_agent_shell_delegate(main_module)
        delegate._agent_shell_provider = "codex"
        delegate._agent_backend_manager = _StreamingFakeAgentBackendManager()

        delegate._send_text_as_command("inspect the failing test")

        calls = delegate.performSelectorOnMainThread_withObject_waitUntilDone_.call_args_list
        token_texts = [
            call.args[1]["text"]
            for call in calls
            if call.args[0] == "commandToken:" and "text" in call.args[1]
        ]
        assert "Codex running: pytest tests/test_agent_backends.py\n" in token_texts
        assert all("SECRET OUTPUT" not in text for text in token_texts)
        assert any(call.args[0] == "commandToolStart:" for call in calls)
        assert any(
            call.args[0] == "narratorSummary:"
            and call.args[1]["summary"] == "checking the focused test"
            for call in calls
        )
        assert calls[-1].args[0] == "commandComplete:"
        assert calls[-1].args[1]["response"] == "done"

    def test_quiet_running_agent_shell_keeps_thinking_gloss_alive(
        self, main_module, monkeypatch
    ):
        monkeypatch.setattr(main_module.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(main_module.time, "sleep", lambda _seconds: None)
        delegate = _make_agent_shell_delegate(main_module)
        delegate._agent_shell_provider = "codex"
        delegate._agent_backend_manager = _QuietRunningAgentBackendManager()

        delegate._send_text_as_command("keep working")

        calls = delegate.performSelectorOnMainThread_withObject_waitUntilDone_.call_args_list
        assert any(
            call.args[0] == "narratorSummary:"
            and call.args[1]["summary"] == "Codex thinking"
            for call in calls
        )
        assert any(
            call.args[0] == "narratorShimmer:"
            and call.args[1]["active"] is True
            for call in calls
        )
        assert any(
            call.args[0] == "narratorShimmer:"
            and call.args[1]["active"] is False
            for call in calls
        )
        assert calls[-1].args[0] == "commandComplete:"
        assert calls[-1].args[1]["response"] == "quiet done"

    def test_epistaxis_shaped_text_stays_with_active_agent_shell_until_executor_exists(
        self, main_module, monkeypatch
    ):
        monkeypatch.setattr(main_module.threading, "Thread", _ImmediateThread)
        delegate = _make_agent_shell_delegate(main_module)
        delegate._agent_shell_provider = "codex"
        delegate._agent_backend_manager = _FakeAgentBackendManager()

        delegate._send_text_as_command("epistaxis zetesis how fares the tyrant state")

        assert delegate._agent_backend_manager.launched == [
            {
                "provider": "codex",
                "prompt": "epistaxis zetesis how fares the tyrant state",
                "cwd": str(Path.cwd()),
                "resume_id": None,
            }
        ]
        delegate._command_client.stream_command_events.assert_not_called()

    def test_provider_switch_is_handled_as_mode_control_not_provider_or_assistant(
        self, main_module, monkeypatch
    ):
        monkeypatch.setattr(main_module.threading, "Thread", _ImmediateThread)
        delegate = _make_agent_shell_delegate(main_module)
        delegate._agent_shell_provider = "claude-code"
        delegate._agent_backend_manager = _FakeAgentBackendManager()

        delegate._send_text_as_command("switch to codex")

        assert delegate._agent_shell_provider == "codex"
        delegate._agent_backend_manager.launched == []
        delegate._command_client.stream_command_events.assert_not_called()
        delegate._save_preference.assert_called_once_with("agent_shell_provider", "codex")
        calls = delegate.performSelectorOnMainThread_withObject_waitUntilDone_.call_args_list
        assert calls[-1].args[0] == "commandComplete:"
        assert "Agent Shell switched to Codex" in calls[-1].args[1]["response"]

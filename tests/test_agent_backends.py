"""Tests for local-auth operator agent backend sessions."""

from __future__ import annotations

import json
import signal
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


class _PreemptiveAgentBackendManager:
    def __init__(self):
        self.launched: list[dict] = []
        self.cancelled: list[str] = []

    def launch(self, **kwargs):
        self.launched.append(kwargs)
        provider = kwargs["provider"]
        return {
            "id": f"agent-backend-{provider}-new",
            "provider": provider,
            "state": "completed",
            "provider_session_id": f"{provider}-provider-new",
            "result": "replacement complete",
            "error": None,
        }

    def get_session(self, session_id):
        if session_id.endswith("-old"):
            return {
                "id": session_id,
                "provider": "gemini-cli",
                "state": "running",
                "provider_session_id": "gemini-provider-old",
                "result": None,
                "error": None,
            }
        return {
            "id": session_id,
            "provider": "gemini-cli",
            "state": "completed",
            "provider_session_id": "gemini-provider-new",
            "result": "replacement complete",
            "error": None,
        }

    def cancel(self, session_id):
        self.cancelled.append(session_id)
        return {"id": session_id, "state": "cancelling"}


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
        monkeypatch.setenv("ANTHROPIC_API_KEY", "forbidden")
        monkeypatch.setenv("GEMINI_API_KEY", "forbidden")
        monkeypatch.setenv("GOOGLE_API_KEY", "forbidden")
        monkeypatch.setenv("GOOGLE_GENAI_API_KEY", "forbidden")
        monkeypatch.setenv("CODEX_HOME", "/tmp/codex-home")

        env = _subscription_only_env()

        assert "OPENAI_API_KEY" not in env
        assert "CODEX_API_KEY" not in env
        assert "ANTHROPIC_API_KEY" not in env
        assert "GEMINI_API_KEY" not in env
        assert "GOOGLE_API_KEY" not in env
        assert "GOOGLE_GENAI_API_KEY" not in env
        assert env["CODEX_HOME"] == "/tmp/codex-home"

    def test_codex_backend_rejects_non_chatgpt_login_status(self, monkeypatch):
        from spoke.agent_backends import AgentBackendUnavailable, _require_codex_subscription_login

        def fake_status(_codex_path, _env):
            return "Logged in using billing credentials"

        monkeypatch.setattr("spoke.agent_backends._codex_login_status", fake_status)

        with pytest.raises(AgentBackendUnavailable, match="ChatGPT subscription"):
            _require_codex_subscription_login("/usr/local/bin/codex", {})

    def test_claude_backend_requires_subscription_auth_status(self, monkeypatch):
        from spoke.agent_backends import (
            AgentBackendUnavailable,
            _require_claude_subscription_login,
        )

        def fake_status(_claude_path, _env):
            return {
                "loggedIn": True,
                "authMethod": "console",
                "apiProvider": "firstParty",
                "subscriptionType": "max",
            }

        monkeypatch.setattr("spoke.agent_backends._claude_auth_status", fake_status)

        with pytest.raises(AgentBackendUnavailable, match="subscription auth"):
            _require_claude_subscription_login("/opt/homebrew/bin/claude", {})

    def test_claude_stream_events_preserve_session_output_and_limit_shape(self):
        from spoke.agent_backends import _events_from_claude_stream_event

        init_events = _events_from_claude_stream_event(
            {
                "type": "system",
                "subtype": "init",
                "cwd": "/tmp/spoke",
                "session_id": "claude-thread-1",
                "model": "claude-opus-4-6[1m]",
                "claude_code_version": "2.1.81",
                "apiKeySource": "none",
            }
        )
        assistant_events = _events_from_claude_stream_event(
            {
                "type": "assistant",
                "session_id": "claude-thread-1",
                "message": {
                    "content": [{"type": "text", "text": "Claude docked."}],
                },
            }
        )
        limit_events = _events_from_claude_stream_event(
            {
                "type": "rate_limit_event",
                "rate_limit_info": {
                    "rateLimitType": "five_hour",
                    "status": "allowed",
                    "isUsingOverage": False,
                },
            }
        )

        assert [(event.kind, event.text, event.data) for event in init_events] == [
            (
                "session_metadata",
                "/tmp/spoke",
                {
                    "provider_session_id": "claude-thread-1",
                    "cwd": "/tmp/spoke",
                    "model": "claude-opus-4-6[1m]",
                    "cli_version": "2.1.81",
                    "credential_source": "none",
                },
            )
        ]
        assert [(event.kind, event.text, event.data) for event in assistant_events] == [
            (
                "agent_message",
                "Claude docked.",
                {"type": "agent_message", "text": "Claude docked."},
            )
        ]
        assert [(event.kind, event.text, event.data) for event in limit_events] == [
            (
                "usage_limits",
                "",
                {
                    "rate_limit_type": "five_hour",
                    "status": "allowed",
                    "is_using_overage": False,
                },
            )
        ]

    def test_gemini_stream_events_preserve_session_tool_and_stats_shape(self):
        from spoke.agent_backends import _events_from_gemini_stream_event

        init_events = _events_from_gemini_stream_event(
            {
                "type": "init",
                "session_id": "gemini-thread-1",
                "model": "auto-gemini-3",
            },
            cwd="/tmp/spoke",
        )
        assistant_events = _events_from_gemini_stream_event(
            {
                "type": "message",
                "role": "assistant",
                "content": "Gemini docked.",
                "delta": True,
            },
            cwd="/tmp/spoke",
        )
        tool_events = _events_from_gemini_stream_event(
            {
                "type": "tool_use",
                "tool_name": "run_shell_command",
                "tool_id": "tool-1",
                "parameters": {"command": "pwd", "description": "Print cwd."},
            },
            cwd="/tmp/spoke",
        )
        result_events = _events_from_gemini_stream_event(
            {
                "type": "result",
                "status": "success",
                "stats": {
                    "duration_ms": 3116,
                    "tool_calls": 1,
                    "models": {"gemini-3-flash-preview": {"total_tokens": 10}},
                },
            },
            cwd="/tmp/spoke",
        )

        assert [(event.kind, event.text, event.data) for event in init_events] == [
            (
                "session_metadata",
                "/tmp/spoke",
                {
                    "provider_session_id": "gemini-thread-1",
                    "cwd": "/tmp/spoke",
                    "model": "auto-gemini-3",
                },
            )
        ]
        assert [(event.kind, event.text) for event in assistant_events] == [
            ("agent_message", "Gemini docked.")
        ]
        assert [(event.kind, event.text, event.data["tool_name"]) for event in tool_events] == [
            ("tool_use", "run_shell_command: pwd", "run_shell_command")
        ]
        assert [(event.kind, event.data["tool_calls"]) for event in result_events] == [
            ("usage_limits", 1)
        ]

    def test_gemini_command_wraps_prompt_for_compact_operator_shell(self):
        from spoke.agent_backends import _gemini_command

        command = _gemini_command(
            gemini_path="/usr/local/bin/gemini",
            prompt="Inspect the Agent Shell backend.",
            resume_id=None,
        )

        prompt = command[2]
        assert "compact operator shell" in prompt
        assert "do not print an implementation plan" in prompt.lower()
        assert "Inspect the Agent Shell backend." in prompt

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

    def test_codex_log_events_preserve_shell_metadata_shape(self):
        from spoke.agent_backends import _events_from_codex_stream_event

        session_events = _events_from_codex_stream_event(
            {
                "type": "session_meta",
                "payload": {
                    "id": "codex-thread-1",
                    "cwd": "/tmp/spoke",
                    "cli_version": "0.125.0",
                    "model_provider": "openai",
                },
            }
        )
        context_events = _events_from_codex_stream_event(
            {
                "type": "turn_context",
                "payload": {
                    "cwd": "/tmp/spoke",
                    "model": "gpt-5.5",
                },
            }
        )
        usage_events = _events_from_codex_stream_event(
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "rate_limits": {
                        "primary": {"used_percent": 19.0, "window_minutes": 300},
                        "secondary": {"used_percent": 18.0, "window_minutes": 10080},
                        "plan_type": "pro",
                    },
                },
            }
        )

        assert len(session_events) == 1
        assert session_events[0].kind == "session_metadata"
        assert session_events[0].text == "/tmp/spoke"
        assert session_events[0].data == {
            "provider_session_id": "codex-thread-1",
            "cwd": "/tmp/spoke",
            "cli_version": "0.125.0",
            "model_provider": "openai",
        }
        assert context_events[0].kind == "session_metadata"
        assert context_events[0].data == {"cwd": "/tmp/spoke", "model": "gpt-5.5"}
        assert usage_events[0].kind == "usage_limits"
        assert usage_events[0].data["five_hour_percent"] == 19.0
        assert usage_events[0].data["seven_day_percent"] == 18.0
        assert usage_events[0].data["plan_type"] == "pro"

    def test_codex_session_log_backfills_metadata_events(self, tmp_path):
        from spoke.agent_backends import _events_from_codex_session_log

        session_id = "019dd871-0786-7243-89d3-849cc0fb023e"
        log_dir = tmp_path / "sessions" / "2026" / "04" / "29"
        log_dir.mkdir(parents=True)
        log_path = log_dir / f"rollout-2026-04-29T04-52-59-{session_id}.jsonl"
        log_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "session_meta",
                            "payload": {
                                "id": session_id,
                                "cwd": "/private/tmp/spoke",
                                "cli_version": "0.125.0",
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "turn_context",
                            "payload": {"cwd": "/private/tmp/spoke", "model": "gpt-5.5"},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "event_msg",
                            "payload": {
                                "type": "token_count",
                                "rate_limits": {
                                    "primary": {
                                        "used_percent": 19.0,
                                        "window_minutes": 300,
                                    },
                                    "secondary": {
                                        "used_percent": 18.0,
                                        "window_minutes": 10080,
                                    },
                                },
                            },
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )

        events = _events_from_codex_session_log(session_id, codex_home=tmp_path)

        assert [(event.kind, event.data) for event in events] == [
            (
                "session_metadata",
                {
                    "provider_session_id": session_id,
                    "cwd": "/private/tmp/spoke",
                    "cli_version": "0.125.0",
                },
            ),
            ("session_metadata", {"cwd": "/private/tmp/spoke", "model": "gpt-5.5"}),
            (
                "usage_limits",
                {"five_hour_percent": 19.0, "seven_day_percent": 18.0},
            ),
        ]

    def test_codex_session_log_extracts_thread_waypoints_from_assistant_messages(
        self, tmp_path
    ):
        from spoke.agent_backends import _events_from_codex_session_log

        session_id = "019dd871-0786-7243-89d3-849cc0fb023e"
        log_dir = tmp_path / "sessions" / "2026" / "04" / "29"
        log_dir.mkdir(parents=True)
        log_path = log_dir / f"rollout-2026-04-29T04-52-59-{session_id}.jsonl"
        log_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "response_item",
                            "payload": {
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": (
                                            "**Anagnosis**\n"
                                            "The lane is turning Codex logs into "
                                            "Agent Thread Cards.\n"
                                            "Next step is a read-side extractor."
                                        ),
                                    }
                                ],
                            },
                        }
                    )
                ]
            ),
            encoding="utf-8",
        )

        events = _events_from_codex_session_log(session_id, codex_home=tmp_path)

        assert [(event.kind, event.data["kind"], event.text) for event in events] == [
            (
                "thread_waypoint",
                "anagnosis",
                "The lane is turning Codex logs into Agent Thread Cards.\n"
                "Next step is a read-side extractor.",
            )
        ]

    def test_public_session_exposes_thread_card_from_waypoints(self):
        from spoke.agent_backends import AgentBackendManager

        session = {
            "id": "agent-backend-codex-1",
            "provider": "codex",
            "prompt": "build thread cards",
            "cwd": "/tmp/spoke",
            "resume_id": None,
            "state": "completed",
            "created_at": "now",
            "started_at": "now",
            "finished_at": "later",
            "provider_session_id": "codex-thread-1",
            "result": "Done. Cards are available.",
            "events": [
                {
                    "sequence": 1,
                    "kind": "thread_waypoint",
                    "text": "The lane is building thread-card bearings.",
                    "data": {
                        "kind": "anagnosis",
                        "source": "codex-log",
                        "sequence": 1,
                    },
                }
            ],
            "event_counter": 1,
            "error": None,
            "backend_unavailable": False,
        }

        public = AgentBackendManager._public_session(session)

        assert public["thread_card"] == {
            "thread_id": "agent-backend-codex-1",
            "provider": "codex",
            "title": "building thread-card bearings",
            "readiness": "ready",
            "bearing": "The lane is building thread-card bearings.",
            "activity_line": "Ready to read",
            "latest_response": "Done. Cards are available.",
            "updated_sequence": 1,
        }

    def test_codex_identity_event_uses_epistaxis_resolver_before_transcript_regex(self):
        from spoke.agent_backends import AgentBackendEvent, _agent_shell_identity_event
        from spoke.agent_shell_identity import AgentShellIdentity

        def resolver(**kwargs):
            assert kwargs["provider"] == "codex"
            assert kwargs["provider_session_id"] == "codex-thread-1"
            assert kwargs["cwd"] == "/tmp/spoke"
            assert "regex-fallback-wrong" in kwargs["transcript_text"]
            return AgentShellIdentity(
                topos_name="codex-epistaxis-owned-topos",
                source="epistaxis-session-id",
                confidence="exact",
            )

        event = _agent_shell_identity_event(
            provider="codex",
            provider_session_id="codex-thread-1",
            cwd="/tmp/spoke",
            final_response="Created Topos: regex-fallback-wrong",
            events=[
                AgentBackendEvent(
                    kind="agent_message",
                    text="Created Topos: regex-fallback-wrong",
                )
            ],
            identity_resolver=resolver,
        )

        assert event == AgentBackendEvent(
            kind="topos_identity",
            text="codex-epistaxis-owned-topos",
            data={
                "name": "codex-epistaxis-owned-topos",
                "source": "epistaxis-session-id",
                "confidence": "exact",
                "provider_session_id": "codex-thread-1",
                "cwd": "/tmp/spoke",
            },
        )

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

    def test_launch_accepts_claude_code_through_shared_session_contract(self, tmp_path):
        from spoke.agent_backends import AgentBackendManager, AgentBackendRunResult

        calls = []
        _DeferredThread.created = []

        def fake_runner(provider, prompt, cwd, resume_id, cancel_check, event_sink):
            calls.append((provider, prompt, cwd, resume_id, cancel_check()))
            return AgentBackendRunResult(
                provider=provider,
                session_id="claude-thread-123",
                final_response="Claude plan complete.",
            )

        manager = AgentBackendManager(
            backend_runner=fake_runner,
            thread_factory=_DeferredThread,
        )

        launched = manager.launch(
            provider="claude-code",
            prompt="inspect the failing tests",
            cwd=str(tmp_path),
            resume_id="prior-claude-session",
        )
        assert launched["id"] == "agent-backend-claude-code-1"
        assert launched["provider"] == "claude-code"

        _DeferredThread.created[-1].run_now()

        result = manager.get_session(launched["id"])
        assert calls == [
            (
                "claude-code",
                "inspect the failing tests",
                str(tmp_path),
                "prior-claude-session",
                False,
            )
        ]
        assert result["state"] == "completed"
        assert result["provider_session_id"] == "claude-thread-123"
        assert result["thread_card"]["provider"] == "claude-code"
        assert result["thread_card"]["latest_response"] == "Claude plan complete."

    def test_launch_accepts_gemini_cli_through_shared_session_contract(self, tmp_path):
        from spoke.agent_backends import AgentBackendManager, AgentBackendRunResult

        calls = []
        _DeferredThread.created = []

        def fake_runner(provider, prompt, cwd, resume_id, cancel_check, event_sink):
            calls.append((provider, prompt, cwd, resume_id, cancel_check()))
            return AgentBackendRunResult(
                provider=provider,
                session_id="gemini-thread-123",
                final_response="Gemini plan complete.",
            )

        manager = AgentBackendManager(
            backend_runner=fake_runner,
            thread_factory=_DeferredThread,
        )

        launched = manager.launch(
            provider="gemini-cli",
            prompt="inspect the failing tests",
            cwd=str(tmp_path),
            resume_id="prior-gemini-session",
        )
        assert launched["id"] == "agent-backend-gemini-cli-1"
        assert launched["provider"] == "gemini-cli"

        _DeferredThread.created[-1].run_now()

        result = manager.get_session(launched["id"])
        assert calls == [
            (
                "gemini-cli",
                "inspect the failing tests",
                str(tmp_path),
                "prior-gemini-session",
                False,
            )
        ]
        assert result["state"] == "completed"
        assert result["provider_session_id"] == "gemini-thread-123"
        assert result["thread_card"]["provider"] == "gemini-cli"
        assert result["thread_card"]["latest_response"] == "Gemini plan complete."

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

    def test_cancel_terminates_registered_backend_process_group(self, monkeypatch):
        from spoke.agent_backends import (
            AgentBackendEvent,
            AgentBackendManager,
            AgentBackendRunResult,
        )

        started = threading.Event()
        release = threading.Event()
        killpg = MagicMock()
        monkeypatch.setattr("os.killpg", killpg)

        def fake_runner(provider, prompt, cwd, resume_id, cancel_check, event_sink):
            event_sink(
                AgentBackendEvent(
                    kind="_process_started",
                    data={"pid": 12345, "pgid": 67890},
                )
            )
            started.set()
            while not cancel_check() and not release.wait(0.01):
                pass
            return AgentBackendRunResult(
                provider=provider,
                session_id="gemini-thread-123",
                final_response="Should be discarded",
            )

        manager = AgentBackendManager(backend_runner=fake_runner)

        launched = manager.launch(
            provider="gemini-cli",
            prompt="run recon",
            cwd="/tmp/project",
        )
        assert started.wait(1)

        cancelled = manager.cancel(launched["id"])
        release.set()

        assert cancelled["state"] == "cancelling"
        killpg.assert_called_once_with(67890, signal.SIGTERM)
        assert all(
            event["kind"] != "_process_started"
            for event in manager.get_session(launched["id"])["backend_events"]
        )


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

    def test_presenter_composes_agent_shell_metadata_footer_and_topos_header(self):
        from spoke.agent_backend_presenter import (
            AgentBackendPresentationState,
            present_backend_events,
        )

        state = AgentBackendPresentationState()
        actions = present_backend_events(
            [
                {
                    "sequence": 1,
                    "kind": "session_metadata",
                    "text": "/private/tmp/spoke",
                    "data": {"cwd": "/private/tmp/spoke", "model": "gpt-5.5"},
                },
                {
                    "sequence": 2,
                    "kind": "usage_limits",
                    "data": {
                        "five_hour_percent": 19.0,
                        "seven_day_percent": 18.5,
                        "plan_type": "pro",
                    },
                },
                {
                    "sequence": 3,
                    "kind": "topos_identity",
                    "text": "codex-spoke-spinal-tap",
                    "data": {
                        "name": "codex-spoke-spinal-tap",
                        "source": "epistaxis-session-id",
                        "confidence": "exact",
                    },
                },
                {
                    "sequence": 4,
                    "kind": "agent_message",
                    "text": "Recorded the Spoke state.",
                    "data": {
                        "id": "msg-1",
                        "type": "agent_message",
                        "text": "Recorded the Spoke state.",
                    },
                },
            ],
            state,
        )

        assert ("metadata_footer", "model gpt-5.5 | cwd /private/tmp/spoke") in [
            (action.kind, action.text[:43]) for action in actions
        ]
        assert any(
            action.kind == "metadata_footer"
            and "5h 19%" in action.text
            and "7d 18.5%" in action.text
            and "pro" in action.text
            for action in actions
        )
        assert any(
            action.kind == "metadata_header"
            and action.text == "Topos: codex-spoke-spinal-tap"
            for action in actions
        )
        assert any(
            action.kind == "response_delta"
            and action.text == "Recorded the Spoke state."
            for action in actions
        )

    def test_presenter_labels_worktree_identity_as_worktree_context(self):
        from spoke.agent_backend_presenter import (
            AgentBackendPresentationState,
            present_backend_events,
        )

        actions = present_backend_events(
            [
                {
                    "sequence": 1,
                    "kind": "topos_identity",
                    "text": "codex-agent-sdk-partyline-spinal-tap-0428",
                    "data": {
                        "name": "codex-agent-sdk-partyline-spinal-tap-0428",
                        "source": "epistaxis-worktree",
                        "confidence": "exact",
                    },
                }
            ],
            AgentBackendPresentationState(),
        )

        assert [(action.kind, action.text) for action in actions] == [
            (
                "metadata_header",
                "Worktree: codex-agent-sdk-partyline-spinal-tap-0428",
            )
        ]

    def test_presenter_refreshes_identity_when_same_name_gets_stronger_source(self):
        from spoke.agent_backend_presenter import (
            AgentBackendPresentationState,
            present_backend_events,
        )

        state = AgentBackendPresentationState()
        first = present_backend_events(
            [
                {
                    "kind": "topos_identity",
                    "data": {
                        "name": "codex-agent-sdk-partyline-spinal-tap-0428",
                        "source": "epistaxis-worktree",
                    },
                }
            ],
            state,
        )
        second = present_backend_events(
            [
                {
                    "kind": "topos_identity",
                    "data": {
                        "name": "codex-agent-sdk-partyline-spinal-tap-0428",
                        "source": "epistaxis-session-id",
                    },
                }
            ],
            state,
        )

        assert first[-1].text == "Worktree: codex-agent-sdk-partyline-spinal-tap-0428"
        assert second[-1].text == "Topos: codex-agent-sdk-partyline-spinal-tap-0428"

    def test_presenter_does_not_infer_topos_from_tool_output_without_identity_event(self):
        from spoke.agent_backend_presenter import (
            AgentBackendPresentationState,
            present_backend_events,
        )

        actions = present_backend_events(
            [
                {
                    "sequence": 1,
                    "kind": "command_execution",
                    "text": "epistaxis update\nTopos: regex-fallback-wrong",
                    "data": {
                        "id": "cmd-1",
                        "type": "command_execution",
                        "command": "epistaxis update",
                        "aggregated_output": "Topos: regex-fallback-wrong",
                        "status": "completed",
                    },
                }
            ],
            AgentBackendPresentationState(),
        )

        assert all(action.kind != "metadata_header" for action in actions)

    def test_presenter_exposes_backend_liveness_actions(self):
        from spoke.agent_backend_presenter import present_backend_liveness

        actions = present_backend_liveness("Codex")

        assert [(action.kind, action.text) for action in actions] == [
            ("narrator_summary", "Codex thinking"),
            ("narrator_shimmer", ""),
        ]
        assert actions[-1].active is True

    def test_presenter_maps_running_thread_card_to_narrator_summary(self):
        from spoke.agent_backend_presenter import (
            AgentBackendPresentationState,
            present_thread_card,
        )

        state = AgentBackendPresentationState()
        card = {
            "title": "build Agent Shell cards",
            "readiness": "working",
            "activity_line": "Running: uv run pytest -q tests/test_agent_thread_cards.py",
        }

        first = present_thread_card(card, state)
        second = present_thread_card(card, state)

        assert [(action.kind, action.text) for action in first] == [
            (
                "narrator_summary",
                "build Agent Shell cards · Running: uv run pytest -q tests/test_agent_thread_cards.py",
            )
        ]
        assert second == []


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

    def test_active_agent_shell_recognizes_gemini_cli_as_distinct_cli_backend(self):
        from spoke.agent_shell import AgentShellState, route_agent_shell_input

        state = AgentShellState(active=True, provider="codex", cwd="/tmp/project")

        decision = route_agent_shell_input("switch to Gemini CLI", state)

        assert decision.kind == "mode_control"
        assert decision.control_action == "switch_provider"
        assert decision.provider == "gemini-cli"

    def test_active_agent_shell_routes_cancel_text_as_mode_control(self):
        from spoke.agent_shell import AgentShellState, route_agent_shell_input

        decision = route_agent_shell_input(
            "cancel this agent run",
            AgentShellState(active=True, provider="gemini-cli", cwd="/tmp/project"),
        )

        assert decision.kind == "mode_control"
        assert decision.control_action == "cancel_active_run"
        assert decision.provider == "gemini-cli"

    def test_inactive_agent_shell_leaves_input_for_normal_assistant(self):
        from spoke.agent_shell import AgentShellState, route_agent_shell_input

        state = AgentShellState(active=False, provider=None, cwd="/tmp/project")

        decision = route_agent_shell_input("inspect the failing test", state)

        assert decision.kind == "normal_assistant"
        assert decision.text == "inspect the failing test"


class TestAgentShellMenuState:
    def test_delegate_exposes_agent_shell_provider_menu_without_backend_replacement(
        self, monkeypatch, main_module
    ):
        delegate = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
        delegate._command_client = MagicMock()
        delegate._command_backend = "local"
        delegate._command_model_id = "qwen-test"
        delegate._command_model_options = [("qwen-test", "qwen-test", True)]
        delegate._command_server_unreachable = False
        delegate._agent_shell_provider = "codex"
        delegate._agent_backend_manager = MagicMock()
        delegate._agent_shell_sessions = {}
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
                ("codex-new-session", "Codex: New Session", False, True),
                ("claude-code", "Claude Code", False, False),
                ("gemini-cli", "Gemini CLI", False, False),
            ],
        }

    def test_delegate_exposes_agent_shell_session_catalog_menu_items(
        self, monkeypatch, main_module
    ):
        delegate = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
        delegate._agent_shell_provider = "codex"
        delegate._agent_backend_manager = MagicMock()
        delegate._agent_shell_sessions = {
            "codex": {
                "provider_session_id": "codex-thread-2",
                "sessions": [
                    {
                        "provider_session_id": "codex-thread-1",
                        "last_utterance": "first codex question",
                        "last_response": "first codex answer",
                    },
                    {
                        "provider_session_id": "codex-thread-2",
                        "last_utterance": "second codex question",
                        "last_response": "second codex answer",
                    },
                ],
            }
        }

        assert delegate._agent_shell_menu_state()["items"] == [
            ("off", "Off", False, True),
            ("codex", "Codex", True, True),
            ("codex-new-session", "Codex: New Session", False, True),
            ("claude-code", "Claude Code", False, False),
            ("gemini-cli", "Gemini CLI", False, False),
            ("codex-session:codex-thread-1", "Codex: first codex question", False, True),
            ("codex-session:codex-thread-2", "Codex: second codex question", True, True),
        ]

    def test_agent_shell_new_codex_session_clears_active_record_but_keeps_catalog(
        self, monkeypatch, main_module
    ):
        delegate = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
        delegate._agent_shell_provider = "codex"
        delegate._agent_backend_manager = MagicMock()
        delegate._agent_shell_sessions = {
            "codex": {
                "spoke_session_id": "spoke-old",
                "provider_session_id": "codex-thread-2",
                "last_utterance": "second codex question",
                "last_response": "second codex answer",
                "last_header": "Worktree: old-tree",
                "last_footer": "model gpt-5.5 | cwd /tmp/old",
                "sessions": [
                    {
                        "provider_session_id": "codex-thread-2",
                        "last_utterance": "second codex question",
                        "last_response": "second codex answer",
                        "last_header": "Worktree: old-tree",
                        "last_footer": "model gpt-5.5 | cwd /tmp/old",
                    }
                ],
            }
        }
        delegate._save_preference = MagicMock()
        delegate._menubar = MagicMock()
        delegate._command_overlay = MagicMock()
        delegate._command_overlay._visible = True
        delegate._command_client = MagicMock()
        delegate._command_client.history = []
        delegate._last_command_utterance = "local prompt"
        delegate._last_command_response = "local response"
        delegate._sync_command_overlay_brightness = MagicMock()
        delegate._detector = MagicMock()

        delegate._apply_agent_shell_selection("codex-new-session")

        record = delegate._agent_shell_sessions["codex"]
        assert delegate._agent_shell_provider == "codex"
        assert record["spoke_session_id"] is None
        assert record["provider_session_id"] is None
        assert record["last_utterance"] is None
        assert record["last_response"] is None
        assert record["last_header"] is None
        assert record["last_footer"] is None
        assert record["sessions"] == [
            {
                "provider_session_id": "codex-thread-2",
                "last_utterance": "second codex question",
                "last_response": "second codex answer",
                "last_header": "Worktree: old-tree",
                "last_footer": "model gpt-5.5 | cwd /tmp/old",
            }
        ]
        delegate._menubar.set_status_text.assert_called_with("Agent Shell: Codex new session")

    def test_agent_shell_session_selection_restores_catalog_snapshot(
        self, monkeypatch, main_module
    ):
        delegate = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
        delegate._agent_shell_provider = "off"
        delegate._agent_backend_manager = MagicMock()
        delegate._agent_shell_sessions = {
            "codex": {
                "provider_session_id": "codex-thread-2",
                "last_utterance": "second codex question",
                "last_response": "second codex answer",
                "sessions": [
                    {
                        "provider_session_id": "codex-thread-1",
                        "last_utterance": "first codex question",
                        "last_response": "first codex answer",
                    },
                    {
                        "provider_session_id": "codex-thread-2",
                        "last_utterance": "second codex question",
                        "last_response": "second codex answer",
                    },
                ],
            }
        }
        delegate._save_preference = MagicMock()
        delegate._menubar = MagicMock()
        delegate._command_overlay = None

        delegate._apply_agent_shell_selection("codex-session:codex-thread-1")

        assert delegate._agent_shell_provider == "codex"
        record = delegate._agent_shell_sessions["codex"]
        assert record["provider_session_id"] == "codex-thread-1"
        assert record["last_utterance"] == "first codex question"
        assert record["last_response"] == "first codex answer"
        assert delegate._save_preference.call_args_list[-1].args == (
            "agent_shell_overlay_snapshots",
            {
                "codex": {
                    "provider_session_id": "codex-thread-1",
                    "last_utterance": "first codex question",
                    "last_response": "first codex answer",
                    "sessions": [
                        {
                            "provider_session_id": "codex-thread-1",
                            "last_utterance": "first codex question",
                            "last_response": "first codex answer",
                        },
                        {
                            "provider_session_id": "codex-thread-2",
                            "last_utterance": "second codex question",
                            "last_response": "second codex answer",
                        },
                    ],
                }
            },
        )

    def test_agent_shell_session_selection_detaches_stale_spoke_run_identity(
        self, main_module
    ):
        delegate = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
        delegate._agent_shell_provider = "codex"
        delegate._agent_backend_manager = MagicMock()
        delegate._agent_backend_manager.get_session.return_value = {
            "id": "spoke-new",
            "provider_session_id": "codex-thread-2",
        }
        delegate._agent_shell_sessions = {
            "codex": {
                "spoke_session_id": "spoke-new",
                "provider_session_id": "codex-thread-2",
                "last_utterance": "second codex question",
                "last_response": "second codex answer",
                "sessions": [
                    {
                        "provider_session_id": "codex-thread-1",
                        "last_utterance": "first codex question",
                        "last_response": "first codex answer",
                    },
                    {
                        "provider_session_id": "codex-thread-2",
                        "last_utterance": "second codex question",
                        "last_response": "second codex answer",
                    },
                ],
            }
        }
        delegate._save_preference = MagicMock()
        delegate._menubar = MagicMock()
        delegate._command_overlay = None

        delegate._apply_agent_shell_selection("codex-session:codex-thread-1")
        record = delegate._agent_shell_session_record("codex")

        assert record["spoke_session_id"] is None
        assert record["provider_session_id"] == "codex-thread-1"
        assert record["last_utterance"] == "first codex question"
        assert delegate._agent_shell_state("codex").provider_session_id == "codex-thread-1"

    def test_agent_shell_thread_cards_snapshot_uses_catalog_sessions(self, main_module):
        delegate = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
        delegate._agent_shell_provider = "codex"
        delegate._agent_backend_manager = MagicMock()
        delegate._agent_shell_sessions = {
            "codex": {
                "provider_session_id": "codex-thread-2",
                "sessions": [
                    {
                        "provider_session_id": "codex-thread-1",
                        "last_utterance": "first codex question",
                        "last_response": "first codex answer",
                    },
                    {
                        "provider_session_id": "codex-thread-2",
                        "last_utterance": "second codex question",
                        "last_response": "second codex answer",
                        "thread_card": {
                            "provider_session_id": "codex-thread-2",
                            "title": "custom second thread",
                            "readiness": "ready",
                            "bearing": "custom bearing",
                            "activity_line": "Ready to read",
                            "latest_response": "second codex answer",
                        },
                    },
                ],
            }
        }

        cards = delegate._agent_shell_thread_cards_snapshot("codex")

        assert [card["provider_session_id"] for card in cards] == [
            "codex-thread-1",
            "codex-thread-2",
        ]
        assert cards[0]["title"] == "first codex question"
        assert cards[0]["selected"] is False
        assert cards[0]["display"]["display_state"] == "inactive"
        assert cards[0]["display"]["show_latest_response"] is False
        assert cards[1]["title"] == "custom second thread"
        assert cards[1]["selected"] is True
        assert cards[1]["display"]["display_state"] == "selected_resting"
        assert cards[1]["display"]["show_latest_response"] is True
        assert cards[1]["display"]["primary_text"] == "second codex answer"

    def test_agent_shell_chrome_events_persist_to_provider_record(self, main_module):
        delegate = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
        delegate._transcription_token = 7
        delegate._agent_shell_provider = "codex"
        delegate._agent_backend_manager = MagicMock()
        delegate._agent_shell_sessions = {"codex": {}}
        delegate._command_overlay = MagicMock()
        delegate._save_preference = MagicMock()

        delegate.agentShellHeader_({"token": 7, "text": "Worktree: codex-spinal-tap"})
        delegate.agentShellFooter_({"token": 7, "text": "model gpt-5.5 | cwd /tmp/spoke"})

        record = delegate._agent_shell_sessions["codex"]
        assert record["last_header"] == "Worktree: codex-spinal-tap"
        assert record["last_footer"] == "model gpt-5.5 | cwd /tmp/spoke"

    def test_recalling_agent_shell_snapshot_restores_persisted_chrome(self, main_module):
        delegate = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
        delegate._command_client = MagicMock()
        delegate._agent_shell_provider = "codex"
        delegate._agent_backend_manager = MagicMock()
        delegate._agent_shell_sessions = {
            "codex": {
                "provider_session_id": "codex-thread-1",
                "last_utterance": "hello codex",
                "last_response": "hello from codex",
                "last_header": "Worktree: codex-spinal-tap",
                "last_footer": "model gpt-5.5 | cwd /tmp/spoke",
                "sessions": [],
            }
        }
        delegate._command_overlay = MagicMock()
        delegate._command_overlay._visible = False
        delegate._detector = MagicMock()
        delegate._sync_command_overlay_brightness = MagicMock()
        delegate._transcribing = False

        delegate._toggle_command_overlay()

        delegate._command_overlay.show.assert_called_once()
        _, kwargs = delegate._command_overlay.show.call_args
        assert kwargs["agent_shell_header"] == "Worktree: codex-spinal-tap"
        assert kwargs["agent_shell_footer"] == "model gpt-5.5 | cwd /tmp/spoke"

    def test_switching_agent_shell_off_clears_visible_chrome(self, main_module):
        delegate = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
        delegate._agent_shell_provider = "codex"
        delegate._agent_backend_manager = MagicMock()
        delegate._command_overlay = MagicMock()
        delegate._command_overlay._visible = True
        delegate._command_overlay.clear_agent_shell_chrome = MagicMock()
        delegate._command_client = MagicMock()
        delegate._command_client.history = []
        delegate._last_command_utterance = "local prompt"
        delegate._last_command_response = "local response"
        delegate._save_preference = MagicMock()
        delegate._menubar = MagicMock()
        delegate._sync_command_overlay_brightness = MagicMock()
        delegate._detector = MagicMock()

        delegate._apply_agent_shell_selection("off")

        delegate._command_overlay.replace_transcript.assert_called_once_with(
            utterance="local prompt",
            response="local response",
            agent_shell_header="",
            agent_shell_footer="",
        )

    def test_repaint_visible_overlay_replaces_transcript_in_one_batch(self, main_module):
        delegate = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
        delegate._agent_shell_provider = "codex"
        delegate._agent_backend_manager = MagicMock()
        delegate._agent_shell_sessions = {
            "codex": {
                "provider_session_id": "codex-thread-1",
                "last_utterance": "hello codex",
                "last_response": "hello from codex",
                "last_header": "Worktree: codex-spinal-tap",
                "last_footer": "model gpt-5.5 | cwd /tmp/spoke",
                "sessions": [],
            }
        }
        delegate._command_overlay = MagicMock()
        delegate._command_overlay._visible = True
        delegate._command_client = MagicMock()
        delegate._sync_command_overlay_brightness = MagicMock()
        delegate._detector = MagicMock()

        delegate._repaint_visible_command_overlay_for_current_route()

        delegate._command_overlay.replace_transcript.assert_called_once_with(
            utterance="hello codex",
            response="hello from codex",
            agent_shell_header="Worktree: codex-spinal-tap",
            agent_shell_footer="model gpt-5.5 | cwd /tmp/spoke",
        )
        delegate._command_overlay.set_utterance.assert_not_called()
        delegate._command_overlay.set_response_text.assert_not_called()


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
        assert delegate._agent_shell_sessions["codex"]["spoke_session_id"] == (
            "agent-backend-codex-1"
        )
        assert delegate._agent_shell_sessions["codex"]["provider_session_id"] == (
            "codex-provider-session-1"
        )
        calls = delegate.performSelectorOnMainThread_withObject_waitUntilDone_.call_args_list
        assert calls[0].args[0] == "commandUtteranceReady:"
        assert calls[-1].args[0] == "commandComplete:"
        assert calls[-1].args[1]["response"] == "Patch looks good."

    def test_agent_shell_transcript_survives_deferred_utterance_overlay_setup(
        self, main_module, monkeypatch
    ):
        monkeypatch.setattr(main_module.threading, "Thread", _ImmediateThread)
        delegate = _make_agent_shell_delegate(main_module)
        delegate._agent_shell_provider = "codex"
        delegate._agent_backend_manager = _FakeAgentBackendManager(
            result="Patch looks good."
        )
        calls = []
        pending_utterance = None

        def perform(selector, payload, wait):
            nonlocal pending_utterance
            calls.append((selector, payload, wait))
            if selector == "commandUtteranceReady:":
                pending_utterance = payload
                return
            if selector == "commandComplete:" and pending_utterance is not None:
                delegate.commandUtteranceReady_(pending_utterance)
                pending_utterance = None
            method_name = selector.replace(":", "_")
            method = getattr(delegate, method_name, None)
            if callable(method):
                method(payload)

        delegate.performSelectorOnMainThread_withObject_waitUntilDone_ = perform

        delegate._send_text_as_command("inspect the failing test")

        assert delegate._agent_shell_sessions["codex"]["provider_session_id"] == (
            "codex-provider-session-1"
        )
        assert delegate._agent_shell_sessions["codex"]["last_utterance"] == (
            "inspect the failing test"
        )
        assert delegate._agent_shell_sessions["codex"]["last_response"] == (
            "Patch looks good."
        )
        assert delegate._save_preference.call_args_list[-1].args == (
            "agent_shell_overlay_snapshots",
            {
                "codex": {
                    "provider_session_id": "codex-provider-session-1",
                    "last_utterance": "inspect the failing test",
                    "last_response": "Patch looks good.",
                    "sessions": [
                        {
                            "provider_session_id": "codex-provider-session-1",
                            "last_utterance": "inspect the failing test",
                            "last_response": "Patch looks good.",
                        }
                    ],
                }
            },
        )
        assert any(
            call[0] == "commandComplete:" and call[1]["response"] == "Patch looks good."
            for call in calls
        )

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

    def test_backend_metadata_events_drive_agent_shell_chrome(
        self, main_module, monkeypatch
    ):
        class _MetadataBackendManager(_StreamingFakeAgentBackendManager):
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
                                "kind": "session_metadata",
                                "text": "/private/tmp/spoke",
                                "data": {
                                    "cwd": "/private/tmp/spoke",
                                    "model": "gpt-5.5",
                                },
                            },
                            {
                                "sequence": 2,
                                "kind": "usage_limits",
                                "data": {
                                    "five_hour_percent": 19.0,
                                    "seven_day_percent": 18.0,
                                },
                            },
                            {
                                "sequence": 3,
                                "kind": "topos_identity",
                                "text": "codex-spoke-spinal-tap",
                                "data": {
                                    "name": "codex-spoke-spinal-tap",
                                    "source": "epistaxis-worktree",
                                    "confidence": "exact",
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

        monkeypatch.setattr(main_module.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(main_module.time, "sleep", lambda _seconds: None)
        delegate = _make_agent_shell_delegate(main_module)
        delegate._agent_shell_provider = "codex"
        delegate._agent_backend_manager = _MetadataBackendManager()

        delegate._send_text_as_command("create your topos")

        calls = delegate.performSelectorOnMainThread_withObject_waitUntilDone_.call_args_list
        assert any(
            call.args[0] == "agentShellFooter:"
            and "model gpt-5.5" in call.args[1]["text"]
            and "5h 19%" in call.args[1]["text"]
            and "7d 18%" in call.args[1]["text"]
            for call in calls
        )
        assert any(
            call.args[0] == "agentShellHeader:"
            and call.args[1]["text"] == "Worktree: codex-spoke-spinal-tap"
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

    def test_running_agent_shell_thread_card_updates_narrator_when_no_reasoning(
        self, main_module, monkeypatch
    ):
        class _CardBackendManager(_StreamingFakeAgentBackendManager):
            def get_session(self, session_id):
                self.poll_count += 1
                if self.poll_count == 1:
                    return {
                        "id": session_id,
                        "provider": "codex",
                        "state": "running",
                        "provider_session_id": "codex-thread-1",
                        "backend_events": [],
                        "thread_card": {
                            "title": "build Agent Shell cards",
                            "readiness": "working",
                            "activity_line": "Running focused tests",
                        },
                        "result": None,
                        "error": None,
                    }
                return {
                    "id": session_id,
                    "provider": "codex",
                    "state": "completed",
                    "provider_session_id": "codex-thread-1",
                    "backend_events": [],
                    "thread_card": {
                        "title": "build Agent Shell cards",
                        "readiness": "ready",
                        "activity_line": "Ready to read",
                    },
                    "result": "done",
                    "error": None,
                }

        monkeypatch.setattr(main_module.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(main_module.time, "sleep", lambda _seconds: None)
        delegate = _make_agent_shell_delegate(main_module)
        delegate._agent_shell_provider = "codex"
        delegate._agent_backend_manager = _CardBackendManager()

        delegate._send_text_as_command("build the cards")

        calls = delegate.performSelectorOnMainThread_withObject_waitUntilDone_.call_args_list
        assert any(
            call.args[0] == "narratorSummary:"
            and call.args[1]["summary"] == "build Agent Shell cards · Running focused tests"
            for call in calls
        )
        assert calls[-1].args[0] == "commandComplete:"
        assert calls[-1].args[1]["response"] == "done"

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

    def test_agent_shell_cancel_text_cancels_active_provider_run(
        self, main_module, monkeypatch
    ):
        monkeypatch.setattr(main_module.threading, "Thread", _ImmediateThread)
        delegate = _make_agent_shell_delegate(main_module)
        delegate._agent_shell_provider = "gemini-cli"
        delegate._agent_backend_manager = _PreemptiveAgentBackendManager()
        delegate._agent_shell_sessions = {
            "gemini-cli": {
                "spoke_session_id": "agent-backend-gemini-cli-old",
                "active_spoke_session_id": "agent-backend-gemini-cli-old",
                "provider_session_id": "gemini-provider-old",
                "sessions": [],
            }
        }

        delegate._send_text_as_command("cancel this agent run")

        assert delegate._agent_backend_manager.cancelled == [
            "agent-backend-gemini-cli-old"
        ]
        assert delegate._agent_backend_manager.launched == []
        calls = delegate.performSelectorOnMainThread_withObject_waitUntilDone_.call_args_list
        assert calls[-1].args[0] == "commandComplete:"
        assert "Cancelling Gemini CLI" in calls[-1].args[1]["response"]

    def test_agent_shell_new_message_preempts_active_provider_run(
        self, main_module, monkeypatch
    ):
        monkeypatch.setattr(main_module.threading, "Thread", _ImmediateThread)
        delegate = _make_agent_shell_delegate(main_module)
        delegate._agent_shell_provider = "gemini-cli"
        delegate._agent_backend_manager = _PreemptiveAgentBackendManager()
        delegate._agent_shell_sessions = {
            "gemini-cli": {
                "spoke_session_id": "agent-backend-gemini-cli-old",
                "active_spoke_session_id": "agent-backend-gemini-cli-old",
                "provider_session_id": "gemini-provider-old",
                "sessions": [],
            }
        }

        delegate._send_text_as_command("look at the next file")

        assert delegate._agent_backend_manager.cancelled == [
            "agent-backend-gemini-cli-old"
        ]
        assert delegate._agent_backend_manager.launched == [
            {
                "provider": "gemini-cli",
                "prompt": "look at the next file",
                "cwd": str(Path.cwd()),
                "resume_id": "gemini-provider-old",
            }
        ]

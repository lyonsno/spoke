"""Tests for internal operator subagent support."""

import ast
from pathlib import Path
from unittest.mock import MagicMock

from spoke.command import CommandStreamEvent
import spoke.subagents as sub_mod


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


class TestSubagentManager:
    def test_launch_search_job_tracks_completion(self):
        calls = []
        _DeferredThread.created = []

        def fake_runner(prompt, cancel_check):
            calls.append((prompt, cancel_check()))
            return "Found the relevant code paths."

        manager = sub_mod.SubagentManager(
            search_runner=fake_runner,
            thread_factory=_DeferredThread,
        )

        launched = manager.launch("search", "find the narrator proxy wiring")
        assert launched["state"] == "queued"
        assert launched["kind"] == "search"

        thread = _DeferredThread.created[-1]
        thread.run_now()

        result = manager.get_job(launched["id"])
        assert calls == [("find the narrator proxy wiring", False)]
        assert result["state"] == "completed"
        assert result["result"] == "Found the relevant code paths."
        assert manager.list_jobs()[0]["id"] == launched["id"]

    def test_cancelled_job_does_not_publish_result(self):
        _DeferredThread.created = []

        def fake_runner(prompt, cancel_check):
            assert cancel_check() is True
            return "Should be discarded"

        manager = sub_mod.SubagentManager(
            search_runner=fake_runner,
            thread_factory=_DeferredThread,
        )

        launched = manager.launch("search", "find stale launch targets")
        cancelled = manager.cancel(launched["id"])
        assert cancelled["state"] == "cancelling"

        thread = _DeferredThread.created[-1]
        thread.run_now()

        result = manager.get_job(launched["id"])
        assert result["state"] == "cancelled"
        assert result["result"] is None

    def test_running_job_advises_against_tight_polling(self):
        manager = sub_mod.SubagentManager(
            search_runner=lambda prompt, cancel_check: "done later",
            thread_factory=_DeferredThread,
        )
        launched = manager.launch("search", "find overlay clipping bug")

        running = manager.get_job(launched["id"])
        assert running["state"] == "queued"
        assert "do not poll" in running["poll_hint"].lower()

    def test_subagents_module_does_not_import_tool_dispatch(self):
        module_path = Path(__file__).resolve().parents[1] / "spoke" / "subagents.py"
        tree = ast.parse(module_path.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name for alias in node.names}
                assert "spoke.tool_dispatch" not in names
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "spoke.tool_dispatch"
                assert node.module != "tool_dispatch"


class TestSearchRunner:
    def test_run_search_subagent_uses_isolated_history_and_search_tools(self):
        stream_events = [
            CommandStreamEvent(kind="assistant_delta", text="Searching..."),
            CommandStreamEvent(
                kind="assistant_final",
                text="Found `spoke/narrator.py` and related tests.",
            ),
        ]
        fake_client = MagicMock()
        fake_client.stream_command_events.return_value = iter(stream_events)
        fake_factory = MagicMock(return_value=fake_client)

        result = sub_mod.run_search_subagent_query(
            "find narrator proxy code",
            base_url="http://localhost:8090",
            model="qwen-test",
            tools=[{"function": {"name": "search_file"}}],
            tool_executor=MagicMock(return_value='{"matches": ""}'),
            api_key="secret",
            command_client_factory=fake_factory,
        )

        kwargs = fake_factory.call_args.kwargs
        assert kwargs["history_path"] is None
        assert kwargs["system_prompt"] == sub_mod._SEARCH_SUBAGENT_SYSTEM_PROMPT
        stream_kwargs = fake_client.stream_command_events.call_args.kwargs
        tool_names = {tool["function"]["name"] for tool in stream_kwargs["tools"]}
        assert tool_names == {"search_file"}
        assert result == "Searching..."

    def test_run_search_subagent_honors_cancel_check(self):
        stream_events = [
            CommandStreamEvent(
                kind="assistant_final",
                text="No result because the run was cancelled.",
            ),
        ]
        fake_client = MagicMock()
        fake_client.stream_command_events.return_value = iter(stream_events)
        fake_factory = MagicMock(return_value=fake_client)

        cancel_check = MagicMock(return_value=True)
        sub_mod.run_search_subagent_query(
            "find command history handling",
            base_url="http://localhost:8090",
            model="qwen-test",
            tools=[{"function": {"name": "find_file"}}],
            tool_executor=MagicMock(return_value='{"matches": []}'),
            command_client_factory=fake_factory,
            cancel_check=cancel_check,
        )

        stream_kwargs = fake_client.stream_command_events.call_args.kwargs
        assert stream_kwargs["cancel_check"] is cancel_check

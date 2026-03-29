"""Tests for tool dispatch in the command client.

Tests tool schema generation, tool call accumulation from SSE deltas,
local tool execution, and the multi-turn tool call loop.
"""

from __future__ import annotations

import importlib
import io
import json
import sys
import time
from unittest.mock import MagicMock, patch

import pytest


def _import_command():
    sys.modules.pop("spoke.command", None)
    return importlib.import_module("spoke.command")


def _import_tools():
    sys.modules.pop("spoke.tool_dispatch", None)
    return importlib.import_module("spoke.tool_dispatch")


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


# ── Tool schemas ─────────────────────────────────────────────────


class TestToolSchemas:
    def test_get_tool_schemas_returns_list(self):
        mod = _import_tools()
        schemas = mod.get_tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) >= 2

    def test_capture_context_schema(self):
        mod = _import_tools()
        schemas = mod.get_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert "capture_context" in names

        cap_schema = next(s for s in schemas if s["function"]["name"] == "capture_context")
        assert cap_schema["type"] == "function"
        params = cap_schema["function"]["parameters"]
        assert "scope" in params.get("properties", {})

    def test_read_aloud_schema(self):
        mod = _import_tools()
        schemas = mod.get_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert "read_aloud" in names

        ra_schema = next(s for s in schemas if s["function"]["name"] == "read_aloud")
        params = ra_schema["function"]["parameters"]
        assert "source_ref" in params.get("properties", {})

    def test_epistaxis_ops_schema(self):
        mod = _import_tools()
        schemas = mod.get_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert "run_epistaxis_ops" in names

        op_schema = next(s for s in schemas if s["function"]["name"] == "run_epistaxis_ops")
        params = op_schema["function"]["parameters"]
        assert "epistaxis_root" in params.get("properties", {})
        assert "target_repo" in params.get("properties", {})
        assert "operations" in params.get("properties", {})

    def test_query_gmail_schema(self):
        mod = _import_tools()
        schemas = mod.get_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert "query_gmail" in names

        gmail_schema = next(s for s in schemas if s["function"]["name"] == "query_gmail")
        params = gmail_schema["function"]["parameters"]
        assert "mode" in params.get("properties", {})
        assert "max_results" in params.get("properties", {})


# ── Tool call accumulation ───────────────────────────────────────


class TestAccumulateToolCalls:
    """Test accumulation of streamed tool call deltas."""

    def test_single_tool_call(self):
        mod = _import_tools()
        acc = mod.ToolCallAccumulator()

        # Index 0, function name
        acc.feed({"index": 0, "id": "call_abc", "function": {"name": "capture_context", "arguments": ""}})
        # Arguments arrive in chunks
        acc.feed({"index": 0, "function": {"arguments": '{"sc'}})
        acc.feed({"index": 0, "function": {"arguments": 'ope":'}})
        acc.feed({"index": 0, "function": {"arguments": ' "active_window"}'}})

        calls = acc.finish()
        assert len(calls) == 1
        assert calls[0]["id"] == "call_abc"
        assert calls[0]["function"]["name"] == "capture_context"
        assert json.loads(calls[0]["function"]["arguments"]) == {"scope": "active_window"}

    def test_multiple_tool_calls(self):
        mod = _import_tools()
        acc = mod.ToolCallAccumulator()

        acc.feed({"index": 0, "id": "call_1", "function": {"name": "capture_context", "arguments": ""}})
        acc.feed({"index": 0, "function": {"arguments": '{"scope": "screen"}'}})

        acc.feed({"index": 1, "id": "call_2", "function": {"name": "read_aloud", "arguments": ""}})
        acc.feed({"index": 1, "function": {"arguments": '{"source_ref": "literal:hello"}'}})

        calls = acc.finish()
        assert len(calls) == 2
        assert calls[0]["function"]["name"] == "capture_context"
        assert calls[1]["function"]["name"] == "read_aloud"

    def test_empty_accumulator(self):
        mod = _import_tools()
        acc = mod.ToolCallAccumulator()
        assert acc.finish() == []

    def test_has_calls(self):
        mod = _import_tools()
        acc = mod.ToolCallAccumulator()
        assert acc.has_calls is False
        acc.feed({"index": 0, "id": "call_1", "function": {"name": "test", "arguments": "{}"}})
        assert acc.has_calls is True


# ── Tool execution ───────────────────────────────────────────────


class TestExecuteTool:
    def test_execute_capture_context(self):
        """Test that execute_tool serializes a SceneCapture to the correct JSON shape."""
        mod = _import_tools()
        sc_mod = importlib.import_module("spoke.scene_capture")
        cache = sc_mod.SceneCaptureCache(max_captures=5)

        fake_capture = sc_mod.SceneCapture(
            scene_ref="scene-test",
            created_at=time.time(),
            scope="active_window",
            app_name="Safari",
            bundle_id="com.apple.Safari",
            window_title="Test Page",
            image_path="/tmp/test.png",
            image_size=(2560, 1440),
            model_image_size=(1280, 720),
            ocr_text="Hello World",
            ocr_blocks=[
                sc_mod.OCRBlock(ref="scene-test:block-0", text="Hello", bbox=(0, 0, 50, 20), confidence=0.99),
                sc_mod.OCRBlock(ref="scene-test:block-1", text="World", bbox=(60, 0, 50, 20), confidence=0.95),
            ],
            ax_hints=[
                sc_mod.AXHint(ref="scene-test:focus", role="AXTextField", label="Search"),
            ],
        )

        # Mock capture_context (the real function), not _execute_capture_context,
        # so the JSON serialization in execute_tool is actually exercised.
        with patch("spoke.scene_capture.capture_context", return_value=fake_capture):
            result = mod.execute_tool(
                name="capture_context",
                arguments={"scope": "active_window"},
                scene_cache=cache,
            )

        parsed = json.loads(result)
        assert parsed["scene_ref"] == "scene-test"
        assert parsed["app_name"] == "Safari"
        assert parsed["scope"] == "active_window"
        assert parsed["window_title"] == "Test Page"
        assert len(parsed["ocr_blocks"]) == 2
        assert parsed["ocr_blocks"][0]["ref"] == "scene-test:block-0"
        assert parsed["ocr_blocks"][0]["text"] == "Hello"
        assert len(parsed["ax_hints"]) == 1
        assert parsed["ax_hints"][0]["role"] == "AXTextField"

    def test_execute_capture_context_failure(self):
        """When capture fails, execute_tool returns an error JSON."""
        mod = _import_tools()
        with patch("spoke.scene_capture.capture_context", return_value=None):
            result = mod.execute_tool(
                name="capture_context",
                arguments={"scope": "active_window"},
            )
        parsed = json.loads(result)
        assert "error" in parsed

    def test_execute_read_aloud_literal(self):
        """Test that execute_tool resolves a literal ref and returns the text."""
        mod = _import_tools()
        result = mod.execute_tool(
            name="read_aloud",
            arguments={"source_ref": "literal:hello world"},
        )
        assert "hello world" in result

    def test_execute_read_aloud_invalid_ref(self):
        """Invalid ref should return an error string, not raise."""
        mod = _import_tools()
        result = mod.execute_tool(
            name="read_aloud",
            arguments={"source_ref": "bogus_kind:value"},
        )
        assert "error" in result.lower()

    def test_execute_unknown_tool(self):
        mod = _import_tools()
        result = mod.execute_tool(name="nonexistent", arguments={})
        assert "unknown tool" in result.lower() or "error" in result.lower()

    def test_execute_epistaxis_ops(self):
        mod = _import_tools()
        fake_result = {"target_repo": "spoke", "operations": [{"op": "git_status", "status": ""}]}
        fake_operator = MagicMock()
        fake_operator.execute_plan.return_value = fake_result["operations"]

        with patch("spoke.tool_dispatch.EpistaxisOperator", return_value=fake_operator):
            result = mod.execute_tool(
                name="run_epistaxis_ops",
                arguments={
                    "epistaxis_root": "/tmp/epistaxis-spoke-operator",
                    "target_repo": "spoke",
                    "operations": [{"op": "git_status"}],
                },
            )

        parsed = json.loads(result)
        assert parsed == fake_result

    def test_execute_epistaxis_ops_error(self):
        mod = _import_tools()
        with patch(
            "spoke.tool_dispatch.EpistaxisOperator",
            side_effect=mod.EpistaxisOperatorError("bad worktree"),
        ):
            result = mod.execute_tool(
                name="run_epistaxis_ops",
                arguments={
                    "epistaxis_root": "/Users/noahlyons/dev/epistaxis",
                    "target_repo": "spoke",
                    "operations": [{"op": "git_status"}],
                },
            )

        parsed = json.loads(result)
        assert parsed["error"] == "bad worktree"

    def test_execute_query_gmail(self):
        mod = _import_tools()
        fake_result = {
            "mode": "starred_recruiter_mail",
            "matched_count": 1,
            "messages": [{"id": "m-1"}],
        }
        fake_operator = MagicMock()
        fake_operator.execute_query.return_value = fake_result

        with patch("spoke.tool_dispatch.GmailOperator", return_value=fake_operator):
            result = mod.execute_tool(
                name="query_gmail",
                arguments={"mode": "starred_recruiter_mail", "max_results": 5},
            )

        assert json.loads(result) == fake_result

    def test_execute_query_gmail_error(self):
        mod = _import_tools()
        with patch(
            "spoke.tool_dispatch.GmailOperator",
            side_effect=mod.GmailOperatorError("missing credentials"),
        ):
            result = mod.execute_tool(
                name="query_gmail",
                arguments={"mode": "starred_recruiter_mail", "max_results": 5},
            )

        parsed = json.loads(result)
        assert parsed["error"] == "missing credentials"

    def test_execute_query_gmail_handles_null_max_results(self):
        mod = _import_tools()

        result = mod.execute_tool(
            name="query_gmail",
            arguments={"mode": "starred_recruiter_mail", "max_results": None},
        )

        parsed = json.loads(result)
        assert "error" in parsed


# ── Command client with tools ────────────────────────────────────


class TestCommandClientToolIntegration:
    """Test that CommandClient sends tool schemas and handles tool call responses."""

    def _make_client(self):
        cmd = _import_command()
        return cmd.CommandClient(
            base_url="http://localhost:9999",
            model="test",
            api_key="key",
        )

    def test_tools_included_in_request_when_provided(self):
        """When tools are provided, they should appear in the request body."""
        tools_mod = _import_tools()
        cmd = _import_command()
        client = self._make_client()

        chunks = [{"choices": [{"index": 0, "delta": {"content": "ok"}}]}]
        fake_resp = _make_sse_response(chunks)

        with patch("urllib.request.urlopen", return_value=fake_resp) as mock_open:
            list(client.stream_command("test", tools=tools_mod.get_tool_schemas()))
            req = mock_open.call_args[0][0]
            body = json.loads(req.data)
            assert "tools" in body
            assert len(body["tools"]) >= 2

    def test_tool_call_triggers_execution_and_followup(self):
        """When the model returns a tool call, execute it and send results back."""
        tools_mod = _import_tools()
        cmd = _import_command()
        client = self._make_client()

        # First response: model calls capture_context
        tool_call_chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [
                {"index": 0, "id": "call_abc", "type": "function",
                 "function": {"name": "capture_context", "arguments": ""}}
            ]}}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '{"scope": "active_window"}'}}
            ]}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ]
        first_resp = _make_sse_response(tool_call_chunks)

        # Second response: model gives final answer after tool result
        final_chunks = [
            {"choices": [{"index": 0, "delta": {"content": "I can see Safari."}}]},
        ]
        second_resp = _make_sse_response(final_chunks)

        call_count = {"n": 0}

        def fake_urlopen(req, timeout=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return first_resp
            return second_resp

        fake_result = '{"scene_ref": "scene-test", "scope": "active_window", "app_name": "Safari"}'
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with patch.object(tools_mod, "execute_tool", return_value=fake_result):
                tokens = list(client.stream_command(
                    "what's on screen",
                    tools=tools_mod.get_tool_schemas(),
                    tool_executor=tools_mod.execute_tool,
                ))

        assert "I can see Safari." in "".join(tokens)
        # Should have made 2 HTTP calls (initial + follow-up with tool result)
        assert call_count["n"] == 2

    def test_no_tool_calls_works_normally(self):
        """When model doesn't call tools, streaming works as before."""
        tools_mod = _import_tools()
        cmd = _import_command()
        client = self._make_client()

        chunks = [
            {"choices": [{"index": 0, "delta": {"content": "Hello"}}]},
            {"choices": [{"index": 0, "delta": {"content": " there"}}]},
        ]
        fake_resp = _make_sse_response(chunks)

        with patch("urllib.request.urlopen", return_value=fake_resp):
            tokens = list(client.stream_command(
                "hi",
                tools=tools_mod.get_tool_schemas(),
                tool_executor=tools_mod.execute_tool,
            ))

        assert tokens == ["Hello", " there"]

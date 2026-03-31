"""Tests for tool dispatch in the command client.

Tests tool schema generation, tool call accumulation from SSE deltas,
local tool execution, and the multi-turn tool call loop.
"""

from __future__ import annotations

import importlib
import io
import unittest.mock
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


    def test_list_directory_schema(self):
        mod = _import_tools()
        schemas = mod.get_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert "list_directory" in names

        schema = next(s for s in schemas if s["function"]["name"] == "list_directory")
        params = schema["function"]["parameters"]
        assert "dir_path" in params.get("properties", {})

    def test_read_file_schema(self):
        mod = _import_tools()
        schemas = mod.get_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert "read_file" in names

        schema = next(s for s in schemas if s["function"]["name"] == "read_file")
        params = schema["function"]["parameters"]
        assert "file_path" in params.get("properties", {})

    def test_write_file_schema(self):
        mod = _import_tools()
        schemas = mod.get_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert "write_file" in names

        schema = next(s for s in schemas if s["function"]["name"] == "write_file")
        params = schema["function"]["parameters"]
        assert "file_path" in params.get("properties", {})
        assert "content" in params.get("properties", {})

    def test_search_file_schema(self):
        mod = _import_tools()
        schemas = mod.get_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert "search_file" in names

        schema = next(s for s in schemas if s["function"]["name"] == "search_file")
        params = schema["function"]["parameters"]
        assert "pattern" in params.get("properties", {})

    def test_add_to_tray_schema(self):
        mod = _import_tools()
        schemas = mod.get_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert "add_to_tray" in names

        add_schema = next(s for s in schemas if s["function"]["name"] == "add_to_tray")
        params = add_schema["function"]["parameters"]
        assert "text" in params.get("properties", {})


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

    def test_execute_read_aloud_uses_blocking_speak(self):
        """read_aloud should block until TTS finishes so multi-call turns play sequentially."""
        mod = _import_tools()
        tts_client = MagicMock()

        result = mod.execute_tool(
            name="read_aloud",
            arguments={"source_ref": "literal:hello world"},
            tts_client=tts_client,
        )

        tts_client.speak.assert_called_once_with("hello world")
        assert result == "Speaking: hello world"

    def test_execute_read_aloud_tts_failure_returns_error(self):
        """Immediate TTS launch failures should surface as tool errors."""
        mod = _import_tools()
        tts_client = MagicMock()
        tts_client.speak.side_effect = RuntimeError("audio device unavailable")

        result = mod.execute_tool(
            name="read_aloud",
            arguments={"source_ref": "literal:hello world"},
            tts_client=tts_client,
        )

        assert result == "Error speaking text: TTS playback failed"

    def test_execute_read_aloud_invalid_ref(self):
        """Invalid ref should return an error string, not raise."""
        mod = _import_tools()
        result = mod.execute_tool(
            name="read_aloud",
            arguments={"source_ref": "bogus_kind:value"},
        )
        assert "error" in result.lower()

    def test_execute_add_to_tray_uses_callback(self):
        mod = _import_tools()
        tray_writer = MagicMock(
            return_value={"status": "added", "tray_visible": False, "stack_size": 1}
        )

        result = mod.execute_tool(
            name="add_to_tray",
            arguments={"text": "Save this"},
            tray_writer=tray_writer,
        )

        tray_writer.assert_called_once_with("Save this")
        parsed = json.loads(result)
        assert parsed["status"] == "added"
        assert parsed["tray_visible"] is False

    def test_execute_add_to_tray_without_callback_returns_error(self):
        mod = _import_tools()
        result = mod.execute_tool(
            name="add_to_tray",
            arguments={"text": "Save this"},
        )
        parsed = json.loads(result)
        assert "error" in parsed


    def test_execute_list_directory(self):
        mod = _import_tools()
        with patch("os.listdir", return_value=["file1.txt", "dir1"]):
            with patch("os.path.isdir", return_value=True):
                result = mod.execute_tool(
                    name="list_directory",
                    arguments={"dir_path": "/tmp/testdir"}
                )
        parsed = json.loads(result)
        assert "file1.txt" in parsed.get("contents", [])
        assert "dir1" in parsed.get("contents", [])

    def test_execute_read_file(self):
        mod = _import_tools()
        fake_content = "file content here"
        with patch("builtins.open", unittest.mock.mock_open(read_data=fake_content)):
            with patch("os.path.isfile", return_value=True):
                result = mod.execute_tool(
                    name="read_file",
                    arguments={"file_path": "/tmp/test.txt"}
                )
        parsed = json.loads(result)
        assert parsed.get("content") == fake_content

    def test_execute_write_file(self):
        mod = _import_tools()
        m = unittest.mock.mock_open()
        with patch("builtins.open", m):
            result = mod.execute_tool(
                name="write_file",
                arguments={"file_path": "/tmp/test.txt", "content": "new content"}
            )
        parsed = json.loads(result)
        assert parsed.get("status") == "success"
        m.assert_called_once_with("/tmp/test.txt", "w", encoding="utf-8")
        m().write.assert_called_once_with("new content")

    def test_execute_search_file(self):
        mod = _import_tools()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="match1\nmatch2", returncode=0)
            result = mod.execute_tool(
                name="search_file",
                arguments={"pattern": "match", "dir_path": "/tmp"}
            )
        parsed = json.loads(result)
        assert "match1" in parsed.get("matches", "")

    def test_execute_unknown_tool(self):
        mod = _import_tools()
        result = mod.execute_tool(name="nonexistent", arguments={})
        assert "unknown tool" in result.lower() or "error" in result.lower()


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



class TestExecuteToolIntegration:
    def test_execute_list_directory_real(self, tmp_path):
        mod = _import_tools()
        d = tmp_path / "mydir"
        d.mkdir()
        (d / "file.txt").write_text("hello")
        
        result = mod.execute_tool("list_directory", {"dir_path": str(d)})
        parsed = json.loads(result)
        assert "file.txt" in parsed.get("contents", [])

    def test_execute_read_file_real(self, tmp_path):
        mod = _import_tools()
        f = tmp_path / "hello.txt"
        f.write_text("world", encoding="utf-8")
        
        result = mod.execute_tool("read_file", {"file_path": str(f)})
        parsed = json.loads(result)
        assert parsed.get("content") == "world"

    def test_execute_write_file_real(self, tmp_path):
        mod = _import_tools()
        f = tmp_path / "nested" / "new.txt"
        
        result = mod.execute_tool("write_file", {"file_path": str(f), "content": "hello there"})
        parsed = json.loads(result)
        assert parsed.get("status") == "success"
        assert f.read_text(encoding="utf-8") == "hello there"

    def test_execute_search_file_real(self, tmp_path):
        mod = _import_tools()
        d = tmp_path / "src"
        d.mkdir()
        (d / "code.py").write_text("def find_me(): pass")
        
        result = mod.execute_tool("search_file", {"pattern": "find_me", "dir_path": str(d)})
        parsed = json.loads(result)
        assert "find_me" in parsed.get("matches", "")


    def test_execute_read_file_edge_cases(self, tmp_path):
        mod = _import_tools()
        # Null file_path
        res1 = json.loads(mod.execute_tool("read_file", {"file_path": None}))
        assert "error" in res1
        
        # Hallucinated end_line=0
        f = tmp_path / "zero.txt"
        f.write_text("line1\nline2")
        res2 = json.loads(mod.execute_tool("read_file", {"file_path": str(f), "end_line": 0}))
        assert "error" in res2
        assert "end_line must be >= 1" in res2["error"]

    def test_execute_write_file_sandbox(self, tmp_path, monkeypatch):
        mod = _import_tools()
        
        # Null file_path
        res1 = json.loads(mod.execute_tool("write_file", {"file_path": None}))
        assert "error" in res1
        
        import os
        # Set a predictable home dir for test
        monkeypatch.setattr(os.path, "expanduser", lambda path: str(tmp_path) if path == "~" else path)
        
        # Writing to home dir is allowed
        res2 = json.loads(mod.execute_tool("write_file", {"file_path": str(tmp_path / "ok.txt"), "content": "ok"}))
        assert res2.get("status") == "success"
        
        # Writing to sensitive .ssh is denied
        res3 = json.loads(mod.execute_tool("write_file", {"file_path": str(tmp_path / ".ssh" / "id_rsa"), "content": "bad"}))
        assert "error" in res3
        assert "Write access denied" in res3["error"]
        
        # Writing to system root is denied
        res4 = json.loads(mod.execute_tool("write_file", {"file_path": "/etc/passwd", "content": "bad"}))
        assert "error" in res4
        assert "Write access denied outside" in res4["error"]

    def test_execute_search_file_edge_cases(self):
        mod = _import_tools()
        # Null pattern
        res1 = json.loads(mod.execute_tool("search_file", {"pattern": None}))
        assert "error" in res1
        
        # Timeout simulated via monkeypatch
        import subprocess
        class TimeoutMock:
            def __init__(self, *args, **kwargs):
                raise subprocess.TimeoutExpired(cmd="grep", timeout=10)
        
        with unittest.mock.patch("subprocess.run", new=TimeoutMock):
            res2 = json.loads(mod.execute_tool("search_file", {"pattern": "slow", "dir_path": "."}))
            assert "error" in res2
            assert "timed out" in res2["error"]

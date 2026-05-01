from __future__ import annotations

import json

from spoke.acp_probe import (
    AcpSessionSupport,
    JsonRpcLineCodec,
    classify_session_support,
    provider_default_mode,
    provider_command,
    summarize_session_update,
)


def test_json_rpc_line_codec_round_trips_multiple_messages():
    codec = JsonRpcLineCodec()
    first = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    second = {"jsonrpc": "2.0", "method": "session/update", "params": {"x": 1}}

    encoded = codec.encode(first) + codec.encode(second)

    assert encoded == (
        json.dumps(first, separators=(",", ":")) + "\n"
        + json.dumps(second, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    assert codec.feed(encoded[:11]) == []
    assert codec.feed(encoded[11:]) == [first, second]


def test_classify_session_support_separates_live_sessions_from_history_rewind():
    init_response = {
        "protocolVersion": 1,
        "agentCapabilities": {
            "loadSession": True,
            "sessionCapabilities": {
                "list": {},
                "resume": {},
            },
        },
    }

    support = classify_session_support(init_response)

    assert support == AcpSessionSupport(
        live_session_switching=True,
        session_catalog=True,
        load_existing_session=True,
        resume_existing_session=True,
        fork_checkpoint=False,
        close_session=False,
    )


def test_summarize_session_update_marks_bearing_relevant_events():
    assert summarize_session_update(
        {
            "sessionUpdate": "agent_thought_chunk",
            "content": {"type": "text", "text": "Checking tests"},
        }
    ) == {
        "kind": "thought",
        "bearing_relevant": True,
        "text": "Checking tests",
    }
    assert summarize_session_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "tool-1",
            "title": "Read",
            "status": "pending",
            "kind": "read",
        }
    ) == {
        "kind": "tool_call",
        "bearing_relevant": True,
        "tool_call_id": "tool-1",
        "title": "Read",
        "status": "pending",
        "tool_kind": "read",
    }
    assert summarize_session_update(
        {
            "sessionUpdate": "session_info_update",
            "title": "ACP smoke",
            "updatedAt": "2026-05-01T00:00:00Z",
        }
    ) == {
        "kind": "session_info",
        "bearing_relevant": False,
        "title": "ACP smoke",
        "updated_at": "2026-05-01T00:00:00Z",
    }


def test_provider_command_prefers_known_acp_transports():
    assert provider_command("gemini-cli") == ("gemini", "--acp")
    assert provider_command("codex") == ("codex-acp",)
    assert provider_command("claude-code") == ("claude-agent-acp",)


def test_provider_default_modes_fence_gemini_git_mutation():
    assert provider_default_mode("codex") == "full-access"
    assert provider_default_mode("claude-code") == "bypassPermissions"
    assert provider_default_mode("gemini-cli") == "default"

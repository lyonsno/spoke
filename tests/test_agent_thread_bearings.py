"""Tests for local Agent Shell thread-bearing compilation."""

from __future__ import annotations


def test_scheduler_coalesces_fast_events_into_one_compile_per_thread():
    from spoke.agent_thread_bearings import (
        CompiledThreadBearing,
        ThreadBearingInput,
        ThreadBearingScheduler,
    )
    from spoke.agent_thread_cards import AgentThreadWaypoint

    calls: list[ThreadBearingInput] = []

    class RecordingCompiler:
        def compile(self, item: ThreadBearingInput) -> CompiledThreadBearing:
            calls.append(item)
            return CompiledThreadBearing(
                thread_id=item.thread_id,
                provider=item.provider,
                bearing=f"{item.thread_id}: {item.waypoints[-1].text}",
                activity_line="compiled",
                updated_sequence=item.updated_sequence,
                sources=tuple(waypoint.source for waypoint in item.waypoints),
                prompt="prompt",
                inference_status="fallback",
                input_signature=item.input_signature(),
            )

    now = 100.0
    scheduler = ThreadBearingScheduler(
        compiler=RecordingCompiler(),
        min_interval_s=2.0,
        clock=lambda: now,
    )
    scheduler.enqueue(
        ThreadBearingInput(
            thread_id="codex-a",
            provider="codex",
            waypoints=(
                AgentThreadWaypoint(
                    kind="current_intent",
                    text="First packet",
                    sequence=1,
                    source="agent_message",
                ),
            ),
            recent_events=(),
        )
    )
    scheduler.enqueue(
        ThreadBearingInput(
            thread_id="codex-a",
            provider="codex",
            waypoints=(
                AgentThreadWaypoint(
                    kind="current_intent",
                    text="Second packet wins",
                    sequence=2,
                    source="agent_message",
                ),
            ),
            recent_events=(
                {"kind": "command_execution", "sequence": 3, "data": {"command": "pytest"}},
            ),
        )
    )

    assert scheduler.flush_due() == []
    now = 102.1

    compiled = scheduler.flush_due()

    assert len(compiled) == 1
    assert len(calls) == 1
    assert calls[0].thread_id == "codex-a"
    assert calls[0].waypoints[0].text == "Second packet wins"
    assert calls[0].updated_sequence == 3


def test_scheduler_skips_unchanged_packets_after_compile():
    from spoke.agent_thread_bearings import DeterministicBearingCompiler, ThreadBearingInput, ThreadBearingScheduler
    from spoke.agent_thread_cards import AgentThreadWaypoint

    now = 10.0
    scheduler = ThreadBearingScheduler(
        compiler=DeterministicBearingCompiler(),
        min_interval_s=1.0,
        clock=lambda: now,
    )
    item = ThreadBearingInput(
        thread_id="codex-a",
        provider="codex",
        waypoints=(
            AgentThreadWaypoint(
                kind="anagnosis",
                text="Building compact cards.",
                sequence=5,
                source="codex-log",
            ),
        ),
        recent_events=(),
    )
    scheduler.enqueue(item)
    now = 11.1
    assert len(scheduler.flush_due()) == 1

    scheduler.enqueue(item)
    now = 12.2

    assert scheduler.flush_due() == []


def test_deterministic_compiler_fallback_is_source_attached_and_compact():
    from spoke.agent_thread_bearings import DeterministicBearingCompiler, ThreadBearingInput
    from spoke.agent_thread_cards import AgentThreadWaypoint

    compiler = DeterministicBearingCompiler(max_bearing_chars=120)
    item = ThreadBearingInput(
        thread_id="codex-local-bearing",
        provider="codex",
        title="Local bearing lane",
        waypoints=(
            AgentThreadWaypoint(
                kind="current_intent",
                text="- Session: compile bearings.\n- Repo/task: scheduler tests.\n- Immediate next step: implement fallback.",
                sequence=7,
                source="agent_message",
            ),
        ),
        recent_events=(
            {
                "kind": "file_change",
                "sequence": 8,
                "text": "spoke/agent_thread_bearings.py",
            },
        ),
    )

    compiled = compiler.compile(item)

    assert compiled.thread_id == "codex-local-bearing"
    assert compiled.provider == "codex"
    assert compiled.sources == ("agent_message", "file_change:8")
    assert compiled.inference_status == "fallback"
    assert "codex-local-bearing" in compiled.bearing
    assert "compile bearings" in compiled.bearing
    assert len(compiled.bearing) <= 120
    assert compiled.activity_line == "Edited spoke/agent_thread_bearings.py"
    assert "THREAD codex-local-bearing" in compiled.prompt


def test_model_client_result_is_bounded_and_still_source_attached():
    from spoke.agent_thread_bearings import DeterministicBearingCompiler, ThreadBearingInput
    from spoke.agent_thread_cards import AgentThreadWaypoint

    prompts: list[str] = []

    class FakeModelClient:
        def compile_bearing(self, prompt: str) -> dict[str, str]:
            prompts.append(prompt)
            return {
                "bearing": "A very useful local model summary that should be clamped after enough characters.",
                "activity_line": "Model activity",
            }

    compiler = DeterministicBearingCompiler(model_client=FakeModelClient(), max_bearing_chars=64)
    compiled = compiler.compile(
        ThreadBearingInput(
            thread_id="codex-a",
            provider="codex",
            waypoints=(
                AgentThreadWaypoint(
                    kind="current_intent",
                    text="Session: compress partyline packets.",
                    sequence=9,
                    source="agent_message",
                ),
            ),
            recent_events=(),
        )
    )

    assert prompts
    assert compiled.inference_status == "model"
    assert compiled.bearing.startswith("[codex/codex-a] A very useful")
    assert len(compiled.bearing) <= 64
    assert compiled.activity_line == "Model activity"
    assert compiled.sources == ("agent_message",)


def test_narrator_state_extracts_latest_prompt_summary_and_verbatim_tail():
    from spoke.agent_thread_narrator import build_agent_thread_narrator_state

    state = build_agent_thread_narrator_state(
        {
            "id": "spoke-session-1",
            "provider": "claude-code",
            "provider_session_id": "claude-provider-1",
            "prompt": "What did you change?",
            "state": "completed",
            "result": "\n".join(
                [
                    "I added the narrator state contract.",
                    "It now carries the latest prompt.",
                    "It also keeps the recent verbatim tail.",
                ]
            ),
            "backend_events": [
                {
                    "kind": "thread_waypoint",
                    "sequence": 4,
                    "text": "Session: implement the deterministic narrator state.",
                    "data": {"source": "anagnosis", "kind": "current_intent"},
                },
                {
                    "kind": "command_execution",
                    "sequence": 5,
                    "data": {
                        "status": "completed",
                        "command": "uv run pytest -q tests/test_agent_thread_bearings.py",
                    },
                },
            ],
            "thread_card": {
                "title": "Narrator state",
                "readiness": "ready",
                "activity_line": "Ready to read",
                "bearing": "Existing card bearing",
            },
        }
    )

    assert state.provider == "claude-code"
    assert state.thread_id == "claude-provider-1"
    assert state.provider_session_id == "claude-provider-1"
    assert state.updated_sequence == 5
    assert state.latest_user_prompt == "What did you change?"
    assert state.bearing == "Session: implement the deterministic narrator state."
    assert state.since_user_prompt == "Command completed; assistant produced 3 recent lines."
    assert state.latest_verbatim_tail == (
        "I added the narrator state contract.",
        "It now carries the latest prompt.",
        "It also keeps the recent verbatim tail.",
    )
    assert state.current_activity == "Ready to read"
    assert state.readiness == "ready"
    assert state.operator_needed is True
    assert state.confidence == "deterministic"
    assert "thread_waypoint:4" in state.source_event_ids
    assert "command_execution:5" in state.source_event_ids


def test_selected_overlay_response_uses_bearing_summary_and_recent_tail():
    from spoke.agent_thread_narrator import (
        build_agent_thread_narrator_state,
        format_selected_thread_narrator_response,
    )

    state = build_agent_thread_narrator_state(
        {
            "provider": "codex",
            "provider_session_id": "codex-thread-1",
            "last_utterance": "Summarize the smoke result.",
            "last_response": "Line one.\nLine two.\nLine three.",
            "thread_card": {
                "title": "Smoke result",
                "readiness": "ready",
                "bearing": "Smoke result lane",
            },
        }
    )

    response = format_selected_thread_narrator_response(state)

    assert response.splitlines() == [
        "Bearing: Smoke result lane",
        "Since your prompt: Assistant replied with 3 visible lines; no tool or state events captured in this slice.",
        "",
        "Recent output:",
        "Line one.",
        "Line two.",
        "Line three.",
    ]


def test_selected_overlay_bearing_does_not_echo_latest_user_prompt():
    from spoke.agent_thread_narrator import (
        build_agent_thread_narrator_state,
        format_selected_thread_narrator_response,
    )

    prompt = (
        "Let's see, so what I'm seeing here right now, I'm not sure if it's "
        "actually the most recent thing in the conversation."
    )
    state = build_agent_thread_narrator_state(
        {
            "provider": "claude-code",
            "provider_session_id": "claude-thread-echo",
            "last_utterance": prompt,
            "last_response": "No lost messages this time; we are in sync.",
            "thread_card": {
                "title": prompt,
                "bearing": prompt,
                "readiness": "ready",
            },
        }
    )

    response = format_selected_thread_narrator_response(state)

    assert state.bearing == "No durable bearing captured yet"
    assert prompt not in response.splitlines()[0]
    assert (
        "Since your prompt: Assistant replied with 1 visible line; "
        "no tool or state events captured in this slice."
    ) in response

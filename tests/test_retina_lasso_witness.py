import json
from datetime import datetime, timezone
from pathlib import Path

import spoke.retina_lasso_witness as witness
from spoke.retina_lasso_witness import (
    build_evidence_split,
    build_launch_target_command,
    build_retina_lasso_command,
    capture_count_for_window,
    collect_trace_events,
    default_fps_for_capture_profile,
    drive_hammer_toggles,
    drive_retarget_during_dismiss_pattern,
    read_trace_events_from_offset,
    run_autonomous_hammer_witness,
    run_trace_triggered_witness,
    run_witness_window,
    trace_event_is_open_ready,
    trace_event_output_slug,
    wait_for_open_ready_trace,
    write_witness_index,
)


def test_capture_count_for_window_rounds_up():
    assert capture_count_for_window(2.1, 10.0) == 21
    assert capture_count_for_window(0.01, 10.0) == 1


def test_capture_profile_defaults_separate_passive_and_stress_pressure():
    assert default_fps_for_capture_profile("low_perturbation") == 6.0
    assert default_fps_for_capture_profile("stress") == 15.0


def test_build_retina_lasso_command_prefers_global_capture_custody(tmp_path):
    command = build_retina_lasso_command(
        output_dir=tmp_path,
        count=3,
        interval_seconds=0.125,
        lane="warpstorm-pit-boss",
        diaulos="Warpstorm Pit Boss",
        source_app="Spoke",
        source_window="Command Overlay",
        trace_path=tmp_path / "trace.jsonl",
        capture_profile="low_perturbation",
        capture_command="/usr/local/bin/global-witness-capture",
    )

    assert command[0] == "/usr/local/bin/global-witness-capture"
    assert command[command.index("--output-dir") + 1] == str(tmp_path)
    assert command[command.index("--count") + 1] == "3"
    assert command[command.index("--interval") + 1] == "0.125000"
    assert command[command.index("--capture-profile") + 1] == "low-perturbation"
    assert command[command.index("--lane") + 1] == "warpstorm-pit-boss"
    assert command[command.index("--diaulos") + 1] == "Warpstorm Pit Boss"
    assert command[command.index("--source-app") + 1] == "Spoke"
    assert command[command.index("--source-window") + 1] == "Command Overlay"
    assert command[command.index("--trace-path") + 1] == str(tmp_path / "trace.jsonl")


def test_build_retina_lasso_command_passes_bounded_global_capture_args(tmp_path):
    command = build_retina_lasso_command(
        output_dir=tmp_path,
        count=3,
        interval_seconds=0.125,
        lane="warpstorm-pit-boss",
        diaulos="Warpstorm Pit Boss",
        source_app="Spoke",
        source_window="Command Overlay",
        trace_path=tmp_path / "trace.jsonl",
        capture_profile="stress",
        capture_command="/usr/local/bin/global-witness-capture",
        capture_mode="rect",
        capture_rect="0,360,2048,850",
        display_id=1,
    )

    assert command[command.index("--mode") + 1] == "rect"
    assert command[command.index("--rect") + 1] == "0,360,2048,850"
    assert command[command.index("--display-id") + 1] == "1"


def test_build_retina_lasso_command_rejects_bounded_legacy_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(witness, "_default_global_capture_command", lambda: None)

    try:
        build_retina_lasso_command(
            output_dir=tmp_path,
            count=1,
            interval_seconds=1.0,
            lane="warpstorm-pit-boss",
            diaulos="Warpstorm Pit Boss",
            source_app="Spoke",
            source_window="Command Overlay",
            capture_mode="rect",
            capture_rect="0,360,2048,850",
        )
    except ValueError as exc:
        assert "bounded capture requires global witness capture" in str(exc)
    else:
        raise AssertionError("bounded capture must not silently fall back to legacy full-screen capture")


def test_build_retina_lasso_command_uses_healthy_global_capture_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(witness.shutil, "which", lambda name: "/usr/local/bin/global-witness-capture")
    monkeypatch.setattr(witness, "_global_capture_command_is_healthy", lambda command: True)

    command = build_retina_lasso_command(
        output_dir=tmp_path,
        count=1,
        interval_seconds=1.0,
        lane="warpstorm-pit-boss",
        diaulos="Warpstorm Pit Boss",
        source_app="Spoke",
        source_window="Command Overlay",
    )

    assert command[0] == "/usr/local/bin/global-witness-capture"


def test_run_witness_window_passes_bounded_capture_contract(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    runner_calls = []

    def runner(command, cwd=None, check=False):
        runner_calls.append(command)
        capture_dir = Path(command[command.index("--output-dir") + 1])
        capture_dir.mkdir(parents=True, exist_ok=True)
        (capture_dir / "manifest.json").write_text(json.dumps({"frames": []}), encoding="utf-8")

    run_witness_window(
        trace_path=trace_path,
        output_dir=tmp_path / "capture",
        duration_seconds=0.25,
        fps=4.0,
        capture_profile="stress",
        capture_command="/usr/local/bin/global-witness-capture",
        capture_mode="rect",
        capture_rect="0,360,2048,850",
        runner=runner,
    )

    command = runner_calls[0]
    assert command[command.index("--mode") + 1] == "rect"
    assert command[command.index("--rect") + 1] == "0,360,2048,850"


def test_build_retina_lasso_command_skips_unhealthy_global_capture(tmp_path, monkeypatch):
    monkeypatch.setenv("UV_BIN", "/opt/homebrew/bin/uv")
    monkeypatch.setattr(witness.shutil, "which", lambda name: "/usr/local/bin/global-witness-capture")
    monkeypatch.setattr(witness, "_global_capture_command_is_healthy", lambda command: False)
    monkeypatch.setattr(witness.Path, "home", staticmethod(lambda: tmp_path))

    command = build_retina_lasso_command(
        output_dir=tmp_path,
        count=1,
        interval_seconds=1.0,
        lane="warpstorm-pit-boss",
        diaulos="Warpstorm Pit Boss",
        source_app="Spoke",
        source_window="Command Overlay",
    )

    assert command[:3] == ["/opt/homebrew/bin/uv", "run", "perceptasia-screen-capture"]


def test_build_retina_lasso_command_uses_absolute_uv_from_env(tmp_path, monkeypatch):
    monkeypatch.setattr(witness, "_default_global_capture_command", lambda: None)
    monkeypatch.setenv("UV_BIN", "/opt/homebrew/bin/uv")

    command = build_retina_lasso_command(
        output_dir=tmp_path,
        count=1,
        interval_seconds=1.0,
        lane="warpstorm-pit-boss",
        diaulos="Warpstorm Pit Boss",
        source_app="Spoke",
        source_window="Command Overlay",
    )

    assert command[:3] == ["/opt/homebrew/bin/uv", "run", "perceptasia-screen-capture"]


def test_build_retina_lasso_command_uses_common_uv_path_without_shell_path(tmp_path, monkeypatch):
    monkeypatch.setattr(witness, "_default_global_capture_command", lambda: None)
    monkeypatch.delenv("UV_BIN", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(witness.shutil, "which", lambda name: None)
    monkeypatch.setattr(witness.Path, "home", staticmethod(lambda: tmp_path))
    local_uv = tmp_path / ".local" / "bin" / "uv"
    local_uv.parent.mkdir(parents=True)
    local_uv.write_text("#!/bin/sh\n", encoding="utf-8")
    local_uv.chmod(0o755)

    command = build_retina_lasso_command(
        output_dir=tmp_path,
        count=1,
        interval_seconds=1.0,
        lane="warpstorm-pit-boss",
        diaulos="Warpstorm Pit Boss",
        source_app="Spoke",
        source_window="Command Overlay",
    )

    assert command[:3] == [str(local_uv), "run", "perceptasia-screen-capture"]


def test_build_launch_target_command_uses_repo_script(tmp_path):
    command = build_launch_target_command(tmp_path, "habeas_target")

    assert command == [str(tmp_path / "scripts" / "launch-target.sh"), "habeas_target"]


def test_drive_hammer_toggles_posts_exact_count_without_trailing_sleep():
    calls = []
    sleeps = []

    def toggle_chord(**kwargs):
        calls.append(kwargs)

    drive_hammer_toggles(
        count=3,
        interval_seconds=0.2,
        key_pause_seconds=0.01,
        toggle_chord=toggle_chord,
        sleep=sleeps.append,
    )

    assert calls == [
        {"key_pause_seconds": 0.01},
        {"key_pause_seconds": 0.01},
        {"key_pause_seconds": 0.01},
    ]
    assert sleeps == [0.2, 0.2]


def test_drive_retarget_during_dismiss_pattern_hits_ordered_edges():
    calls = []
    sleeps = []

    def toggle_chord(**kwargs):
        calls.append(kwargs)

    drive_retarget_during_dismiss_pattern(
        repeats=2,
        open_dwell_seconds=0.7,
        dismiss_retarget_delay_seconds=0.08,
        reopen_dwell_seconds=0.6,
        cycle_pause_seconds=0.2,
        key_pause_seconds=0.01,
        toggle_chord=toggle_chord,
        sleep=sleeps.append,
    )

    assert calls == [{"key_pause_seconds": 0.01}] * 8
    assert sleeps == [0.7, 0.08, 0.6, 0.2, 0.7, 0.08, 0.6]


def test_trace_event_is_open_ready_requires_visible_text_and_ready_body():
    assert trace_event_is_open_ready(
        {
            "event": "overlay.visual_ready.push",
            "presentation_text_state": "visible",
            "presentation_body_state": "body_ready",
        }
    )
    assert not trace_event_is_open_ready(
        {
            "event": "overlay.visual_ready.push",
            "presentation_text_state": "hidden",
            "presentation_body_state": "body_ready",
        }
    )
    assert not trace_event_is_open_ready(
        {
            "event": "overlay.visual_ready.push",
            "presentation_text_state": "visible",
            "presentation_body_state": "materializing",
        }
    )


def test_wait_for_open_ready_trace_returns_ready_event_after_offset(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-22T00:00:00Z",
                "event": "overlay.visual_ready.push",
                "presentation_text_state": "visible",
                "presentation_body_state": "body_ready",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    offset = trace_path.stat().st_size
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": "2026-05-22T00:00:01Z",
                    "event": "overlay.visual_ready.push",
                    "presentation_text_state": "hidden",
                    "presentation_body_state": "body_ready",
                    "presentation_generation": 2,
                }
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {
                    "timestamp": "2026-05-22T00:00:02Z",
                    "event": "overlay.show.begin",
                    "presentation_requested_state": "opening",
                    "presentation_generation": 2,
                }
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {
                    "timestamp": "2026-05-22T00:00:03Z",
                    "event": "overlay.visual_ready.push",
                    "presentation_text_state": "visible",
                    "presentation_body_state": "body_ready",
                    "presentation_generation": 2,
                }
            )
            + "\n"
        )

    calls = iter([0.0, 0.01])
    new_offset, event = wait_for_open_ready_trace(
        trace_path,
        offset=offset,
        timeout_seconds=1.0,
        poll_interval_seconds=0.01,
        monotonic=lambda: next(calls),
        sleep=lambda _seconds: None,
    )

    assert new_offset == trace_path.stat().st_size
    assert event is not None
    assert event["timestamp"] == "2026-05-22T00:00:03Z"


def test_wait_for_open_ready_trace_ignores_stale_and_mismatched_ready_events(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": "2026-05-22T00:00:01Z",
                    "event": "overlay.visual_ready.push",
                    "presentation_text_state": "visible",
                    "presentation_body_state": "body_ready",
                    "presentation_generation": 1,
                }
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {
                    "timestamp": "2026-05-22T00:00:02Z",
                    "event": "overlay.show.begin",
                    "presentation_requested_state": "opening",
                    "presentation_generation": 2,
                }
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {
                    "timestamp": "2026-05-22T00:00:03Z",
                    "event": "overlay.visual_ready.push",
                    "presentation_text_state": "visible",
                    "presentation_body_state": "body_ready",
                    "presentation_generation": 1,
                }
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {
                    "timestamp": "2026-05-22T00:00:04Z",
                    "event": "overlay.visual_ready.push",
                    "presentation_text_state": "visible",
                    "presentation_body_state": "body_ready",
                    "presentation_generation": 2,
                }
            )
            + "\n"
        )

    calls = iter([0.0, 0.01])
    new_offset, event = wait_for_open_ready_trace(
        trace_path,
        offset=0,
        timeout_seconds=1.0,
        poll_interval_seconds=0.01,
        monotonic=lambda: next(calls),
        sleep=lambda _seconds: None,
    )

    assert new_offset == trace_path.stat().st_size
    assert event is not None
    assert event["timestamp"] == "2026-05-22T00:00:04Z"


def test_drive_retarget_during_dismiss_pattern_waits_for_open_ready_trace(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    calls = []

    def toggle_chord(**kwargs):
        calls.append("toggle")

    def ready_waiter(path, **kwargs):
        assert path == trace_path
        calls.append(("wait_ready", kwargs["offset"]))
        return kwargs["offset"] + 10, {
            "event": "overlay.visual_ready.push",
            "presentation_generation": kwargs["offset"] + 1,
        }

    results = drive_retarget_during_dismiss_pattern(
        repeats=1,
        open_dwell_seconds=0.7,
        dismiss_retarget_delay_seconds=0.08,
        reopen_dwell_seconds=0.6,
        key_pause_seconds=0.01,
        trace_path=trace_path,
        toggle_chord=toggle_chord,
        ready_waiter=ready_waiter,
        sleep=lambda seconds: calls.append(("sleep", seconds)),
    )

    assert calls == [
        "toggle",
        ("wait_ready", 0),
        ("sleep", 0.7),
        "toggle",
        ("sleep", 0.08),
        "toggle",
        ("wait_ready", 10),
        ("sleep", 0.6),
        "toggle",
    ]
    assert results == [
        {"cycle": 1, "phase": "open", "status": "ready", "presentation_generation": 1},
        {"cycle": 1, "phase": "reopen", "status": "ready", "presentation_generation": 11},
    ]


def test_collect_trace_events_filters_to_capture_window(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"timestamp": "2026-05-22T00:00:00Z", "event": "before"}),
                json.dumps({"timestamp": "2026-05-22T00:00:02Z", "event": "overlay.show.begin"}),
                json.dumps({"timestamp": "2026-05-22T00:00:03-00:00", "event": "overlay.visual_ready.push"}),
                json.dumps({"timestamp": "2026-05-22T00:00:05Z", "event": "after"}),
                "not-json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    events = collect_trace_events(
        trace_path,
        started_at=datetime(2026, 5, 22, 0, 0, 1, tzinfo=timezone.utc),
        ended_at=datetime(2026, 5, 22, 0, 0, 4, tzinfo=timezone.utc),
    )

    assert [event["event"] for event in events] == [
        "overlay.show.begin",
        "overlay.visual_ready.push",
    ]
    assert [event["trace_line"] for event in events] == [2, 3]


def test_read_trace_events_from_offset_returns_only_new_valid_jsonl(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps({"timestamp": "2026-05-22T00:00:00Z", "event": "before"}) + "\n",
        encoding="utf-8",
    )
    offset = trace_path.stat().st_size
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write("not-json\n")
        handle.write(json.dumps({"timestamp": "2026-05-22T00:00:01Z", "event": "overlay.show.begin"}) + "\n")

    new_offset, events = read_trace_events_from_offset(trace_path, offset=offset)

    assert new_offset == trace_path.stat().st_size
    assert [event["event"] for event in events] == ["overlay.show.begin"]


def test_read_trace_events_from_offset_resets_after_trace_truncation(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps({"timestamp": "2026-05-22T00:00:00Z", "event": "before", "padding": "x" * 200}) + "\n",
        encoding="utf-8",
    )
    stale_offset = trace_path.stat().st_size
    trace_path.write_text(
        json.dumps({"timestamp": "2026-05-22T00:00:01Z", "event": "overlay.show.begin"}) + "\n",
        encoding="utf-8",
    )

    new_offset, events = read_trace_events_from_offset(trace_path, offset=stale_offset)

    assert new_offset == trace_path.stat().st_size
    assert [event["event"] for event in events] == ["overlay.show.begin"]


def test_trace_event_output_slug_keeps_event_and_index_legible():
    slug = trace_event_output_slug(
        {"event": "overlay.show.retarget_dismiss_to_summon", "timestamp": "2026-05-22T00:00:01-04:00"},
        index=7,
    )

    assert slug.startswith("007-overlay-show-retarget_dismiss_to_summon-2026-05-22T00-00-01-04-00")


def test_trace_triggered_witness_captures_fade_out_receipts_by_default(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    output_dir = tmp_path / "watch"
    runner_calls = []
    sleep_calls = []
    monotonic_instants = iter([0.0, 0.0, 0.1])
    now_seconds = iter(range(8))

    def now():
        return datetime(2026, 5, 22, 0, 0, next(now_seconds), tzinfo=timezone.utc)

    def sleep(seconds):
        sleep_calls.append(seconds)
        if len(sleep_calls) == 1:
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "timestamp": "2026-05-22T00:00:01Z",
                            "event": "overlay.fade_out.start",
                            "presentation_generation": 4,
                        }
                    )
                    + "\n"
                )
                handle.write(
                    json.dumps(
                        {
                            "timestamp": "2026-05-22T00:00:02Z",
                            "event": "overlay.fade_out.complete",
                            "presentation_generation": 4,
                        }
                    )
                    + "\n"
                )

    def runner(command, cwd=None, check=False):
        runner_calls.append(command)
        capture_dir = Path(command[command.index("--output-dir") + 1])
        capture_dir.mkdir(parents=True, exist_ok=True)
        (capture_dir / "manifest.json").write_text(
            json.dumps({"frames": [{"path": "frame-000.png"}]}),
            encoding="utf-8",
        )

    watch_index_path = run_trace_triggered_witness(
        trace_path=trace_path,
        output_dir=output_dir,
        watch_timeout_seconds=1.0,
        event_capture_duration_seconds=0.25,
        poll_interval_seconds=0.01,
        max_captures=2,
        max_trigger_lag_seconds=10.0,
        fps=4.0,
        capture_command="/usr/local/bin/global-witness-capture",
        runner=runner,
        now=now,
        sleep=sleep,
        monotonic=lambda: next(monotonic_instants),
    )

    payload = json.loads(watch_index_path.read_text(encoding="utf-8"))
    assert len(runner_calls) == 2
    assert payload["capture_count"] == 2
    assert [capture["trigger_event"]["event"] for capture in payload["captures"]] == [
        "overlay.fade_out.start",
        "overlay.fade_out.complete",
    ]
    assert "overlay.fade_out.start" in payload["trigger_events"]
    assert "overlay.fade_out.complete" in payload["trigger_events"]
    assert payload["skipped_trigger_count"] == 0
    assert all("trigger_lag_seconds" in capture for capture in payload["captures"])


def test_trace_triggered_witness_skips_stale_trigger_instead_of_late_capture(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    output_dir = tmp_path / "watch"
    runner_calls = []
    sleep_calls = []
    monotonic_instants = iter([0.0, 0.0, 0.1, 1.1])
    now_values = iter(
        [
            datetime(2026, 5, 22, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 5, 22, 0, 0, 5, tzinfo=timezone.utc),
            datetime(2026, 5, 22, 0, 0, 6, tzinfo=timezone.utc),
        ]
    )

    def now():
        return next(now_values)

    def sleep(seconds):
        sleep_calls.append(seconds)
        if len(sleep_calls) == 1:
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "timestamp": "2026-05-22T00:00:01Z",
                            "event": "overlay.fade_out.start",
                            "presentation_generation": 4,
                        }
                    )
                    + "\n"
                )

    def runner(command, cwd=None, check=False):
        runner_calls.append(command)

    watch_index_path = run_trace_triggered_witness(
        trace_path=trace_path,
        output_dir=output_dir,
        watch_timeout_seconds=1.0,
        event_capture_duration_seconds=0.25,
        poll_interval_seconds=0.01,
        max_captures=2,
        max_trigger_lag_seconds=1.0,
        fps=4.0,
        capture_command="/usr/local/bin/global-witness-capture",
        runner=runner,
        now=now,
        sleep=sleep,
        monotonic=lambda: next(monotonic_instants),
    )

    payload = json.loads(watch_index_path.read_text(encoding="utf-8"))
    assert runner_calls == []
    assert payload["capture_count"] == 0
    assert payload["skipped_trigger_count"] == 1
    assert payload["skipped_triggers"][0]["reason"] == "stale_trigger_capture_suppressed"
    assert payload["skipped_triggers"][0]["trigger_lag_seconds"] == 4.0


def test_build_evidence_split_keeps_witness_and_lifecycle_roles_separate():
    split = build_evidence_split(
        manifest_loaded=True,
        frame_count=4,
        trace_event_count=7,
        capture_profile="stress",
    )

    assert split["visual_witness"]["role"] == "perturbing_visual_stress_witness"
    assert split["visual_witness"]["can_prove_absence"] is False
    assert split["visual_witness"]["can_raise_candidate_bad_frame"] is True
    assert split["visual_witness"]["capture_profile"] == "stress"
    assert split["visual_witness"]["known_capture_artifact_signatures"][0]["signature"] == (
        "horizontal_tear_or_phase_split"
    )
    assert split["lifecycle_trace"]["role"] == "generation_lifecycle_receipts"
    assert split["lifecycle_trace"]["required_for_extraction_clearance"] is True
    assert split["classification_rule"]["witness_clean"] == "not_clearance"
    assert split["classification_rule"]["trace_unlawful_publication"] == "primitive_lifecycle_blocker"
    assert split["classification_rule"]["known_capture_artifact_without_trace_violation"] == (
        "not_a_primitive_blocker_by_itself"
    )


def test_write_witness_index_links_manifest_and_trace_events(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"frames": [{"path": "a.png"}, {"path": "b.png"}]}),
        encoding="utf-8",
    )

    index_path = write_witness_index(
        output_dir=tmp_path,
        trace_path=tmp_path / "trace.jsonl",
        started_at=datetime(2026, 5, 22, 0, 0, 1, tzinfo=timezone.utc),
        ended_at=datetime(2026, 5, 22, 0, 0, 2, tzinfo=timezone.utc),
        command=["uv", "run", "perceptasia-screen-capture"],
        trace_events=[{"event": "overlay.show.begin"}],
        capture_profile="low_perturbation",
    )

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "spoke.retina_lasso_trace_witness.v1"
    assert payload["frame_count"] == 2
    assert payload["trace_event_count"] == 1
    assert payload["trace_events"][0]["event"] == "overlay.show.begin"
    assert payload["evidence_split"]["visual_witness"]["frame_count"] == 2
    assert payload["capture_profile"] == "low_perturbation"
    assert payload["evidence_split"]["visual_witness"]["capture_profile"] == "low_perturbation"
    assert payload["evidence_split"]["lifecycle_trace"]["trace_event_count"] == 1
    assert payload["evidence_split"]["classification_rule"]["witness_bad_frame"] == (
        "candidate_violation_until_trace_correlated"
    )


def test_autonomous_retarget_witness_starts_capture_before_stimulus(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"timestamp": "2026-05-22T00:00:01Z", "event": "overlay.cancel_dismiss.begin"}),
                json.dumps(
                    {
                        "timestamp": "2026-05-22T00:00:01.050Z",
                        "event": "overlay.show.retarget_dismiss_to_summon",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "capture"
    calls = []
    instants = iter(
        [
            datetime(2026, 5, 22, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 5, 22, 0, 0, 2, tzinfo=timezone.utc),
        ]
    )

    class Capture:
        def wait(self):
            calls.append("capture_wait")
            return 0

        def terminate(self):
            calls.append("capture_terminate")

    def popen(command, cwd=None):
        calls.append("capture_start")
        out = command[command.index("--output-dir") + 1]
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "manifest.json").write_text(
            json.dumps({"frames": [{"path": "frame-000.png"}]}),
            encoding="utf-8",
        )
        return Capture()

    def retarget_driver(**kwargs):
        calls.append(("stimulus_start", kwargs["repeats"]))
        return [{"cycle": 1, "phase": "open", "status": "ready", "presentation_generation": 7}]

    index_path = run_autonomous_hammer_witness(
        trace_path=trace_path,
        output_dir=output_dir,
        repo_root=tmp_path,
        duration_seconds=2.0,
        fps=4.0,
        capture_profile="stress",
        retarget_during_dismiss_repeats=2,
        pre_hammer_delay_seconds=0.25,
        capture_command="/usr/local/bin/global-witness-capture",
        popen=popen,
        now=lambda: next(instants),
        sleep=lambda seconds: calls.append(("sleep", seconds)),
        retarget_driver=retarget_driver,
    )

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert calls[:3] == ["capture_start", ("sleep", 0.25), ("stimulus_start", 2)]
    assert "capture_wait" in calls
    assert payload["stimulus"]["mode"] == "retarget-during-dismiss"
    assert payload["stimulus"]["repeats"] == 2
    assert payload["stimulus"]["open_ready_gate"] == (
        "fresh overlay.show.begin followed by overlay.visual_ready.push "
        "with matching generation, visible text, and body_ready"
    )
    assert payload["stimulus"]["retarget_gate_results"] == [
        {"cycle": 1, "phase": "open", "status": "ready", "presentation_generation": 7}
    ]
    assert payload["frame_count"] == 1
    assert [event["event"] for event in payload["trace_events"]] == [
        "overlay.cancel_dismiss.begin",
        "overlay.show.retarget_dismiss_to_summon",
    ]

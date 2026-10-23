import importlib

import pytest
from unittest.mock import MagicMock


def transition_api():
    lifecycle = importlib.import_module("spoke.optical_lifecycle")
    assert callable(getattr(lifecycle, "HouseTransition", None)), "House transition adapter missing"
    return lifecycle.HouseTransition


def test_shared_transition_uses_house_choreography_and_consumer_namespace():
    transition = transition_api()()
    house = importlib.import_module("spoke.house_optical_primitive")
    final = house.compile_house_optical_shell_config(720, 560)
    final.update(client_id="spoke.teleporter", role="hud", z_index=43)
    transition.summon(10)
    frame = transition.frame(final, 10 + house.OPTICAL_MATERIALIZATION_S * .62)
    assert frame.progress == pytest.approx(.62)
    assert frame.configs["spoke.teleporter"] == house.materialized_house_optical_shell_config(final, frame.progress)
    assert frame.body == house.house_materialization_fill_state(frame.progress)
    assert frame.text["alpha"] == float(house.optical_entrance_text_ready(frame.progress))
    transition.frame(final, 11)
    transition.rest()
    transition.dismiss(11)
    frame = transition.frame(final, 11 + house.OPTICAL_MATERIALIZATION_DISMISS_S * .7)
    assert frame.phase == "dismiss"
    assert frame.progress == pytest.approx(.3)
    assert frame.body == house.house_dismiss_materialization_fill_state(frame.progress)
    assert frame.text == house.house_dismiss_text_collapse_state(frame.progress)
    seam = frame.configs["spoke.teleporter.dismiss_seam"]
    expected = house.dismiss_seam_latch_house_shell_config(final, frame.progress)
    for key in ("warp_mode", "scar_amount", "content_height_points"):
        assert seam[key] == expected[key]
    assert seam["client_id"] == "spoke.teleporter.dismiss_seam"
    assert seam["role"] == "hud"
    assert seam["z_index"] == 42
    radial = frame.configs["spoke.teleporter.dismiss_radial_pucker"]
    assert radial["warp_mode"] == 2
    assert radial["visible"] is True
    assert final["client_id"] == "spoke.teleporter"


def test_tail_clock_consumes_elapsed_time_without_adding_a_frame_delay():
    transition = transition_api()()
    house = importlib.import_module("spoke.house_optical_primitive")
    final = dict(house.compile_house_optical_shell_config(720, 560), client_id="spoke.teleporter")
    transition.summon(0)
    transition.frame(final, 2)
    transition.dismiss(2)
    tail_start = 2 + house.OPTICAL_MATERIALIZATION_DISMISS_S
    frame = transition.frame(final, tail_start + .1)
    assert frame.phase == "tail"
    assert frame.tail_progress == pytest.approx(house.OPTICAL_MATERIALIZATION_PUCKER_PREARM_TAIL_PROGRESS + .1 / house.OPTICAL_MATERIALIZATION_PUCKER_TAIL_S)
    assert frame.configs["spoke.teleporter"]["visible"] is False
    assert frame.text["alpha"] == 0
    frame = transition.frame(final, tail_start + 1)
    assert frame.phase == "hidden"
    assert all(not config["visible"] for config in frame.configs.values())


def test_reversal_uses_landed_slit_reentry_and_never_releases_text_early():
    transition = transition_api()()
    house = importlib.import_module("spoke.house_optical_primitive")
    final = dict(house.compile_house_optical_shell_config(720, 560), client_id="spoke.teleporter")
    transition.summon(0)
    transition.frame(final, 2)
    transition.dismiss(2)
    transition.summon(2 + house.OPTICAL_MATERIALIZATION_DISMISS_S * .2)
    frame = transition.frame(final, 2 + house.OPTICAL_MATERIALIZATION_DISMISS_S * .2)
    assert frame.progress == pytest.approx(.24)
    assert frame.text["alpha"] == 0
    assert not frame.configs["spoke.teleporter.dismiss_seam"]["visible"]


def test_dismiss_mid_open_starts_at_current_progress_not_full_body():
    transition = transition_api()()
    house = importlib.import_module("spoke.house_optical_primitive")
    final = dict(house.compile_house_optical_shell_config(720, 560), client_id="spoke.teleporter")
    transition.summon(0)
    transition.dismiss(house.OPTICAL_MATERIALIZATION_S * .2)
    frame = transition.frame(final, house.OPTICAL_MATERIALIZATION_S * .2)
    assert frame.progress == pytest.approx(.2)
    assert frame.text["alpha"] == 0


def test_interrupting_quarantined_late_summon_never_reveals_foreground():
    transition = transition_api()()
    house = importlib.import_module("spoke.house_optical_primitive")
    final = dict(house.compile_house_optical_shell_config(720, 560), client_id="spoke.teleporter")
    transition.summon(10)
    now = 10 + .8 * house.OPTICAL_MATERIALIZATION_S
    before = transition.frame(final, now)
    assert before.text["alpha"] == 0
    transition.dismiss(now)
    for elapsed in (0, .02, .1):
        after = transition.frame(final, now + elapsed)
        assert after.text["alpha"] <= before.text["alpha"]
        assert after.text["height_frac"] <= before.text["height_frac"]


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("progress", [.1, .3, .55, .9])
def test_transition_matches_actual_assistant_driver_publications(mock_pyobjc, monkeypatch, direction, progress):
    from tests.test_command_overlay import _make_overlay

    monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
    overlay, command = _make_overlay(mock_pyobjc)
    final = dict(command._command_optical_shell_config(720, 560),
                 client_id="assistant.command", role="assistant", z_index=11)
    main, seam, radial = object(), object(), object()
    overlay._fullscreen_compositor = main
    overlay._ensure_dismiss_seam_compositor = MagicMock(return_value=seam)
    overlay._ensure_dismiss_radial_pucker_compositor = MagicMock(return_value=radial)
    overlay._stop_dismiss_seam_compositor = MagicMock()
    overlay._materialization_final_shell_config = final
    overlay._materialization_direction = direction
    overlay._materialization_started_at = 0
    overlay._materialization_progress = 0 if direction > 0 else 1
    overlay._materialization_timer = object()
    overlay._check_optical_entrance_readiness = MagicMock()
    overlay._set_fullscreen_compositor_human_visible = MagicMock()
    published = {}

    def publish(updates):
        published.update(updates)
        return True

    overlay._publish_shared_compositor_configs = publish
    transition = transition_api()()
    if direction > 0:
        transition.summon(0)
        elapsed = progress * command._OPTICAL_MATERIALIZATION_S
    else:
        transition.rest()
        transition.dismiss(0)
        elapsed = (1 - progress) * command._OPTICAL_MATERIALIZATION_DISMISS_S
    monkeypatch.setattr(command.time, "perf_counter", lambda: elapsed)
    overlay.materializationStep_(overlay._materialization_timer)
    frame = transition.frame(final, elapsed)
    assert published[main] == pytest.approx(frame.configs["assistant.command"])
    for handle, suffix in ((seam, "dismiss_seam"), (radial, "dismiss_radial_pucker")):
        config = frame.configs[f"assistant.command.{suffix}"]
        assert (handle in published) is config["visible"]
        if config["visible"]:
            assert published[handle] == pytest.approx(config)

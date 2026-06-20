from __future__ import annotations

import importlib

import pytest


def _fresh_modules(monkeypatch):
    monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
    monkeypatch.setenv("SPOKE_COMMAND_GPU_MATERIAL_ENABLED", "1")
    import sys

    sys.modules.pop("spoke.command_overlay", None)
    sys.modules.pop("spoke.house_optical_primitive", None)
    return (
        importlib.import_module("spoke.command_overlay"),
        importlib.import_module("spoke.house_optical_primitive"),
    )


def test_house_optical_primitive_exports_assistant_parity_compiler(mock_pyobjc, monkeypatch):
    command_overlay, primitive = _fresh_modules(monkeypatch)

    base = command_overlay._command_optical_shell_config(600.0, 80.0)
    exported_base = primitive.compile_house_optical_shell_config(600.0, 80.0)

    assert exported_base == base
    for progress in (0.0, 0.20, 0.62, 0.90, 1.0):
        assert primitive.materialized_house_optical_shell_config(
            exported_base, progress
        ) == command_overlay._materialized_optical_shell_config(base, progress)

    for progress in (0.0, 0.35, 0.55, 0.90, 1.0):
        assert primitive.house_materialization_fill_state(
            progress
        ) == command_overlay._materialization_fill_state(progress)
        assert primitive.house_dismiss_materialization_fill_state(
            progress
        ) == command_overlay._dismiss_materialization_fill_state(progress)
        assert primitive.house_dismiss_text_collapse_state(
            progress
        ) == command_overlay._dismiss_text_collapse_state(progress)


def test_house_optical_primitive_exports_assistant_dismiss_sidecars(mock_pyobjc, monkeypatch):
    command_overlay, primitive = _fresh_modules(monkeypatch)

    base = command_overlay._command_optical_shell_config(600.0, 80.0)
    exported_base = primitive.compile_house_optical_shell_config(600.0, 80.0)

    for progress in (0.0, 0.20, 0.42, 0.72, 1.0):
        assert primitive.dismiss_seam_latch_house_shell_config(
            exported_base, progress
        ) == command_overlay._dismiss_seam_latch_shell_config(base, progress)
        assert primitive.dismiss_radial_pucker_house_shell_config(
            exported_base, progress
        ) == command_overlay._dismiss_radial_pucker_shell_config(base, progress)

    assert primitive.hidden_dismiss_main_house_shell_config(
        exported_base
    ) == command_overlay._hidden_dismiss_main_shell_config(base)


def test_command_overlay_delegates_to_extracted_house_primitive(mock_pyobjc, monkeypatch):
    command_overlay, _primitive = _fresh_modules(monkeypatch)

    sentinel = {"enabled": True, "content_width_points": 12.0}

    def fake_compile(width=None, height=None):
        assert width == 600.0
        assert height == 80.0
        return sentinel

    monkeypatch.setattr(
        command_overlay,
        "_house_compile_optical_shell_config",
        fake_compile,
    )
    assert command_overlay._command_optical_shell_config(600.0, 80.0) is sentinel

    materialized = {"materialized": True}

    def fake_materialized(config, progress):
        assert config is sentinel
        assert progress == pytest.approx(0.42)
        return materialized

    monkeypatch.setattr(
        command_overlay,
        "_house_materialized_optical_shell_config",
        fake_materialized,
    )
    assert command_overlay._materialized_optical_shell_config(sentinel, 0.42) is materialized

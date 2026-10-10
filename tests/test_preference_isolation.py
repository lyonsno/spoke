"""The test suite must never use the operator's persistent preferences."""

import json
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("iteration", [0, 1])
def test_model_preferences_are_isolated_per_test(tmp_path, iteration):
    configured = os.environ.get("SPOKE_MODEL_PREFERENCES_PATH")
    assert configured, "Tests must redirect Spoke's production model preferences"
    path = Path(configured)
    assert path.parent == tmp_path
    assert not path.exists(), "Each test must start with independent preferences"
    path.write_text(str(iteration))


def test_saved_fallback_stays_in_test_preferences(main_module, tmp_path, monkeypatch):
    delegate = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
    preview = "mlx-community/whisper-small.en-mlx"
    model = "mlx-community/whisper-medium.en-mlx"
    assert delegate._save_model_preferences(preview, model)
    delegate._load_model_preferences = lambda: {
        "preview_model": preview,
        "transcription_model": main_module._NEMOTRON_CPU_MODEL_ID,
    }
    for key in ("SPOKE_PREVIEW_MODEL", "SPOKE_TRANSCRIPTION_MODEL", "SPOKE_WHISPER_MODEL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        main_module.NemotronCPUClient, "availability_error",
        lambda **kwargs: "missing test model",
    )
    resolved = delegate._resolve_model_ids()
    payload = json.loads((tmp_path / "model_preferences.json").read_text())
    assert payload["transcription_model"] == resolved[1]
    assert resolved[1] == "mlx-community/whisper-medium.en-mlx-8bit"

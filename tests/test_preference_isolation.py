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


def test_real_model_preference_writes_stay_in_test_storage(main_module, tmp_path):
    delegate = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
    path = tmp_path / "model_preferences.json"
    assert delegate._preferences_path() == path
    preview = "mlx-community/whisper-small.en-mlx"
    model = "mlx-community/whisper-medium.en-mlx"
    assert delegate._save_model_preferences(preview, model)
    payload = json.loads(path.read_text())
    expected = {"preview_model": preview, "transcription_model": model}
    assert payload == expected
    assert delegate._load_model_preferences() == expected

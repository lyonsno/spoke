"""Contract tests for the dev launcher script."""

from pathlib import Path


def test_launch_script_preserves_startup_logs():
    """The launcher should route the actual background process output into the durable log."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "launch-dev.sh"
    text = script.read_text()

    assert "spoke-dev-launch.log" in text
    assert ">/dev/null 2>&1" not in text
    assert '</dev/null >>"$LOG_FILE" 2>&1 &' in text


def test_launch_script_preserves_dual_model_defaults():
    """The launcher should not collapse preview/final back to one legacy model."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "launch-dev.sh"
    text = script.read_text()

    assert 'SPOKE_PREVIEW_MODEL="${SPOKE_PREVIEW_MODEL:-mlx-community/whisper-medium.en-mlx-8bit}"' in text
    assert 'SPOKE_TRANSCRIPTION_MODEL="${SPOKE_TRANSCRIPTION_MODEL:-mlx-community/whisper-large-v3-turbo}"' in text
    assert 'SPOKE_WHISPER_MODEL="${SPOKE_WHISPER_MODEL:-mlx-community/whisper-medium.en-mlx-8bit}"' not in text


def test_launch_script_avoids_nohup_detach():
    """The launcher should use the plain background-launch form that survived manual smoke."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "launch-dev.sh"
    text = script.read_text()

    assert "nohup " not in text
    assert 'env -C "$REPO_ROOT"' in text
    assert text.rstrip().endswith("&")

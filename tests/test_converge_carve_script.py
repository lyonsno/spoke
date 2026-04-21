"""Tests for converge-carve.py history loading with the new format."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

# Import the carve script as a module
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
import importlib

_carve = importlib.import_module("converge-carve")


class TestLoadHistoryUtterances:
    def test_new_format_message_chains(self, tmp_path):
        history = [
            [
                {"role": "user", "content": "what time is it"},
                {"role": "assistant", "content": "It's 3pm."},
            ],
            [
                {"role": "user", "content": "read the screen"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "tc1", "function": {"name": "capture_context"}}]},
                {"role": "tool", "tool_call_id": "tc1", "content": '{"scene_ref":"s1"}'},
                {"role": "assistant", "content": "I see your terminal."},
            ],
        ]
        history_path = tmp_path / "history.json"
        history_path.write_text(json.dumps(history))

        with patch.object(_carve, "_HISTORY_PATH", history_path):
            pairs = _carve._load_history_utterances()

        assert len(pairs) == 2
        assert pairs[0] == {"user": "what time is it", "assistant": "It's 3pm."}
        # Tool-using turn: first user msg + first non-null assistant content
        assert pairs[1]["user"] == "read the screen"
        assert pairs[1]["assistant"] == "I see your terminal."

    def test_skips_turns_without_user_message(self, tmp_path):
        history = [
            [
                {"role": "system", "content": "you are helpful"},
                {"role": "assistant", "content": "hello"},
            ],
        ]
        history_path = tmp_path / "history.json"
        history_path.write_text(json.dumps(history))

        with patch.object(_carve, "_HISTORY_PATH", history_path):
            pairs = _carve._load_history_utterances()

        assert pairs == []

    def test_missing_file_returns_empty(self, tmp_path):
        with patch.object(_carve, "_HISTORY_PATH", tmp_path / "missing.json"):
            assert _carve._load_history_utterances() == []

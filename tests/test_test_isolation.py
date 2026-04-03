"""Tests that the test suite itself cannot interfere with a live spoke process.

These tests verify that SPOKE_LOCK_PATH and SPOKE_HEARTBEAT_PATH are redirected
to temp paths so that importing spoke.__main__ or calling _acquire_instance_lock()
from tests cannot SIGTERM a production spoke instance.
"""

import os


class TestRuntimeIsolation:
    def test_lock_path_is_not_production(self, main_module):
        """The lock path used by tests must not be the production path."""
        production_path = os.path.expanduser("~/Library/Logs/.spoke.lock")
        assert main_module._LOCK_PATH != production_path, (
            f"Tests are using the production lock path {production_path}! "
            "This means running pytest can kill a live spoke instance."
        )

    def test_lock_path_is_in_temp(self, main_module):
        """The lock path should be in a temp directory."""
        assert "/tmp/" in main_module._LOCK_PATH or "spoke-test" in main_module._LOCK_PATH

    def test_heartbeat_path_is_not_production(self):
        """The heartbeat path used by tests must not be the production path."""
        from spoke.heartbeat import HEARTBEAT_PATH

        production_path = os.path.expanduser("~/Library/Logs/.spoke-heartbeat.json")
        assert HEARTBEAT_PATH != production_path, (
            f"Tests are using the production heartbeat path {production_path}! "
            "This means running pytest can interfere with a live spoke instance."
        )

    def test_heartbeat_path_is_in_temp(self):
        """The heartbeat path should be in a temp directory."""
        from spoke.heartbeat import HEARTBEAT_PATH

        assert "/tmp/" in HEARTBEAT_PATH or "spoke-test" in HEARTBEAT_PATH

    def test_env_vars_are_set(self):
        """The isolation env vars must be set before any spoke import."""
        assert "SPOKE_LOCK_PATH" in os.environ
        assert "SPOKE_HEARTBEAT_PATH" in os.environ
        assert "spoke-test" in os.environ["SPOKE_LOCK_PATH"]
        assert "spoke-test" in os.environ["SPOKE_HEARTBEAT_PATH"]

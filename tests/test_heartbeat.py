"""Tests for spoke.heartbeat — process heartbeat & model TTL."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

from spoke.heartbeat import (
    DEFAULT_MODEL_TTL_S,
    HEARTBEAT_INTERVAL_S,
    STALE_THRESHOLD_S,
    HeartbeatManager,
    _is_process_alive,
    _remove_heartbeat,
    zombie_sweep,
)


# ── HeartbeatManager ────────────────────────────────────────────


class TestHeartbeatManager:
    def test_write_heartbeat_creates_file(self, tmp_path):
        path = str(tmp_path / "heartbeat.json")
        hb = HeartbeatManager(heartbeat_path=path)
        hb.tick()
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert data["pid"] == os.getpid()
        assert "timestamp" in data
        assert data["models_loaded"] == []

    def test_register_and_unregister_model(self, tmp_path):
        path = str(tmp_path / "heartbeat.json")
        hb = HeartbeatManager(heartbeat_path=path)
        hb.register_model("whisper-base")
        hb.tick()
        data = json.loads(Path(path).read_text())
        assert "whisper-base" in data["models_loaded"]

        hb.unregister_model("whisper-base")
        hb.tick()
        data = json.loads(Path(path).read_text())
        assert "whisper-base" not in data["models_loaded"]

    def test_touch_updates_last_use(self, tmp_path):
        path = str(tmp_path / "heartbeat.json")
        hb = HeartbeatManager(heartbeat_path=path, model_ttl_s=0.1)
        hb.register_model("whisper-base")
        time.sleep(0.15)
        # Without touch, should expire.
        expired = hb.tick()
        assert "whisper-base" in expired

    def test_touch_prevents_expiry(self, tmp_path):
        path = str(tmp_path / "heartbeat.json")
        hb = HeartbeatManager(heartbeat_path=path, model_ttl_s=0.5)
        hb.register_model("whisper-base")
        time.sleep(0.2)
        hb.touch("whisper-base")
        expired = hb.tick()
        assert expired == []

    def test_ttl_expiry_returns_model_ids(self, tmp_path):
        path = str(tmp_path / "heartbeat.json")
        hb = HeartbeatManager(heartbeat_path=path, model_ttl_s=0.05)
        hb.register_model("model-a")
        hb.register_model("model-b")
        time.sleep(0.1)
        expired = hb.tick()
        assert sorted(expired) == ["model-a", "model-b"]

    def test_evict_callback_called_on_expiry(self, tmp_path):
        path = str(tmp_path / "heartbeat.json")
        hb = HeartbeatManager(heartbeat_path=path, model_ttl_s=0.05)
        evicted = []
        hb.set_evict_callback(lambda mid: evicted.append(mid))
        hb.register_model("whisper-base")
        time.sleep(0.1)
        hb.tick()
        assert "whisper-base" in evicted

    def test_evict_callback_exception_does_not_crash(self, tmp_path):
        path = str(tmp_path / "heartbeat.json")
        hb = HeartbeatManager(heartbeat_path=path, model_ttl_s=0.05)
        hb.set_evict_callback(lambda mid: (_ for _ in ()).throw(RuntimeError("boom")))
        hb.register_model("bad-model")
        time.sleep(0.1)
        # Should not raise.
        expired = hb.tick()
        assert "bad-model" in expired

    def test_set_context(self, tmp_path):
        path = str(tmp_path / "heartbeat.json")
        hb = HeartbeatManager(heartbeat_path=path)
        hb.set_context(launch_target="dev", worktree="/tmp/spoke-dev")
        hb.tick()
        data = json.loads(Path(path).read_text())
        assert data["launch_target"] == "dev"
        assert data["worktree"] == "/tmp/spoke-dev"

    def test_remove_deletes_file(self, tmp_path):
        path = str(tmp_path / "heartbeat.json")
        hb = HeartbeatManager(heartbeat_path=path)
        hb.tick()
        assert Path(path).exists()
        hb.remove()
        assert not Path(path).exists()

    def test_remove_no_file_no_error(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        hb = HeartbeatManager(heartbeat_path=path)
        hb.remove()  # Should not raise.

    def test_atomic_write(self, tmp_path):
        """Heartbeat file should not have a .tmp leftover after write."""
        path = str(tmp_path / "heartbeat.json")
        hb = HeartbeatManager(heartbeat_path=path)
        hb.tick()
        assert not Path(path + ".tmp").exists()
        assert Path(path).exists()

    def test_clear_metal_cache_no_mlx(self, tmp_path):
        """clear_metal_cache should be a no-op when MLX is not importable."""
        path = str(tmp_path / "heartbeat.json")
        hb = HeartbeatManager(heartbeat_path=path)
        with mock.patch.dict(sys.modules, {"mlx.core": None, "mlx": None}):
            hb.clear_metal_cache()  # Should not raise.


# ── zombie_sweep ────────────────────────────────────────────────


class TestZombieSweep:
    def test_no_heartbeat_file(self, tmp_path):
        """Should return cleanly when no heartbeat file exists."""
        zombie_sweep(str(tmp_path / "nonexistent.json"))

    def test_dead_process_cleans_up(self, tmp_path):
        path = str(tmp_path / "heartbeat.json")
        # Write a heartbeat for a PID that definitely doesn't exist.
        data = {"pid": 99999999, "timestamp": "2020-01-01T00:00:00+00:00"}
        Path(path).write_text(json.dumps(data))
        zombie_sweep(path)
        assert not Path(path).exists()

    def test_fresh_heartbeat_skips(self, tmp_path):
        path = str(tmp_path / "heartbeat.json")
        from datetime import datetime, timezone

        data = {
            "pid": os.getpid(),  # Our own PID — definitely alive and fresh.
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        Path(path).write_text(json.dumps(data))
        zombie_sweep(path)
        # Should not delete — it's us and it's fresh.
        assert Path(path).exists()

    def test_stale_heartbeat_kills_spoke_process(self, tmp_path):
        """Spawn a subprocess that looks like spoke, write stale heartbeat, verify kill."""
        path = str(tmp_path / "heartbeat.json")
        # Spawn a process whose command line contains '-m spoke' so
        # _is_spoke_process identifies it as a spoke process.
        proc = subprocess.Popen(
            [sys.executable, "-c",
             "import sys; sys.argv = ['python', '-m', 'spoke']; "
             "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        pid = proc.pid
        assert _is_process_alive(pid)

        data = {"pid": pid, "timestamp": "2020-01-01T00:00:00+00:00"}
        Path(path).write_text(json.dumps(data))

        zombie_sweep(path)

        # Process should be dead now.
        proc.wait(timeout=5)
        assert proc.returncode is not None
        assert not Path(path).exists()

    def test_stale_heartbeat_non_spoke_process_not_killed(self, tmp_path):
        """Don't kill a non-spoke process even if heartbeat is stale."""
        path = str(tmp_path / "heartbeat.json")
        # This process has no 'spoke' in its command line.
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        pid = proc.pid

        data = {"pid": pid, "timestamp": "2020-01-01T00:00:00+00:00"}
        Path(path).write_text(json.dumps(data))

        zombie_sweep(path)

        # Process should still be alive — we don't kill non-spoke processes.
        assert _is_process_alive(pid)
        proc.terminate()
        proc.wait(timeout=5)

    def test_invalid_json(self, tmp_path):
        path = str(tmp_path / "heartbeat.json")
        Path(path).write_text("not json at all")
        zombie_sweep(path)  # Should not raise.

    def test_missing_fields(self, tmp_path):
        path = str(tmp_path / "heartbeat.json")
        Path(path).write_text(json.dumps({"unrelated": True}))
        zombie_sweep(path)  # Should not raise.

    def test_stale_zombie_treated_as_dead(self, tmp_path):
        """A zombie process with a stale heartbeat should be treated as dead.

        The sweep should recognize the zombie at the liveness check and
        take the "already dead" path (clean up heartbeat, no SIGTERM),
        rather than falling through to staleness/identity checks.
        """
        import spoke.heartbeat as hb_module

        path = str(tmp_path / "heartbeat.json")
        data = {"pid": 77777, "timestamp": "2020-01-01T00:00:00+00:00"}
        Path(path).write_text(json.dumps(data))

        kill_calls = []

        def fake_kill(pid, sig_num):
            kill_calls.append((pid, sig_num))
            if sig_num == 0:
                return  # kill(pid, 0) succeeds — kernel sees the zombie

        # ps reports zombie state for the liveness check, and a spoke-like
        # command for the identity check.  If the sweep calls _is_spoke_process
        # at all, that means it didn't detect the zombie early enough.
        is_spoke_called = []
        original_is_spoke = hb_module._is_spoke_process

        def spy_is_spoke(pid):
            is_spoke_called.append(pid)
            return True  # pretend it's spoke — if reached, the sweep will SIGTERM

        with mock.patch.object(hb_module.os, "kill", side_effect=fake_kill):
            with mock.patch.object(hb_module, "_is_spoke_process", side_effect=spy_is_spoke):
                with mock.patch("spoke.heartbeat.subprocess.run", return_value=mock.MagicMock(returncode=0, stdout="Z\n")):
                    zombie_sweep(path)

        # Should have cleaned up the heartbeat.
        assert not Path(path).exists()
        # The zombie should have been caught at the liveness check, before
        # the sweep ever reached the identity or staleness checks.
        assert is_spoke_called == [], (
            "_is_spoke_process should not be called for a zombie — "
            "the liveness check should have caught it"
        )
        assert (77777, signal.SIGTERM) not in kill_calls


# ── _is_process_alive ──────────────────────────────────────────


class TestIsProcessAlive:
    def test_current_process(self):
        assert _is_process_alive(os.getpid())

    def test_nonexistent_pid(self):
        assert not _is_process_alive(99999999)

    def test_generic_ps_lookup_error_preserves_successful_kill_probe(self):
        with mock.patch("spoke.heartbeat.os.kill") as mock_kill:
            with mock.patch(
                "spoke.heartbeat.subprocess.run",
                side_effect=OSError("ps failed"),
            ):
                assert _is_process_alive(77777)

        mock_kill.assert_called_once_with(77777, 0)


# ── Client unload methods ──────────────────────────────────────


class TestClientUnload:
    def test_local_transcription_unload(self):
        from spoke.transcribe_local import LocalTranscriptionClient

        client = LocalTranscriptionClient(model="mlx-community/whisper-tiny.en-mlx")
        assert not client.is_loaded
        # Simulate a load by setting internal state.
        client._loaded = True
        client._model_instance = object()
        assert client.is_loaded
        client.unload()
        assert not client.is_loaded
        assert client._model_instance is None

    def test_local_transcription_unload_idempotent(self):
        from spoke.transcribe_local import LocalTranscriptionClient

        client = LocalTranscriptionClient(model="mlx-community/whisper-tiny.en-mlx")
        client.unload()  # Should not raise when not loaded.
        assert not client.is_loaded

    def test_qwen_unload(self):
        from spoke.transcribe_qwen import LocalQwenClient

        client = LocalQwenClient(model="Qwen/test")
        assert not client.is_loaded
        client._session = object()
        assert client.is_loaded
        client.unload()
        assert not client.is_loaded
        assert client._session is None

    def test_tts_unload(self):
        from spoke.tts import TTSClient

        client = TTSClient.__new__(TTSClient)
        client._model = None
        client._model_id = "test-model"
        assert not client.is_loaded
        client._model = object()
        assert client.is_loaded
        client.unload()
        assert not client.is_loaded
        assert client._model is None

    def test_tts_unload_idempotent(self):
        from spoke.tts import TTSClient

        client = TTSClient.__new__(TTSClient)
        client._model = None
        client._model_id = "test-model"
        client.unload()  # Should not raise when not loaded.


# ── Integration: eviction round-trip ───────────────────────────


class TestEvictionRoundTrip:
    def test_heartbeat_evicts_and_model_reloads(self, tmp_path):
        """Simulate the full cycle: register, expire, evict, re-prepare."""
        from spoke.transcribe_local import LocalTranscriptionClient

        path = str(tmp_path / "heartbeat.json")
        hb = HeartbeatManager(heartbeat_path=path, model_ttl_s=0.05)

        client = LocalTranscriptionClient(model="mlx-community/whisper-tiny.en-mlx")
        # Simulate loaded state.
        client._loaded = True
        client._model_instance = object()

        evicted_models = []

        def evict_cb(model_id):
            client.unload()
            hb.unregister_model(model_id)
            evicted_models.append(model_id)

        hb.set_evict_callback(evict_cb)
        hb.register_model("mlx-community/whisper-tiny.en-mlx")
        time.sleep(0.1)
        expired = hb.tick()

        assert "mlx-community/whisper-tiny.en-mlx" in expired
        assert "mlx-community/whisper-tiny.en-mlx" in evicted_models
        assert not client.is_loaded
        # Heartbeat should show no models loaded after eviction.
        data = json.loads(Path(path).read_text())
        assert data["models_loaded"] == []

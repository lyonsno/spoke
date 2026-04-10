from unittest.mock import MagicMock

from spoke.wakeword import WakeWordListener


class TestWakeWordListenerStop:
    def test_stop_aborts_and_closes_stream_before_join(self):
        listener = WakeWordListener(access_key="test", keywords=["computer"])
        events: list[str] = []
        stream = MagicMock()
        stream.abort.side_effect = lambda: events.append("abort")
        stream.close.side_effect = lambda: events.append("close")
        thread = MagicMock()
        thread.join.side_effect = lambda timeout=None: events.append("join")
        thread.is_alive.return_value = False
        porcupine = MagicMock()
        listener._running = True
        listener._stream = stream
        listener._thread = thread
        listener._porcupine = porcupine

        listener.stop()

        assert events[:3] == ["abort", "close", "join"]
        thread.join.assert_called_once_with(timeout=2.0)
        porcupine.delete.assert_called_once_with()
        assert listener._running is False
        assert listener._stream is None
        assert listener._thread is None
        assert listener._porcupine is None

    def test_stop_still_joins_and_cleans_up_if_stream_abort_fails(self):
        listener = WakeWordListener(access_key="test", keywords=["computer"])
        stream = MagicMock()
        stream.abort.side_effect = RuntimeError("abort failed")
        stream.close.side_effect = RuntimeError("close failed")
        thread = MagicMock()
        thread.is_alive.return_value = False
        porcupine = MagicMock()
        listener._running = True
        listener._stream = stream
        listener._thread = thread
        listener._porcupine = porcupine

        listener.stop()

        thread.join.assert_called_once_with(timeout=2.0)
        porcupine.delete.assert_called_once_with()
        assert listener._running is False
        assert listener._stream is None
        assert listener._thread is None
        assert listener._porcupine is None

from unittest.mock import MagicMock

from spoke.handsfree import HandsFreeController, HandsFreeState, handsfree_env_ready


class TestHandsFreeControllerWakeWords:
    def test_handsfree_env_ready_accepts_openwakeword_models(self, monkeypatch):
        monkeypatch.delenv("SPOKE_PICOVOICE_PORCUPINE_ACCESS_KEY", raising=False)
        monkeypatch.setenv("SPOKE_WAKEWORD_BACKEND", "openwakeword")
        monkeypatch.setenv("SPOKE_WAKEWORD_LISTEN_MODEL", "/tmp/listen.tflite")
        monkeypatch.setenv("SPOKE_WAKEWORD_SLEEP_MODEL", "/tmp/sleep.tflite")

        assert handsfree_env_ready() is True

    def test_enable_uses_openwakeword_models_when_backend_selected(self, monkeypatch):
        created = {}

        class FakeWakeWordListener:
            def __init__(self, **kwargs):
                created.update(kwargs)

            def start(self):
                return None

        monkeypatch.setenv("SPOKE_WAKEWORD_BACKEND", "openwakeword")
        monkeypatch.setenv("SPOKE_WAKEWORD_LISTEN_MODEL", "/tmp/listen.tflite")
        monkeypatch.setenv("SPOKE_WAKEWORD_SLEEP_MODEL", "/tmp/sleep.tflite")
        monkeypatch.setenv("SPOKE_PICOVOICE_PORCUPINE_ACCESS_KEY", "")
        monkeypatch.setattr("spoke.wakeword.WakeWordListener", FakeWakeWordListener)

        controller = HandsFreeController(delegate=MagicMock())

        controller.enable()

        assert created["backend"] == "openwakeword"
        assert created["model_paths"] == ["/tmp/listen.tflite", "/tmp/sleep.tflite"]
        assert controller.state == HandsFreeState.LISTENING

    def test_sleep_wake_word_keeps_listener_active_while_listening(self):
        controller = HandsFreeController(delegate=MagicMock())
        controller._state = HandsFreeState.LISTENING
        controller.disable = MagicMock()
        controller._stop_dictating = MagicMock()

        controller.handle_wake_word("sleep")

        controller.disable.assert_not_called()
        controller._stop_dictating.assert_not_called()
        assert controller.state == HandsFreeState.LISTENING

    def test_sleep_wake_word_returns_dictating_to_listener(self):
        controller = HandsFreeController(delegate=MagicMock())
        controller._state = HandsFreeState.DICTATING
        controller.disable = MagicMock()
        controller._stop_dictating = MagicMock(
            side_effect=lambda: controller._set_state(HandsFreeState.LISTENING)
        )

        controller.handle_wake_word("sleep")

        controller.disable.assert_not_called()
        controller._stop_dictating.assert_called_once_with()
        assert controller.state == HandsFreeState.LISTENING

    def test_sleep_wake_word_returns_transcribing_to_listener(self):
        controller = HandsFreeController(delegate=MagicMock())
        controller._state = HandsFreeState.TRANSCRIBING
        controller.disable = MagicMock()
        controller._stop_dictating = MagicMock(
            side_effect=lambda: controller._set_state(HandsFreeState.LISTENING)
        )

        controller.handle_wake_word("sleep")

        controller.disable.assert_not_called()
        controller._stop_dictating.assert_called_once_with()
        assert controller.state == HandsFreeState.LISTENING

    def test_segment_transcription_of_sleep_word_routes_to_wake_handler(self, monkeypatch):
        class ImmediateThread:
            def __init__(self, target=None, args=(), kwargs=None, **_ignored):
                self._target = target
                self._args = args
                self._kwargs = kwargs or {}

            def start(self):
                if self._target is not None:
                    self._target(*self._args, **self._kwargs)

        delegate = MagicMock()
        delegate._client = MagicMock(transcribe=MagicMock(return_value="Terminator"))
        controller = HandsFreeController(delegate=delegate)
        controller._state = HandsFreeState.DICTATING
        monkeypatch.setattr("spoke.handsfree.threading.Thread", ImmediateThread)

        controller._on_segment(b"fake-audio")

        delegate.performSelectorOnMainThread_withObject_waitUntilDone_.assert_called_once_with(
            "handleWakeWord:", {"role": "sleep"}, False
        )

    def test_segment_transcription_of_sleep_word_ignores_case_and_punctuation(
        self, monkeypatch
    ):
        class ImmediateThread:
            def __init__(self, target=None, args=(), kwargs=None, **_ignored):
                self._target = target
                self._args = args
                self._kwargs = kwargs or {}

            def start(self):
                if self._target is not None:
                    self._target(*self._args, **self._kwargs)

        delegate = MagicMock()
        delegate._client = MagicMock(transcribe=MagicMock(return_value="terminator."))
        controller = HandsFreeController(delegate=delegate)
        controller._state = HandsFreeState.DICTATING
        monkeypatch.setattr("spoke.handsfree.threading.Thread", ImmediateThread)

        controller._on_segment(b"fake-audio")

        delegate.performSelectorOnMainThread_withObject_waitUntilDone_.assert_called_once_with(
            "handleWakeWord:", {"role": "sleep"}, False
        )

    def test_segment_transcription_of_repeated_sleep_word_routes_to_wake_handler(
        self, monkeypatch
    ):
        class ImmediateThread:
            def __init__(self, target=None, args=(), kwargs=None, **_ignored):
                self._target = target
                self._args = args
                self._kwargs = kwargs or {}

            def start(self):
                if self._target is not None:
                    self._target(*self._args, **self._kwargs)

        delegate = MagicMock()
        delegate._client = MagicMock(
            transcribe=MagicMock(return_value="Terminator, terminator!")
        )
        controller = HandsFreeController(delegate=delegate)
        controller._state = HandsFreeState.DICTATING
        monkeypatch.setattr("spoke.handsfree.threading.Thread", ImmediateThread)

        controller._on_segment(b"fake-audio")

        delegate.performSelectorOnMainThread_withObject_waitUntilDone_.assert_called_once_with(
            "handleWakeWord:", {"role": "sleep"}, False
        )

    def test_segment_transcription_of_phrase_containing_sleep_word_still_injects_text(
        self, monkeypatch
    ):
        class ImmediateThread:
            def __init__(self, target=None, args=(), kwargs=None, **_ignored):
                self._target = target
                self._args = args
                self._kwargs = kwargs or {}

            def start(self):
                if self._target is not None:
                    self._target(*self._args, **self._kwargs)

        delegate = MagicMock()
        delegate._client = MagicMock(
            transcribe=MagicMock(return_value="the terminator word is weird")
        )
        controller = HandsFreeController(delegate=delegate)
        controller._state = HandsFreeState.DICTATING
        monkeypatch.setattr("spoke.handsfree.threading.Thread", ImmediateThread)

        controller._on_segment(b"fake-audio")

        delegate.performSelectorOnMainThread_withObject_waitUntilDone_.assert_called_once_with(
            "handsFreeInject:",
            {"text": "the terminator word is weird", "dest": "cursor"},
            False,
        )

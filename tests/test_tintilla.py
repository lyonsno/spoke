"""Tests for the Vision Quest Tintilla control surface."""

from unittest.mock import MagicMock


class TestLayerVisibilityState:
    def test_defaults_all_layers_on_and_notifies_on_change(self, mock_pyobjc):
        from spoke.tintilla import (
            COMMAND_FILL_LAYER_ID,
            LayerVisibilityState,
            PREVIEW_FILL_LAYER_ID,
            SCREEN_GLOW_WIDE_BLOOM_LAYER_ID,
        )

        state = LayerVisibilityState()
        listener = MagicMock()
        state.add_listener(listener)

        assert state.is_visible(SCREEN_GLOW_WIDE_BLOOM_LAYER_ID) is True
        assert state.is_visible(PREVIEW_FILL_LAYER_ID) is True
        assert state.is_visible(COMMAND_FILL_LAYER_ID) is True

        state.set_enabled(SCREEN_GLOW_WIDE_BLOOM_LAYER_ID, False)

        assert state.is_visible(SCREEN_GLOW_WIDE_BLOOM_LAYER_ID) is False
        listener.assert_called_once_with(state)

    def test_set_all_enabled_restores_visibility(self, mock_pyobjc):
        from spoke.tintilla import (
            LayerVisibilityState,
            SCREEN_GLOW_CORE_LAYER_ID,
            SCREEN_VIGNETTE_TAIL_LAYER_ID,
        )

        state = LayerVisibilityState()
        state.set_enabled(SCREEN_GLOW_CORE_LAYER_ID, False)
        state.set_enabled(SCREEN_VIGNETTE_TAIL_LAYER_ID, False)

        state.set_all_enabled(True)

        assert state.is_visible(SCREEN_GLOW_CORE_LAYER_ID) is True
        assert state.is_visible(SCREEN_VIGNETTE_TAIL_LAYER_ID) is True


class TestTintillaPanelController:
    def test_show_activates_app_and_brings_panel_forward(self, mock_pyobjc):
        import spoke.tintilla as tintilla_module
        from spoke.tintilla import LayerVisibilityState, TintillaPanelController

        controller = TintillaPanelController.alloc().initWithState_(LayerVisibilityState())
        panel = MagicMock()
        controller._panel = panel
        controller._refresh_controls = MagicMock()

        controller.show()

        tintilla_module.NSApp.activateIgnoringOtherApps_.assert_called_once_with(True)
        panel.makeKeyAndOrderFront_.assert_called_once_with(None)

"""Tests for the Vision Quest Tintilla control surface."""

from unittest.mock import MagicMock


class TestLayerVisibilityState:
    def test_defaults_all_layers_on_and_notifies_on_change(self, mock_pyobjc):
        from spoke.tintilla import (
            ADDITIVE_CURVE_MODE_EXPONENTIAL,
            ADDITIVE_MASK_INTENSITY_LEVELS,
            COMMAND_FILL_LAYER_ID,
            LayerVisibilityState,
            PREVIEW_FILL_LAYER_ID,
            SCREEN_DIMMER_LAYER_ID,
            SCREEN_GLOW_WIDE_BLOOM_LAYER_ID,
            WIDE_BLOOM_PROFILE_QUEST,
        )

        state = LayerVisibilityState()
        listener = MagicMock()
        state.add_listener(listener)

        assert state.is_visible(SCREEN_GLOW_WIDE_BLOOM_LAYER_ID) is True
        assert state.is_visible(SCREEN_DIMMER_LAYER_ID) is True
        assert state.is_visible(PREVIEW_FILL_LAYER_ID) is True
        assert state.is_visible(COMMAND_FILL_LAYER_ID) is True
        assert state.additive_curve_mode() == ADDITIVE_CURVE_MODE_EXPONENTIAL
        assert state.additive_mask_intensity() == ADDITIVE_MASK_INTENSITY_LEVELS[0]
        assert state.wide_bloom_profile() == WIDE_BLOOM_PROFILE_QUEST
        assert state.vignette_curve_mode() == ADDITIVE_CURVE_MODE_EXPONENTIAL
        assert state.vignette_mask_intensity() == ADDITIVE_MASK_INTENSITY_LEVELS[0]
        assert state.vignette_profile() == WIDE_BLOOM_PROFILE_QUEST

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

    def test_static_tuning_controls_notify_and_ignore_invalid_values(self, mock_pyobjc):
        from spoke.tintilla import (
            ADDITIVE_CURVE_MODE_EXPONENTIAL,
            ADDITIVE_CURVE_MODE_RATIONAL,
            LayerVisibilityState,
            WIDE_BLOOM_PROFILE_MIST,
            WIDE_BLOOM_PROFILE_QUEST,
        )

        state = LayerVisibilityState()
        listener = MagicMock()
        state.add_listener(listener)

        state.set_additive_curve_mode(ADDITIVE_CURVE_MODE_RATIONAL)
        state.set_additive_mask_intensity(1.5)
        state.set_wide_bloom_profile(WIDE_BLOOM_PROFILE_MIST)
        state.set_vignette_curve_mode(ADDITIVE_CURVE_MODE_RATIONAL)
        state.set_vignette_mask_intensity(2.0)
        state.set_vignette_profile(WIDE_BLOOM_PROFILE_MIST)
        state.set_additive_curve_mode("bogus")
        state.set_additive_mask_intensity(9.0)
        state.set_wide_bloom_profile("bogus")
        state.set_vignette_curve_mode("bogus")
        state.set_vignette_mask_intensity(9.0)
        state.set_vignette_profile("bogus")

        assert state.additive_curve_mode() == ADDITIVE_CURVE_MODE_RATIONAL
        assert state.additive_mask_intensity() == 1.5
        assert state.wide_bloom_profile() == WIDE_BLOOM_PROFILE_MIST
        assert state.vignette_curve_mode() == ADDITIVE_CURVE_MODE_RATIONAL
        assert state.vignette_mask_intensity() == 2.0
        assert state.vignette_profile() == WIDE_BLOOM_PROFILE_MIST
        assert listener.call_count == 6

        state.set_additive_curve_mode(ADDITIVE_CURVE_MODE_EXPONENTIAL)
        state.set_wide_bloom_profile(WIDE_BLOOM_PROFILE_QUEST)
        state.set_vignette_curve_mode(ADDITIVE_CURVE_MODE_EXPONENTIAL)
        state.set_vignette_profile(WIDE_BLOOM_PROFILE_QUEST)

        assert state.additive_curve_mode() == ADDITIVE_CURVE_MODE_EXPONENTIAL
        assert state.wide_bloom_profile() == WIDE_BLOOM_PROFILE_QUEST
        assert state.vignette_curve_mode() == ADDITIVE_CURVE_MODE_EXPONENTIAL
        assert state.vignette_profile() == WIDE_BLOOM_PROFILE_QUEST


class TestTintillaPanelController:
    def test_setup_enables_panel_frame_autosave(self, mock_pyobjc):
        from spoke.tintilla import LayerVisibilityState, TintillaPanelController

        controller = TintillaPanelController.alloc().initWithState_(LayerVisibilityState())

        controller.setup()

        controller._panel.setFrameAutosaveName_.assert_called_once_with("SpokeTintillaPanel")

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

    def test_refresh_controls_reflects_static_tuning_state(self, mock_pyobjc):
        from spoke.tintilla import (
            ADDITIVE_CURVE_MODE_RATIONAL,
            LayerVisibilityState,
            TintillaPanelController,
            WIDE_BLOOM_PROFILE_MIST,
        )

        state = LayerVisibilityState()
        state.set_additive_curve_mode(ADDITIVE_CURVE_MODE_RATIONAL)
        state.set_additive_mask_intensity(1.5)
        state.set_wide_bloom_profile(WIDE_BLOOM_PROFILE_MIST)
        state.set_vignette_curve_mode(ADDITIVE_CURVE_MODE_RATIONAL)
        state.set_vignette_mask_intensity(2.0)
        state.set_vignette_profile(WIDE_BLOOM_PROFILE_MIST)

        controller = TintillaPanelController.alloc().initWithState_(state)
        controller._additive_curve_mode_button = MagicMock()
        controller._additive_mask_intensity_buttons_by_value = {
            1.0: MagicMock(),
            1.5: MagicMock(),
            2.0: MagicMock(),
        }
        controller._wide_bloom_profile_buttons_by_value = {
            "tight": MagicMock(),
            "quest": MagicMock(),
            "mist": MagicMock(),
        }
        controller._vignette_curve_mode_button = MagicMock()
        controller._vignette_mask_intensity_buttons_by_value = {
            1.0: MagicMock(),
            1.5: MagicMock(),
            2.0: MagicMock(),
        }
        controller._vignette_profile_buttons_by_value = {
            "tight": MagicMock(),
            "quest": MagicMock(),
            "mist": MagicMock(),
        }

        controller._refresh_controls()

        controller._additive_curve_mode_button.setTitle_.assert_called_once_with("Glow Curve: Rational")
        controller._additive_mask_intensity_buttons_by_value[1.5].setState_.assert_called_once_with(1)
        controller._additive_mask_intensity_buttons_by_value[1.0].setState_.assert_called_once_with(0)
        controller._wide_bloom_profile_buttons_by_value["mist"].setState_.assert_called_once_with(1)
        controller._wide_bloom_profile_buttons_by_value["quest"].setState_.assert_called_once_with(0)
        controller._vignette_curve_mode_button.setTitle_.assert_called_once_with("Vignette Curve: Rational")
        controller._vignette_mask_intensity_buttons_by_value[2.0].setState_.assert_called_once_with(1)
        controller._vignette_mask_intensity_buttons_by_value[1.0].setState_.assert_called_once_with(0)
        controller._vignette_profile_buttons_by_value["mist"].setState_.assert_called_once_with(1)
        controller._vignette_profile_buttons_by_value["quest"].setState_.assert_called_once_with(0)

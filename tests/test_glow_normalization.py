"""Tests for the glow overlay's adaptive noise floor normalization."""


class TestNoiseFloorAdaptation:
    """Test the noise floor math used in glow amplitude normalization."""

    def _simulate_updates(self, rms_values):
        """Simulate the noise floor adaptation logic and return (floor, signal) pairs."""
        noise_floor = 0.0
        results = []
        for rms in rms_values:
            if rms < noise_floor or noise_floor == 0.0:
                noise_floor += (rms - noise_floor) * 0.05
            else:
                noise_floor += (rms - noise_floor) * 0.002
            signal = max(rms - noise_floor, 0.0)
            results.append((noise_floor, signal))
        return results

    def test_silence_floor_stays_near_zero(self):
        """In silence, the noise floor should stay near zero."""
        results = self._simulate_updates([0.0001] * 100)
        floor, signal = results[-1]
        assert floor < 0.001

    def test_fan_noise_raises_floor(self):
        """Constant background noise should raise the floor over time."""
        # Simulate fan noise at RMS ~0.005
        results = self._simulate_updates([0.005] * 1000)
        floor, signal = results[-1]
        # Floor should approach the fan noise level
        assert floor > 0.003
        # Signal should be near zero (it's just noise)
        assert signal < 0.002

    def test_speech_over_fan_produces_signal(self):
        """Speech louder than fan noise should produce positive signal."""
        # 500 updates of fan noise to establish floor
        fan = [0.005] * 500
        # Then speech at 0.05 (10x fan)
        speech = [0.05] * 50
        results = self._simulate_updates(fan + speech)
        # Last result should have high signal
        _, signal = results[-1]
        assert signal > 0.03

    def test_floor_drops_when_noise_stops(self):
        """Floor should adapt down when noise goes away."""
        # Fan noise, then silence
        fan = [0.005] * 500
        silence = [0.0001] * 200
        results = self._simulate_updates(fan + silence)
        floor_during_fan = results[499][0]
        floor_after_silence = results[-1][0]
        assert floor_after_silence < floor_during_fan * 0.5

    def test_floor_rises_slowly(self):
        """Floor should rise slowly so speech doesn't raise it much."""
        # Brief loud speech shouldn't move the floor significantly
        silence = [0.001] * 100
        speech = [0.05] * 20  # short burst
        more_silence = [0.001] * 100
        results = self._simulate_updates(silence + speech + more_silence)
        floor_before = results[99][0]
        floor_after = results[-1][0]
        # Floor shouldn't have risen much from a short burst
        assert floor_after < 0.005

    def test_signal_never_negative(self):
        """Signal should never go below zero."""
        # Wild fluctuations
        import random
        random.seed(42)
        rms_values = [random.random() * 0.1 for _ in range(500)]
        results = self._simulate_updates(rms_values)
        for _, signal in results:
            assert signal >= 0.0

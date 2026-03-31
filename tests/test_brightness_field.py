import numpy as np
import pytest
from spoke.glow import BrightnessField

def test_brightness_field_average():
    samples = [0.1] * 8 + [0.9] * 8
    field = BrightnessField(samples)
    assert field.average == pytest.approx(0.5)

def test_brightness_field_interpolation_corners():
    # Grid points are at 0.125, 0.375, 0.625, 0.875
    samples = [0.0] * 16
    samples[0] = 1.0  # (0.125, 0.125)
    field = BrightnessField(samples)
    
    # Exact point
    assert field.sample_at(0.125, 0.125) == pytest.approx(1.0)
    # Outside corner (clamped)
    assert field.sample_at(0.0, 0.0) == pytest.approx(1.0)
    # Middle of first cell (0.25, 0.25)
    # u = (0.25 - 0.125) * 4 = 0.5
    # v = (0.25 - 0.125) * 4 = 0.5
    # s00=1, others=0. (1-0.5)*(1-0.5) = 0.25
    assert field.sample_at(0.25, 0.25) == pytest.approx(0.25)

def test_brightness_field_bilinear():
    samples = [0.0] * 16
    samples[0] = 1.0  # (0,0) grid
    samples[1] = 0.5  # (1,0) grid
    samples[4] = 0.5  # (0,1) grid
    samples[5] = 0.0  # (1,1) grid
    field = BrightnessField(samples)
    
    # Center of this 2x2 patch
    assert field.sample_at(0.25, 0.25) == pytest.approx(0.5)

def test_local_brightness_field_generation(monkeypatch):
    from spoke.overlay import TranscriptionOverlay
    from AppKit import NSScreen, NSWindow, NSView
    from Foundation import NSMakeRect, NSSize, NSPoint
    
    # Mock screen and window for overlay
    class MockSize:
        def __init__(self, w, h): self.width, self.height = w, h
    class MockOrigin:
        def __init__(self, x, y): self.x, self.y = x, y
    class MockFrame:
        def __init__(self, x, y, w, h):
            self.origin = MockOrigin(x, y)
            self.size = MockSize(w, h)
            
    class MockWindow:
        def frame(self): return MockFrame(100, 100, 600, 100)
    class MockScreen:
        def frame(self): return MockFrame(0, 0, 2000, 1000)
        def backingScaleFactor(self): return 2.0

    overlay = TranscriptionOverlay.alloc().init()
    overlay._window = MockWindow()
    overlay._screen = MockScreen()
    
    # Grid: bottom half dark (0.0), top half bright (1.0)
    samples = [0.0] * 8 + [1.0] * 8
    field = BrightnessField(samples)
    overlay.set_brightness(field)
    
    # Overlay is at y=100 to 200 on 1000 screen height.
    # Screen fractions: y = 0.1 to 0.2.
    # Our grid points are at y=0.125, 0.375...
    # So the overlay is mostly in the 0.0 zone.
    
    local_field = overlay._get_local_brightness_field(100, 20)
    assert local_field.shape == (20, 100)
    assert np.mean(local_field) < 0.1  # mostly dark

import importlib
import sys

import numpy as np
from unittest.mock import MagicMock


def _install_quartz_image_stubs(mock_pyobjc):
    quartz = mock_pyobjc["Quartz"]
    quartz.CGColorSpaceCreateDeviceRGB = MagicMock(return_value=MagicMock())
    quartz.CGDataProviderCreateWithCFData = MagicMock(return_value=MagicMock())
    quartz.CGImageCreate = MagicMock(return_value=MagicMock())
    quartz.kCGImageAlphaPremultipliedLast = "premultiplied-last"
    quartz.kCGRenderingIntentDefault = "default"


def test_fill_field_to_image_rounds_alpha_at_final_output_boundary(mock_pyobjc):
    """A barely-visible fill sample should survive if final encoding rounds instead of floors."""
    _install_quartz_image_stubs(mock_pyobjc)
    sys.modules.pop("spoke.overlay", None)
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.overlay")
    try:
        alpha = np.full((1, 1), 0.51 / 255.0, dtype=np.float32)
        _, _payload = mod._fill_field_to_image(alpha, 255, 255, 255)

        glow_mod = importlib.import_module("spoke.glow")
        encoded = glow_mod.NSData.dataWithBytes_length_.call_args[0][0]
        assert encoded == bytes([1, 1, 1, 1])
    finally:
        sys.modules.pop("spoke.overlay", None)
        sys.modules.pop("spoke.glow", None)


def test_fill_field_to_image_keeps_premultiplied_rgba_alignment(mock_pyobjc):
    """Encoded fill channels should stay premultiplied with the rounded alpha sample."""
    _install_quartz_image_stubs(mock_pyobjc)
    sys.modules.pop("spoke.overlay", None)
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.overlay")
    try:
        alpha = np.full((1, 1), 127.6 / 255.0, dtype=np.float32)
        _, _payload = mod._fill_field_to_image(alpha, 255, 128, 0)

        glow_mod = importlib.import_module("spoke.glow")
        encoded = glow_mod.NSData.dataWithBytes_length_.call_args[0][0]
        assert encoded == bytes([128, 64, 0, 128])
    finally:
        sys.modules.pop("spoke.overlay", None)
        sys.modules.pop("spoke.glow", None)

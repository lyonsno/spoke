"""Tests for the Terror Form Metal SDF card renderer."""

from __future__ import annotations

import struct
from unittest.mock import patch

import pytest


def test_uniform_layout_size():
    """Uniform buffer must hold 8 global + 32*8 card floats."""
    from spoke.terraform_metal import _UNIFORM_SIZE, _UNIFORM_FLOAT_COUNT
    assert _UNIFORM_FLOAT_COUNT == 8 + 32 * 8  # 264 floats
    assert _UNIFORM_SIZE == _UNIFORM_FLOAT_COUNT * 4  # 1056 bytes


def test_card_info_fields():
    """CardInfo stores all 8 float fields."""
    from spoke.terraform_metal import CardInfo
    c = CardInfo(x=10.0, y=20.0, width=300.0, height=60.0,
                 r=0.5, g=0.6, b=0.8, alpha=0.75)
    assert c.x == 10.0
    assert c.alpha == 0.75


def test_uniform_packing_round_trip():
    """Pack uniforms the same way draw_frame does, verify round-trip."""
    from spoke.terraform_metal import (
        _UNIFORM_LAYOUT, _UNIFORM_FLOAT_COUNT, _MAX_CARDS, _CARD_FLOATS, CardInfo,
    )

    cards = [
        CardInfo(x=10.0, y=200.0, width=300.0, height=60.0,
                 r=0.5, g=0.6, b=0.84, alpha=0.75),
        CardInfo(x=10.0, y=130.0, width=300.0, height=60.0,
                 r=0.8, g=0.3, b=0.2, alpha=0.75),
    ]

    # Pack like draw_frame does
    values = [640.0, 1800.0, 20.0, 5.0, 0.65, 0.0, 42.0, float(len(cards))]
    for i in range(_MAX_CARDS):
        if i < len(cards):
            c = cards[i]
            values.extend([c.x, c.y, c.width, c.height, c.r, c.g, c.b, c.alpha])
        else:
            values.extend([0.0] * _CARD_FLOATS)

    assert len(values) == _UNIFORM_FLOAT_COUNT
    payload = struct.pack(_UNIFORM_LAYOUT, *values)

    # Unpack and verify
    unpacked = struct.unpack(_UNIFORM_LAYOUT, payload)
    assert unpacked[0] == 640.0  # viewport_w
    assert unpacked[1] == 1800.0  # viewport_h
    assert unpacked[7] == 2.0  # card_count
    # First card starts at index 8
    assert unpacked[8] == 10.0  # card[0].x
    assert unpacked[9] == 200.0  # card[0].y
    assert unpacked[14] == pytest.approx(0.84)  # card[0].b
    assert unpacked[15] == pytest.approx(0.75)  # card[0].alpha
    # Second card at index 16
    assert unpacked[16] == 10.0  # card[1].x
    assert unpacked[17] == 130.0  # card[1].y
    assert unpacked[20] == pytest.approx(0.8)  # card[1].r


def test_max_cards_clamped():
    """set_cards should silently clamp to MAX_CARDS."""
    from spoke.terraform_metal import _MAX_CARDS, CardInfo, TerraformCardRenderer

    # Create renderer (requires Metal — skip if unavailable)
    from spoke.terraform_metal import metal_available
    if not metal_available():
        pytest.skip("Metal not available")

    renderer = TerraformCardRenderer((320, 900), 2.0)
    big_list = [
        CardInfo(x=0, y=float(i * 66), width=300, height=60,
                 r=0.5, g=0.5, b=0.5, alpha=0.5)
        for i in range(50)
    ]
    renderer.set_cards(big_list)
    assert len(renderer._cards) == _MAX_CARDS


def test_metal_available_returns_bool():
    """metal_available() returns a bool without raising."""
    from spoke.terraform_metal import metal_available
    result = metal_available()
    assert isinstance(result, bool)


def test_fallback_when_metal_unavailable():
    """When Metal is unavailable, metal_available() returns False."""
    with patch("spoke.terraform_metal._metal_device_available", return_value=False):
        from spoke.terraform_metal import _metal_device_available
        assert not _metal_device_available()


def test_temp_fill_color_blending():
    """Temperature fill colors blend base with temperature tint."""
    from spoke.terraform_hud import _temp_fill_color, _SDF_BASE_DARK, _TEMP_COLORS

    # "hot" on dark bg should shift base toward red
    r, g, b = _temp_fill_color("hot", brightness=0.0)
    br, bg, bb = _SDF_BASE_DARK
    assert r > br  # red component increased
    assert g < bg  # green component decreased (toward 0.3)

    # Unknown temperature should return near-base color
    r2, g2, b2 = _temp_fill_color(None, brightness=0.0)
    assert abs(r2 - br) < 0.05
    assert abs(g2 - bg) < 0.05

    # Light bg should produce darker base
    r3, g3, b3 = _temp_fill_color(None, brightness=1.0)
    assert r3 < r2  # darker on light bg


def test_temp_fill_color_all_temperatures():
    """All known temperatures produce valid RGB in [0, 1]."""
    from spoke.terraform_hud import _temp_fill_color, _TEMP_COLORS

    for temp in list(_TEMP_COLORS.keys()) + [None, "unknown"]:
        r, g, b = _temp_fill_color(temp)
        assert 0.0 <= r <= 1.0, f"{temp}: r={r}"
        assert 0.0 <= g <= 1.0, f"{temp}: g={g}"
        assert 0.0 <= b <= 1.0, f"{temp}: b={b}"

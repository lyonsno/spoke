import pytest

from spoke.optical_frame_witness import (
    CropRect,
    ffmpeg_crop_filter,
    motion_bbox_from_gray_frames,
    parse_crop_rect,
    sample_timestamps,
    tile_layout,
)


def test_parse_crop_rect_accepts_comma_and_colon_forms():
    assert parse_crop_rect("10,20,300,200") == CropRect(10, 20, 300, 200)
    assert parse_crop_rect("10:20:300:200") == CropRect(10, 20, 300, 200)


@pytest.mark.parametrize("value", ["", "1,2,3", "1,2,0,4", "-1,2,3,4", "x,y,w,h"])
def test_parse_crop_rect_rejects_invalid_shapes(value):
    with pytest.raises(ValueError):
        parse_crop_rect(value)


def test_ffmpeg_crop_filter_uses_integer_rect_components():
    assert ffmpeg_crop_filter(CropRect(10, 20, 300, 200)) == "crop=300:200:10:20"


def test_sample_timestamps_is_dense_and_clamped_to_duration():
    assert sample_timestamps(duration_s=0.35, fps=10) == [0.0, 0.1, 0.2, 0.3]
    assert sample_timestamps(duration_s=2.0, fps=4, start_s=0.5, end_s=1.1) == [
        0.5,
        0.75,
        1.0,
    ]


def test_sample_timestamps_can_evenly_cap_frame_count():
    assert sample_timestamps(duration_s=1.0, fps=60, max_frames=5) == [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]


def test_motion_bbox_from_gray_frames_unions_changed_pixels_with_margin():
    width = 5
    height = 4
    base = bytes([0] * (width * height))
    changed = bytearray(base)
    changed[1 + 1 * width] = 100
    changed[2 + 1 * width] = 100
    changed[3 + 2 * width] = 100

    assert motion_bbox_from_gray_frames([base, bytes(changed)], width, height, threshold=8, margin=1) == CropRect(
        0,
        0,
        5,
        4,
    )


def test_motion_bbox_from_gray_frames_returns_none_when_pixels_do_not_change():
    frame = bytes([12] * 9)
    assert motion_bbox_from_gray_frames([frame, frame], 3, 3, threshold=8) is None


def test_tile_layout_uses_concrete_rows_for_ffmpeg():
    assert tile_layout(1, columns=6) == "6x1"
    assert tile_layout(13, columns=6) == "6x3"

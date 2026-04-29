import ctypes
import sys
import types

import pytest

from spoke import metal_warp


def test_metal_warp_owns_visual_tuning_constants_while_backdrop_is_fallback():
    from spoke import backdrop_stream

    expected_metal_tuning = {
        "_WARP_CENTER_FLOOR": 0.94,
        "_WARP_FIELD_EXPONENT": 0.25,
        "_WARP_EXTERIOR_MAG_STRENGTH": 0.3,
    }

    for name, expected in expected_metal_tuning.items():
        assert getattr(metal_warp, name) == pytest.approx(expected)
        assert getattr(metal_warp, name) != getattr(backdrop_stream, name)


def test_warp_alias_mip_bias_stays_zero_for_near_identity_warp():
    assert metal_warp._warp_alias_mip_bias(1.0, 1.0) == 0.0
    assert metal_warp._warp_alias_mip_bias(1.08, 0.96) == 0.0


def test_warp_alias_mip_bias_rises_for_violent_warp():
    assert metal_warp._warp_alias_mip_bias(2.6, 1.7) > 0.0
    assert metal_warp._warp_alias_mip_bias(0.38, 0.52) > 0.0


def test_pack_warp_params_uses_shell_specific_bleed_zone_frac():
    payload = metal_warp._pack_warp_params(
        1440.0,
        900.0,
        {
            "content_width_points": 640.0,
            "content_height_points": 120.0,
            "corner_radius_points": 20.0,
            "bleed_zone_frac": 0.8,
            "exterior_mix_width_points": 20.0,
        },
    )
    values = metal_warp.struct.unpack("20f", payload)
    assert values[16] == pytest.approx(0.8)
    assert values[17] == pytest.approx(20.0)


def test_pack_warp_params_uses_shell_specific_axis_squeeze():
    payload = metal_warp._pack_warp_params(
        1440.0,
        900.0,
        {
            "content_width_points": 640.0,
            "content_height_points": 120.0,
            "corner_radius_points": 20.0,
            "x_squeeze": 3.25,
            "y_squeeze": 1.2,
        },
    )
    values = metal_warp.struct.unpack("20f", payload)
    assert values[18] == pytest.approx(3.25)
    assert values[19] == pytest.approx(1.2)


def test_warp_dispatch_box_respects_shell_specific_bleed_zone_frac():
    wide = metal_warp._warp_dispatch_box(
        1440.0,
        900.0,
        {
            "center_x": 720.0,
            "center_y": 450.0,
            "content_width_points": 640.0,
            "content_height_points": 120.0,
            "bleed_zone_frac": 0.8,
        },
    )
    tight = metal_warp._warp_dispatch_box(
        1440.0,
        900.0,
        {
            "center_x": 720.0,
            "center_y": 450.0,
            "content_width_points": 640.0,
            "content_height_points": 120.0,
            "bleed_zone_frac": 0.4,
        },
    )
    assert tight[0] > wide[0]
    assert tight[1] > wide[1]
    assert tight[2] < wide[2]
    assert tight[3] < wide[3]


def test_warp_dispatch_box_respects_shell_specific_corner_radius():
    rounder = metal_warp._warp_dispatch_box(
        1440.0,
        900.0,
        {
            "center_x": 720.0,
            "center_y": 450.0,
            "content_width_points": 640.0,
            "content_height_points": 120.0,
            "corner_radius_points": 60.0,
            "bleed_zone_frac": 0.8,
        },
    )
    squarer = metal_warp._warp_dispatch_box(
        1440.0,
        900.0,
        {
            "center_x": 720.0,
            "center_y": 450.0,
            "content_width_points": 640.0,
            "content_height_points": 120.0,
            "corner_radius_points": 20.0,
            "bleed_zone_frac": 0.8,
        },
    )
    assert squarer[0] > rounder[0]
    assert squarer[1] > rounder[1]
    assert squarer[2] < rounder[2]
    assert squarer[3] < rounder[3]


def test_warp_exterior_mix_weight_keeps_boundary_strength_but_starts_later_with_tighter_width():
    assert metal_warp._warp_exterior_mix_weight(0.0, 40.0) == pytest.approx(1.0)
    assert metal_warp._warp_exterior_mix_weight(0.0, 20.0) == pytest.approx(1.0)
    assert metal_warp._warp_exterior_mix_weight(10.0, 20.0) < metal_warp._warp_exterior_mix_weight(10.0, 40.0)
    assert metal_warp._warp_exterior_mix_weight(30.0, 20.0) == pytest.approx(0.0)


def test_multi_shell_draw_uses_distinct_params_buffers(monkeypatch):
    """Each shell pass needs immutable params for the queued Metal command."""

    class FakeBuffer:
        def __init__(self, data_or_length):
            if isinstance(data_or_length, bytes):
                self._raw = ctypes.create_string_buffer(data_or_length)
            else:
                self._raw = ctypes.create_string_buffer(int(data_or_length))

        def contents(self):
            return ctypes.addressof(self._raw)

        def payload(self):
            return bytes(self._raw.raw[: metal_warp._WARP_PARAMS_SIZE])

    class FakeTexture:
        def __init__(self, width, height):
            self._width = width
            self._height = height

        def width(self):
            return self._width

        def height(self):
            return self._height

    class FakeTextureDescriptor:
        def __init__(self, width=0, height=0):
            self.width = width
            self.height = height

        @classmethod
        def texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            cls, _fmt, width, height, _mipmapped
        ):
            return cls(width, height)

        def setUsage_(self, _usage):
            return None

    fake_objc = types.ModuleType("objc")
    fake_objc.lookUpClass = lambda name: FakeTextureDescriptor
    monkeypatch.setitem(sys.modules, "objc", fake_objc)

    class FakeDevice:
        def newTextureWithDescriptor_iosurface_plane_(self, desc, _surface, _plane):
            return FakeTexture(desc.width, desc.height)

        def newTextureWithDescriptor_(self, desc):
            return FakeTexture(desc.width, desc.height)

        def newBufferWithLength_options_(self, length, _options):
            return FakeBuffer(length)

    class FakeBlitEncoder:
        def copyFromTexture_toTexture_(self, *_args):
            return None

        def copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_(
            self, *_args
        ):
            return None

        def generateMipmapsForTexture_(self, _texture):
            return None

        def endEncoding(self):
            return None

    class FakeComputeEncoder:
        def __init__(self):
            self.params_buffer = None

        def setComputePipelineState_(self, _pipeline):
            return None

        def setTexture_atIndex_(self, _texture, _index):
            return None

        def setBuffer_offset_atIndex_(self, buffer, _offset, index):
            if index == 0:
                self.params_buffer = buffer

        def dispatchThreads_threadsPerThreadgroup_(self, _grid, _threadgroup):
            return None

        def endEncoding(self):
            return None

    class FakeCommandBuffer:
        def __init__(self):
            self.encoders = []

        def blitCommandEncoder(self):
            return FakeBlitEncoder()

        def computeCommandEncoder(self):
            encoder = FakeComputeEncoder()
            self.encoders.append(encoder)
            return encoder

        def presentDrawable_(self, _drawable):
            return None

        def commit(self):
            return None

    class FakeCommandQueue:
        def __init__(self):
            self.command_buffers = []

        def commandBuffer(self):
            command_buffer = FakeCommandBuffer()
            self.command_buffers.append(command_buffer)
            return command_buffer

    class FakeDrawable:
        def texture(self):
            return FakeTexture(100, 50)

    monkeypatch.setattr(
        metal_warp,
        "_create_metal_buffer",
        lambda _device, data: FakeBuffer(data),
    )

    pipeline = metal_warp.MetalWarpPipeline.__new__(metal_warp.MetalWarpPipeline)
    pipeline._device = FakeDevice()
    pipeline._command_queue = FakeCommandQueue()
    pipeline._pipeline = object()
    pipeline._mip_texture = None
    pipeline._mip_texture_size = None
    pipeline._accum_textures = [None, None]
    pipeline._accum_texture_size = None
    pipeline._accum_index = 0
    pipeline._accum_generation = 0
    pipeline._thread_exec_width = 8
    pipeline._max_tg_height = 8
    pipeline._params_buffer = FakeBuffer(metal_warp._WARP_PARAMS_SIZE)

    assert pipeline.warp_to_drawable(
        object(),
        FakeDrawable(),
        width=100,
        height=50,
        shell_config=[
            {
                "center_x": 25.0,
                "center_y": 20.0,
                "content_width_points": 20.0,
                "content_height_points": 10.0,
            },
            {
                "center_x": 75.0,
                "center_y": 20.0,
                "content_width_points": 20.0,
                "content_height_points": 10.0,
            },
        ],
    )

    encoders = pipeline._command_queue.command_buffers[-1].encoders
    params_buffers = [encoder.params_buffer for encoder in encoders]

    assert len(params_buffers) == 2
    assert len({id(buffer) for buffer in params_buffers}) == 2
    assert [
        metal_warp.struct.unpack("20f", buffer.payload())[10]
        for buffer in params_buffers
    ] == [pytest.approx(25.0), pytest.approx(75.0)]


def test_warp_diagnostics_record_fullscreen_mip_cost_against_bounded_shell(monkeypatch):
    """Current blur source work is full-frame even when the shell dispatch is bounded."""

    class FakeBuffer:
        def __init__(self, data_or_length):
            if isinstance(data_or_length, bytes):
                self._raw = ctypes.create_string_buffer(data_or_length)
            else:
                self._raw = ctypes.create_string_buffer(int(data_or_length))

        def contents(self):
            return ctypes.addressof(self._raw)

    class FakeTexture:
        def __init__(self, width, height):
            self._width = width
            self._height = height

        def width(self):
            return self._width

        def height(self):
            return self._height

    class FakeTextureDescriptor:
        def __init__(self, width=0, height=0):
            self.width = width
            self.height = height

        @classmethod
        def texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            cls, _fmt, width, height, _mipmapped
        ):
            return cls(width, height)

        def setUsage_(self, _usage):
            return None

    fake_objc = types.ModuleType("objc")
    fake_objc.lookUpClass = lambda name: FakeTextureDescriptor
    monkeypatch.setitem(sys.modules, "objc", fake_objc)

    class FakeDevice:
        def newTextureWithDescriptor_iosurface_plane_(self, desc, _surface, _plane):
            return FakeTexture(desc.width, desc.height)

        def newTextureWithDescriptor_(self, desc):
            return FakeTexture(desc.width, desc.height)

        def newBufferWithLength_options_(self, length, _options):
            return FakeBuffer(length)

    class FakeBlitEncoder:
        def copyFromTexture_toTexture_(self, *_args):
            return None

        def copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_(
            self, *_args
        ):
            return None

        def generateMipmapsForTexture_(self, _texture):
            return None

        def endEncoding(self):
            return None

    class FakeComputeEncoder:
        def setComputePipelineState_(self, _pipeline):
            return None

        def setTexture_atIndex_(self, _texture, _index):
            return None

        def setBuffer_offset_atIndex_(self, _buffer, _offset, _index):
            return None

        def dispatchThreads_threadsPerThreadgroup_(self, _grid, _threadgroup):
            return None

        def endEncoding(self):
            return None

    class FakeCommandBuffer:
        def blitCommandEncoder(self):
            return FakeBlitEncoder()

        def computeCommandEncoder(self):
            return FakeComputeEncoder()

        def presentDrawable_(self, _drawable):
            return None

        def commit(self):
            return None

    class FakeCommandQueue:
        def commandBuffer(self):
            return FakeCommandBuffer()

    class FakeDrawable:
        def texture(self):
            return FakeTexture(100, 50)

    pipeline = metal_warp.MetalWarpPipeline.__new__(metal_warp.MetalWarpPipeline)
    pipeline._device = FakeDevice()
    pipeline._command_queue = FakeCommandQueue()
    pipeline._pipeline = object()
    pipeline._mip_texture = None
    pipeline._mip_texture_size = None
    pipeline._accum_textures = [None, None]
    pipeline._accum_texture_size = None
    pipeline._accum_index = 0
    pipeline._accum_generation = 0
    pipeline._thread_exec_width = 8
    pipeline._max_tg_height = 8
    pipeline._params_buffer = FakeBuffer(metal_warp._WARP_PARAMS_SIZE)

    shell_config = {
        "center_x": 50.0,
        "center_y": 25.0,
        "content_width_points": 20.0,
        "content_height_points": 10.0,
        "bleed_zone_frac": 0.0,
    }
    expected_box = metal_warp._warp_dispatch_box(100, 50, shell_config)
    expected_dispatch_pixels = (expected_box[2] - expected_box[0]) * (expected_box[3] - expected_box[1])

    assert pipeline.warp_to_drawable(
        object(),
        FakeDrawable(),
        width=100,
        height=50,
        shell_config=shell_config,
    )

    diagnostics = pipeline.diagnostics_snapshot()
    assert diagnostics["drawable_copy_frames"] == 1
    assert diagnostics["drawable_copy_pixels"] == 5_000
    assert diagnostics["mip_generation_frames"] == 1
    assert diagnostics["mip_generation_source_pixels"] == 5_000
    assert diagnostics["warp_dispatch_pixels"] == expected_dispatch_pixels
    assert diagnostics["warp_dispatch_pixels"] < diagnostics["mip_generation_source_pixels"]

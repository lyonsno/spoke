"""Metal SDF card renderer for the Terror Form HUD.

Renders per-card rounded-rect SDF surfaces in a single draw call.
The fragment shader evaluates N analytical SDFs and composites with
max-alpha to avoid brightness doubling at overlap zones.

Self-contained Metal bootstrap — does not depend on glow.py's Metal
infrastructure (which hasn't landed on main yet).
"""

from __future__ import annotations

import ctypes
import logging
import struct
import time
from dataclasses import dataclass

import objc

logger = logging.getLogger(__name__)

# ── Metal constants ──────────────────────────────────────────────────
_MTL_PIXEL_FORMAT_BGRA8_UNORM = 80
_MTL_LOAD_ACTION_CLEAR = 2
_MTL_STORE_ACTION_STORE = 1
_MTL_PRIMITIVE_TYPE_TRIANGLE_STRIP = 4

_MAX_CARDS = 32
# Global uniforms: 8 floats.  Per-card: 8 floats each.
_GLOBAL_FLOATS = 8
_CARD_FLOATS = 8
_UNIFORM_FLOAT_COUNT = _GLOBAL_FLOATS + _MAX_CARDS * _CARD_FLOATS
_UNIFORM_LAYOUT = "<" + "f" * _UNIFORM_FLOAT_COUNT
_UNIFORM_SIZE = struct.calcsize(_UNIFORM_LAYOUT)

# ── Metal bootstrap ──────────────────────────────────────────────────
_metal_bundle_loaded = False
_metal_bundle_error: Exception | None = None
MTLCreateSystemDefaultDevice = None  # populated by _load_metal_symbols


def _load_metal_symbols() -> None:
    global _metal_bundle_loaded, _metal_bundle_error, MTLCreateSystemDefaultDevice
    if _metal_bundle_loaded:
        return
    if _metal_bundle_error is not None:
        raise _metal_bundle_error
    try:
        _ns = {}
        metal_bundle = objc.loadBundle(
            "Metal", _ns,
            bundle_path="/System/Library/Frameworks/Metal.framework",
        )
        objc.loadBundle(
            "QuartzCore", _ns,
            bundle_path="/System/Library/Frameworks/QuartzCore.framework",
        )
        objc.loadBundleFunctions(
            metal_bundle, _ns,
            [("MTLCreateSystemDefaultDevice", b"@")],
        )
        MTLCreateSystemDefaultDevice = _ns["MTLCreateSystemDefaultDevice"]
        _metal_bundle_loaded = True
    except Exception as exc:
        _metal_bundle_error = exc
        raise


def _metal_device_available() -> bool:
    try:
        _load_metal_symbols()
        return MTLCreateSystemDefaultDevice() is not None
    except Exception:
        return False


def _copy_bytes_to_metal_buffer(buffer, payload: bytes) -> None:
    ctypes.memmove(int(buffer.contents()), payload, len(payload))


# ── Shader source ────────────────────────────────────────────────────
_TERRAFORM_SHADER_SOURCE = """
#include <metal_stdlib>
using namespace metal;

#define MAX_CARDS 32

struct VSOut {
    float4 position [[position]];
    float2 uv;
};

struct CardRect {
    float4 rect;    // x, y, width, height (pixels)
    float4 color;   // r, g, b, alpha
};

struct Uniforms {
    float  viewport_w;
    float  viewport_h;
    float  corner_radius;
    float  fill_width;
    float  interior_floor;
    float  scroll_offset_y;
    float  time;
    float  card_count_f;   // cast to uint in shader
    CardRect cards[MAX_CARDS];
};

vertex VSOut vs_main(uint vid [[vertex_id]]) {
    float2 positions[4] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0,  1.0),
    };
    float2 uvs[4] = {
        float2(0.0, 1.0),
        float2(1.0, 1.0),
        float2(0.0, 0.0),
        float2(1.0, 0.0),
    };
    VSOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

// Rounded-rect SDF — same math as _overlay_rounded_rect_sdf in overlay.py
inline float card_sdf(float2 pixel, float4 rect, float corner_radius) {
    float2 center = rect.xy + rect.zw * 0.5;
    float2 half_size = rect.zw * 0.5 - corner_radius;
    float2 q = abs(pixel - center) - half_size;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - corner_radius;
}

// Exact port of overlay.py _glow_fill_alpha:
// Stretched-exponential: exp(-sqrt(|d|/width))
// Inside: cusp at boundary, floors at interior_floor
// Outside: same cusp, drops all the way to zero
inline float overlay_fill_alpha(float sd, float width, float floor_val) {
    float raw = exp(-sqrt(abs(sd) / max(width, 1e-6)));
    if (sd <= 0.0) {
        return floor_val + (1.0 - floor_val) * raw;
    }
    return raw;
}

// 1px anti-aliased stroke at the card boundary.
inline float hairline(float sd) {
    return clamp(1.0 - abs(sd), 0.0, 1.0);
}

fragment float4 fs_fill(
    VSOut in [[stage_in]],
    constant Uniforms& u [[buffer(0)]]
) {
    float2 pixel = float2(in.uv.x * u.viewport_w,
                          in.uv.y * u.viewport_h);

    uint count = min(uint(u.card_count_f), uint(MAX_CARDS));

    // Hairline stroke parameters
    float hl_strength = 0.22;
    float3 hl_rgb = float3(1.0, 1.0, 1.0);

    float best_alpha = 0.0;
    float3 best_rgb = float3(0.0);
    float best_hl = 0.0;

    for (uint i = 0; i < count; i++) {
        float4 rect = u.cards[i].rect;
        rect.y -= u.scroll_offset_y;

        float sd = card_sdf(pixel, rect, u.corner_radius);

        // Fill: exact overlay profile
        float a = overlay_fill_alpha(sd, u.fill_width, u.interior_floor)
                  * u.cards[i].color.a;

        if (a > best_alpha) {
            float t = best_alpha / (best_alpha + a + 1e-6);
            best_rgb = mix(u.cards[i].color.rgb, best_rgb, t);
            best_alpha = a;
        } else if (a > best_alpha * 0.05) {
            float t = a / (best_alpha + a + 1e-6);
            best_rgb = mix(best_rgb, u.cards[i].color.rgb, t);
        }

        // Hairline at boundary
        best_hl = max(best_hl, hairline(sd));
    }

    // Additive hairline over fill
    float3 hl = hl_rgb * (best_hl * hl_strength);
    float hl_a = best_hl * hl_strength;

    float final_alpha = best_alpha + hl_a * (1.0 - best_alpha);
    float3 final_rgb = best_rgb * best_alpha + hl * (1.0 - best_alpha);

    return float4(final_rgb, final_alpha);
}
"""


# ── Pipeline builder ─────────────────────────────────────────────────
def _build_terraform_pipeline(device):
    """Build the single fill+chroma pipeline."""
    _load_metal_symbols()
    result = device.newLibraryWithSource_options_error_(
        _TERRAFORM_SHADER_SOURCE, None, None,
    )
    library = result[0] if isinstance(result, tuple) else result
    if library is None:
        error = result[1] if isinstance(result, tuple) else None
        raise RuntimeError(f"Metal shader compilation failed: {error}")

    desc = objc.lookUpClass("MTLRenderPipelineDescriptor").alloc().init()
    desc.setVertexFunction_(library.newFunctionWithName_("vs_main"))
    desc.setFragmentFunction_(library.newFunctionWithName_("fs_fill"))
    attachment = desc.colorAttachments().objectAtIndexedSubscript_(0)
    attachment.setPixelFormat_(_MTL_PIXEL_FORMAT_BGRA8_UNORM)
    attachment.setBlendingEnabled_(True)
    attachment.setSourceRGBBlendFactor_(1)        # One (premultiplied)
    attachment.setDestinationRGBBlendFactor_(5)    # OneMinusSourceAlpha
    attachment.setSourceAlphaBlendFactor_(1)        # One
    attachment.setDestinationAlphaBlendFactor_(5)   # OneMinusSourceAlpha
    result = device.newRenderPipelineStateWithDescriptor_error_(desc, None)
    pipeline = result[0] if isinstance(result, tuple) else result
    if pipeline is None:
        error = result[1] if isinstance(result, tuple) else None
        raise RuntimeError(f"Pipeline creation failed: {error}")
    return pipeline


# ── Data ─────────────────────────────────────────────────────────────
@dataclass
class CardInfo:
    """Card rectangle and fill color, all in pixel coordinates."""
    x: float
    y: float
    width: float
    height: float
    r: float
    g: float
    b: float
    alpha: float


# ── Renderer ─────────────────────────────────────────────────────────
class TerraformCardRenderer:
    """GPU-accelerated SDF card renderer for the Terror Form HUD.

    Single CAMetalLayer. Fill + chromatic aberration rings rendered in
    one shader pass — no separate layers, no timing mismatch on scroll.
    """

    def __init__(self, frame_size: tuple[float, float], scale: float):
        _load_metal_symbols()
        device = MTLCreateSystemDefaultDevice()
        if device is None:
            raise RuntimeError("Metal default device unavailable")

        self._device = device
        self._queue = device.newCommandQueue()
        self._pipeline = _build_terraform_pipeline(device)
        self._uniform_buffer = device.newBufferWithLength_options_(_UNIFORM_SIZE, 0)

        self._cards: list[CardInfo] = []
        self._scroll_offset_y: float = 0.0
        self._corner_radius: float = 10.0
        self._fill_width: float = 2.5   # matches overlay width param
        self._interior_floor: float = 0.55  # matches overlay dark-bg floor
        self._pixel_width: int = 1
        self._pixel_height: int = 1

        self._layer = objc.lookUpClass("CAMetalLayer").layer()
        self._layer.setDevice_(device)
        self._layer.setPixelFormat_(_MTL_PIXEL_FORMAT_BGRA8_UNORM)
        self._layer.setFramebufferOnly_(True)
        self._layer.setOpaque_(False)
        self._set_layer_geometry(frame_size, scale)

    def layer(self):
        """Return the CAMetalLayer for insertion into the view hierarchy."""
        return self._layer

    def set_geometry(self, width_pt: float, height_pt: float, scale: float) -> None:
        self._set_layer_geometry((width_pt, height_pt), scale)

    def set_cards(self, cards: list[CardInfo]) -> None:
        self._cards = cards[:_MAX_CARDS]

    def set_scroll_offset(self, offset_px: float) -> None:
        self._scroll_offset_y = offset_px

    def draw_frame(self) -> bool:
        drawable = self._layer.nextDrawable()
        if drawable is None:
            return False

        values = [
            float(self._pixel_width), float(self._pixel_height),
            float(self._corner_radius), float(self._fill_width),
            float(self._interior_floor), float(self._scroll_offset_y),
            float(time.monotonic() % 1000.0), float(len(self._cards)),
        ]
        for i in range(_MAX_CARDS):
            if i < len(self._cards):
                c = self._cards[i]
                values.extend([c.x, c.y, c.width, c.height, c.r, c.g, c.b, c.alpha])
            else:
                values.extend([0.0] * _CARD_FLOATS)

        payload = struct.pack(_UNIFORM_LAYOUT, *values)
        _copy_bytes_to_metal_buffer(self._uniform_buffer, payload)

        rpd = objc.lookUpClass("MTLRenderPassDescriptor").renderPassDescriptor()
        att = rpd.colorAttachments().objectAtIndexedSubscript_(0)
        att.setTexture_(drawable.texture())
        att.setLoadAction_(_MTL_LOAD_ACTION_CLEAR)
        att.setStoreAction_(_MTL_STORE_ACTION_STORE)
        att.setClearColor_((0.0, 0.0, 0.0, 0.0))

        cmd = self._queue.commandBuffer()
        enc = cmd.renderCommandEncoderWithDescriptor_(rpd)
        enc.setRenderPipelineState_(self._pipeline)
        enc.setFragmentBuffer_offset_atIndex_(self._uniform_buffer, 0, 0)
        enc.drawPrimitives_vertexStart_vertexCount_(_MTL_PRIMITIVE_TYPE_TRIANGLE_STRIP, 0, 4)
        enc.endEncoding()
        cmd.presentDrawable_(drawable)
        cmd.commit()
        return True

    # ── private ──

    def _set_layer_geometry(self, frame_size: tuple[float, float], scale: float) -> None:
        w_pt, h_pt = frame_size
        self._layer.setFrame_(((0, 0), (w_pt, h_pt)))
        self._layer.setContentsScale_(scale)
        self._pixel_width = max(int(round(w_pt * scale)), 1)
        self._pixel_height = max(int(round(h_pt * scale)), 1)
        self._layer.setDrawableSize_((float(self._pixel_width), float(self._pixel_height)))
        self._fill_width = 2.5 * scale
        self._corner_radius = 10.0 * scale


def metal_available() -> bool:
    """Check if Metal is available without creating a renderer."""
    return _metal_device_available()

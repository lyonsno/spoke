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

// Card fill: flat interior at floor, fast Gaussian outer scoop.
// No bright peak at the boundary — uniform material throughout,
// fading to zero outside.
inline float fill_alpha(float sd, float width, float floor_val) {
    if (sd <= 0.0) {
        // Interior: flat at floor level — no edge highlight
        return floor_val;
    }
    // Outside: fast Gaussian scoop from floor to zero
    float normalized = sd / max(width, 1e-6);
    return floor_val * exp(-normalized * normalized);
}

// Ring profile: peaks near the card boundary, fades both inward and outward.
inline float ring_alpha(float sd, float ring_width, float ring_offset) {
    float shifted = sd - ring_offset;
    float normalized = shifted / max(ring_width, 1e-6);
    return exp(-normalized * normalized);
}

fragment float4 fs_fill(
    VSOut in [[stage_in]],
    constant Uniforms& u [[buffer(0)]]
) {
    float2 pixel = float2(in.uv.x * u.viewport_w,
                          in.uv.y * u.viewport_h);

    uint count = min(uint(u.card_count_f), uint(MAX_CARDS));

    float best_alpha = 0.0;
    float3 best_rgb = float3(0.0);

    for (uint i = 0; i < count; i++) {
        float4 rect = u.cards[i].rect;
        rect.y -= u.scroll_offset_y;

        float sd = card_sdf(pixel, rect, u.corner_radius);
        float a = fill_alpha(sd, u.fill_width, u.interior_floor)
                  * u.cards[i].color.a;

        if (a > best_alpha) {
            float t = best_alpha / (best_alpha + a + 1e-6);
            best_rgb = mix(u.cards[i].color.rgb, best_rgb, t);
            best_alpha = a;
        } else if (a > best_alpha * 0.05) {
            float t = a / (best_alpha + a + 1e-6);
            best_rgb = mix(best_rgb, u.cards[i].color.rgb, t);
        }
    }

    return float4(best_rgb * best_alpha, best_alpha);
}

// Chromatic ring pass — renders a single-color ring around each card.
// The ring_color is baked into the uniforms' card colors by the Python side.
fragment float4 fs_ring(
    VSOut in [[stage_in]],
    constant Uniforms& u [[buffer(0)]]
) {
    float2 pixel = float2(in.uv.x * u.viewport_w,
                          in.uv.y * u.viewport_h);

    uint count = min(uint(u.card_count_f), uint(MAX_CARDS));

    float ring_width = u.fill_width * 6.0;     // wide diffuse halo, not a border
    float ring_peak_dist = u.fill_width * 4.0; // peak well outside the card edge

    float best_ring = 0.0;

    for (uint i = 0; i < count; i++) {
        float4 rect = u.cards[i].rect;
        rect.y -= u.scroll_offset_y;

        float sd = card_sdf(pixel, rect, u.corner_radius);
        float r = ring_alpha(sd, ring_width, ring_peak_dist);
        best_ring = max(best_ring, r);
    }

    // card colors carry the ring tint (set by Python: gold or blue)
    float3 ring_color = u.cards[0].color.rgb;
    float ring_a = best_ring * u.cards[0].color.a;  // alpha = ring strength

    return float4(ring_color * ring_a, ring_a);
}
"""


# ── Pipeline builder ─────────────────────────────────────────────────
def _build_terraform_pipelines(device):
    """Build fill and ring pipelines from the shared shader source."""
    _load_metal_symbols()
    result = device.newLibraryWithSource_options_error_(
        _TERRAFORM_SHADER_SOURCE, None, None,
    )
    library = result[0] if isinstance(result, tuple) else result
    if library is None:
        error = result[1] if isinstance(result, tuple) else None
        raise RuntimeError(f"Metal shader compilation failed: {error}")

    vs = library.newFunctionWithName_("vs_main")
    pipelines = {}
    for fs_name in ("fs_fill", "fs_ring"):
        desc = objc.lookUpClass("MTLRenderPipelineDescriptor").alloc().init()
        desc.setVertexFunction_(vs)
        desc.setFragmentFunction_(library.newFunctionWithName_(fs_name))
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
            raise RuntimeError(f"Pipeline creation failed for {fs_name}: {error}")
        pipelines[fs_name] = pipeline
    return pipelines


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
# Chromatic aberration config
_GOLD_RGB = (1.0, 0.85, 0.4)
_BLUE_RGB = (0.4, 0.6, 1.0)
_RING_STRENGTH = 0.07        # per-ring opacity
_CHROMA_OFFSET_PT = 1.5      # point offset for chromatic split


class TerraformCardRenderer:
    """GPU-accelerated SDF card renderer for the Terror Form HUD.

    Three CAMetalLayers: fill (main cards), gold ring, blue ring.
    The ring layers are offset in opposite directions for chromatic
    aberration. All share the same pipeline and uniform buffer format.
    """

    def __init__(self, frame_size: tuple[float, float], scale: float):
        _load_metal_symbols()
        device = MTLCreateSystemDefaultDevice()
        if device is None:
            raise RuntimeError("Metal default device unavailable")

        self._device = device
        self._queue = device.newCommandQueue()
        self._pipelines = _build_terraform_pipelines(device)
        self._uniform_buffer = device.newBufferWithLength_options_(_UNIFORM_SIZE, 0)
        self._ring_uniform_buffer = device.newBufferWithLength_options_(_UNIFORM_SIZE, 0)

        self._cards: list[CardInfo] = []
        self._scroll_offset_y: float = 0.0
        self._corner_radius: float = 10.0
        self._fill_width: float = 2.5
        self._interior_floor: float = 0.45
        self._pixel_width: int = 1
        self._pixel_height: int = 1
        self._scale: float = scale

        # Three CAMetalLayers: gold ring (back), blue ring (middle), fill (front)
        CAMetalLayerClass = objc.lookUpClass("CAMetalLayer")
        self._fill_layer = self._make_layer(CAMetalLayerClass, device)
        self._gold_layer = self._make_layer(CAMetalLayerClass, device)
        self._blue_layer = self._make_layer(CAMetalLayerClass, device)
        self._set_layer_geometry(frame_size, scale)

    def _make_layer(self, cls, device):
        layer = cls.layer()
        layer.setDevice_(device)
        layer.setPixelFormat_(_MTL_PIXEL_FORMAT_BGRA8_UNORM)
        layer.setFramebufferOnly_(True)
        layer.setOpaque_(False)
        return layer

    def layers(self):
        """Return (gold_layer, blue_layer, fill_layer) for insertion.

        Insert in this order so fill is on top, rings behind.
        """
        return (self._gold_layer, self._blue_layer, self._fill_layer)

    def layer(self):
        """Return the fill layer (primary, for compositing filter)."""
        return self._fill_layer

    def set_geometry(self, width_pt: float, height_pt: float, scale: float) -> None:
        """Update layer size (called on resize or content height change)."""
        self._set_layer_geometry((width_pt, height_pt), scale)

    def set_cards(self, cards: list[CardInfo]) -> None:
        """Update the card list. Clamped to MAX_CARDS."""
        self._cards = cards[:_MAX_CARDS]

    def set_scroll_offset(self, offset_px: float) -> None:
        """Update the vertical scroll offset in pixels."""
        self._scroll_offset_y = offset_px

    def draw_frame(self) -> bool:
        """Render one frame across all three layers."""
        # Pack fill uniforms
        fill_payload = self._pack_uniforms(self._cards)
        _copy_bytes_to_metal_buffer(self._uniform_buffer, fill_payload)

        # Pack ring uniforms — same card positions but with ring color/strength
        ring_payload = self._pack_uniforms(self._cards)  # positions shared
        _copy_bytes_to_metal_buffer(self._ring_uniform_buffer, ring_payload)

        ok = True
        # Draw fill layer
        ok = self._draw_layer(self._fill_layer, self._pipelines["fs_fill"],
                              self._uniform_buffer) and ok
        # Draw gold ring layer
        ok = self._draw_ring_layer(self._gold_layer, _GOLD_RGB) and ok
        # Draw blue ring layer
        ok = self._draw_ring_layer(self._blue_layer, _BLUE_RGB) and ok
        return ok

    def _pack_uniforms(self, cards: list[CardInfo]) -> bytes:
        values = [
            float(self._pixel_width),
            float(self._pixel_height),
            float(self._corner_radius),
            float(self._fill_width),
            float(self._interior_floor),
            float(self._scroll_offset_y),
            float(time.monotonic() % 1000.0),
            float(len(cards)),
        ]
        for i in range(_MAX_CARDS):
            if i < len(cards):
                c = cards[i]
                values.extend([c.x, c.y, c.width, c.height, c.r, c.g, c.b, c.alpha])
            else:
                values.extend([0.0] * _CARD_FLOATS)
        return struct.pack(_UNIFORM_LAYOUT, *values)

    def _draw_layer(self, layer, pipeline, uniform_buffer) -> bool:
        drawable = layer.nextDrawable()
        if drawable is None:
            return False

        rpd = objc.lookUpClass("MTLRenderPassDescriptor").renderPassDescriptor()
        att = rpd.colorAttachments().objectAtIndexedSubscript_(0)
        att.setTexture_(drawable.texture())
        att.setLoadAction_(_MTL_LOAD_ACTION_CLEAR)
        att.setStoreAction_(_MTL_STORE_ACTION_STORE)
        att.setClearColor_((0.0, 0.0, 0.0, 0.0))

        cmd = self._queue.commandBuffer()
        enc = cmd.renderCommandEncoderWithDescriptor_(rpd)
        enc.setRenderPipelineState_(pipeline)
        enc.setFragmentBuffer_offset_atIndex_(uniform_buffer, 0, 0)
        enc.drawPrimitives_vertexStart_vertexCount_(_MTL_PRIMITIVE_TYPE_TRIANGLE_STRIP, 0, 4)
        enc.endEncoding()
        cmd.presentDrawable_(drawable)
        cmd.commit()
        return True

    def _draw_ring_layer(self, layer, ring_rgb: tuple) -> bool:
        """Draw a chromatic ring layer with the given tint color."""
        # Build ring uniforms: same card positions, but color = ring tint,
        # alpha = ring strength. Only card[0] color is used by fs_ring.
        ring_cards = []
        for c in self._cards:
            ring_cards.append(CardInfo(
                x=c.x, y=c.y, width=c.width, height=c.height,
                r=ring_rgb[0], g=ring_rgb[1], b=ring_rgb[2],
                alpha=_RING_STRENGTH,
            ))
        payload = self._pack_uniforms(ring_cards)

        # Use a temporary buffer (reuse ring buffer)
        _copy_bytes_to_metal_buffer(self._ring_uniform_buffer, payload)
        return self._draw_layer(layer, self._pipelines["fs_ring"],
                                self._ring_uniform_buffer)

    # ── private ──

    def _set_layer_geometry(self, frame_size: tuple[float, float], scale: float) -> None:
        w_pt, h_pt = frame_size
        self._pixel_width = max(int(round(w_pt * scale)), 1)
        self._pixel_height = max(int(round(h_pt * scale)), 1)
        drawable_size = (float(self._pixel_width), float(self._pixel_height))
        offset = _CHROMA_OFFSET_PT

        for layer in (self._fill_layer, self._gold_layer, self._blue_layer):
            layer.setContentsScale_(scale)
            layer.setDrawableSize_(drawable_size)

        # Fill: centered
        self._fill_layer.setFrame_(((0, 0), (w_pt, h_pt)))
        # Gold: offset one direction
        self._gold_layer.setFrame_(((-offset, -offset * 0.5), (w_pt, h_pt)))
        # Blue: offset opposite direction
        self._blue_layer.setFrame_(((offset, offset * 0.5), (w_pt, h_pt)))

        self._fill_width = 2.5 * scale
        self._corner_radius = 10.0 * scale
        self._scale = scale


def metal_available() -> bool:
    """Check if Metal is available without creating a renderer."""
    return _metal_device_available()

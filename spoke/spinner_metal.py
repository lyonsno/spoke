"""Metal-accelerated spinner effect for the command overlay thinking state.

Renders a continuously animated SDF effect (concentric ripples + domain-warped
simplex noise) inside a small circular region. The effect is revealed
progressively by the radar-sweep cutout in the overlay fill — it's only
visible through the transparent hole the sweep carves.

The Metal layer sits behind the fill layer so it's only visible where the
fill has been cut away (alpha = 0 in the fill image).
"""

from __future__ import annotations

import ctypes
import logging
import math
import struct

logger = logging.getLogger(__name__)

# MSL shader source — concentric ripples + domain-warped simplex noise
_SHADER_SOURCE = """
#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float time;
    float sweep_angle;  // radians, CW from top
    float sweep_fill;   // 0..1
    float radius;       // circle radius in pixels
    float2 center;      // center in pixel coords
    float2 size;        // layer size in pixels
    float3 hue_color;   // current hue rotation color
    float brightness;   // screen brightness 0..1
};

// Simplex 2D noise — compact implementation
float2 _snoise_grad(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    float a = fract(sin(dot(i + float2(0,0), float2(127.1, 311.7))) * 43758.5453);
    float b = fract(sin(dot(i + float2(1,0), float2(127.1, 311.7))) * 43758.5453);
    float c = fract(sin(dot(i + float2(0,1), float2(127.1, 311.7))) * 43758.5453);
    float d = fract(sin(dot(i + float2(1,1), float2(127.1, 311.7))) * 43758.5453);
    float v = mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
    return float2(v, v);
}

float snoise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    float a = fract(sin(dot(i + float2(0,0), float2(127.1, 311.7))) * 43758.5453);
    float b = fract(sin(dot(i + float2(1,0), float2(127.1, 311.7))) * 43758.5453);
    float c = fract(sin(dot(i + float2(0,1), float2(127.1, 311.7))) * 43758.5453);
    float d = fract(sin(dot(i + float2(1,1), float2(127.1, 311.7))) * 43758.5453);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Domain-warped noise: warp the input coordinates by another noise field
float warped_noise(float2 p, float t) {
    float2 warp = float2(
        snoise(p * 2.0 + float2(t * 0.3, 0.0)),
        snoise(p * 2.0 + float2(0.0, t * 0.4))
    );
    return snoise(p + warp * 0.8 + float2(t * 0.1));
}

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex VertexOut spinner_vertex(
    uint vid [[vertex_id]]
) {
    // Full-screen quad from vertex ID
    float2 positions[] = {
        float2(-1, -1), float2(1, -1), float2(-1, 1),
        float2(-1, 1), float2(1, -1), float2(1, 1)
    };
    VertexOut out;
    out.position = float4(positions[vid], 0, 1);
    out.uv = positions[vid] * 0.5 + 0.5;
    return out;
}

fragment float4 spinner_fragment(
    VertexOut in [[stage_in]],
    constant Uniforms& u [[buffer(0)]]
) {
    // Pixel coordinates
    float2 px = in.uv * u.size;
    float2 d = px - u.center;
    float dist = length(d);

    // Circle mask with soft edge
    float circle = smoothstep(u.radius, u.radius - 2.0, dist);
    if (circle < 0.001) discard_fragment();

    // Normalized radial distance (0 at center, 1 at edge)
    float r = dist / max(u.radius, 1.0);

    // Angle (CW from top, matching sweep convention)
    float angle = atan2(d.x, d.y);
    if (angle < 0.0) angle += 2.0 * M_PI_F;

    // Only render in the swept (cutout) region
    float fill_end = u.sweep_fill * 2.0 * M_PI_F;
    float in_sweep = smoothstep(fill_end + 0.05, fill_end - 0.05, angle);
    if (in_sweep < 0.001) discard_fragment();

    float t = u.time;

    // === Effect 1: Concentric ripples ===
    // Time-varying frequency creates a sense of emanation
    float ripple_freq = 6.0 + 2.0 * sin(t * 0.7);
    float ripples = 0.5 + 0.5 * sin(r * ripple_freq - t * 3.0);
    // Fade ripples toward the edge for a natural boundary
    ripples *= (1.0 - r * r);

    // === Effect 2: Domain-warped simplex noise ===
    // Organic, fluid-like swirling motion
    float2 noise_coord = d / max(u.radius, 1.0) * 3.0;
    float noise = warped_noise(noise_coord, t);
    // Second octave for detail
    noise += 0.5 * warped_noise(noise_coord * 2.0, t * 1.3);
    noise *= 0.67; // normalize

    // === Composite ===
    // Blend ripples and noise — noise dominates center, ripples at edges
    float effect = mix(noise, ripples, r * 0.7);

    // Sweep-hand glow — bright edge at the boundary
    float angle_to_hand = abs(angle - fill_end);
    angle_to_hand = min(angle_to_hand, 2.0 * M_PI_F - angle_to_hand);
    float hand_glow = exp(-angle_to_hand / 0.12) * circle;

    // Color: use the hue rotation color, modulated by the effect
    float3 base_color = u.hue_color;
    // Darken toward the center, lighten at edges
    float3 color = base_color * (0.3 + 0.7 * effect);
    // Sweep hand: bright white-ish edge
    color += float3(0.8, 0.85, 0.9) * hand_glow * 0.6;

    // Alpha: gentle, not overwhelming — this is behind the overlay
    float alpha = circle * in_sweep * (0.15 + 0.25 * effect + 0.5 * hand_glow);

    // Brightness adaptation — less visible on bright screens
    alpha *= mix(1.0, 0.5, u.brightness);

    return float4(color * alpha, alpha);  // premultiplied
}
"""

# Uniform struct layout (matches MSL)
_UNIFORM_FMT = "f f f f  ff  ff  fff f"  # time, sweep_angle, sweep_fill, radius, center.xy, size.xy, hue_color.rgb, brightness
_UNIFORM_SIZE = struct.calcsize(_UNIFORM_FMT)


def _try_create_metal_device():
    """Try to create a Metal device. Returns (device, error_str)."""
    try:
        from Metal import MTLCreateSystemDefaultDevice
        device = MTLCreateSystemDefaultDevice()
        if device is None:
            return None, "MTLCreateSystemDefaultDevice returned None"
        return device, None
    except ImportError:
        return None, "Metal framework not available"
    except Exception as e:
        return None, str(e)


class SpinnerMetalLayer:
    """Manages a CAMetalLayer that renders the spinner SDF effect.

    The layer is positioned behind the overlay fill, so the effect is only
    visible through the cutout region carved by the sweep.
    """

    def __init__(self):
        self._device = None
        self._layer = None
        self._pipeline = None
        self._uniform_buffer = None
        self._command_queue = None
        self._ready = False

    def setup(self, parent_layer, frame, scale: float) -> bool:
        """Create the Metal layer and compile the shader.

        Returns True if Metal is available and setup succeeded.
        """
        device, err = _try_create_metal_device()
        if device is None:
            logger.info("Metal spinner not available: %s", err)
            return False

        self._device = device

        try:
            from Quartz import CAMetalLayer
            from Metal import (
                MTLCompileOptions,
                MTLRenderPipelineDescriptor,
            )

            # Create the Metal layer
            self._layer = CAMetalLayer.alloc().init()
            self._layer.setDevice_(device)
            self._layer.setPixelFormat_(80)  # MTLPixelFormatBGRA8Unorm
            self._layer.setFramebufferOnly_(True)
            self._layer.setContentsScale_(scale)
            self._layer.setFrame_(frame)
            self._layer.setOpaque_(False)  # transparent background

            # Compile shader
            options = MTLCompileOptions.alloc().init()
            library, error = device.newLibraryWithSource_options_error_(
                _SHADER_SOURCE, options, None
            )
            if library is None:
                logger.error("Metal shader compilation failed: %s", error)
                return False

            vertex_fn = library.newFunctionWithName_("spinner_vertex")
            fragment_fn = library.newFunctionWithName_("spinner_fragment")

            # Pipeline
            desc = MTLRenderPipelineDescriptor.alloc().init()
            desc.setVertexFunction_(vertex_fn)
            desc.setFragmentFunction_(fragment_fn)
            color_att = desc.colorAttachments().objectAtIndexedSubscript_(0)
            color_att.setPixelFormat_(80)  # MTLPixelFormatBGRA8Unorm
            # Premultiplied alpha blending
            color_att.setBlendingEnabled_(True)
            color_att.setSourceRGBBlendFactor_(1)  # one
            color_att.setDestinationRGBBlendFactor_(10)  # oneMinusSourceAlpha
            color_att.setSourceAlphaBlendFactor_(1)
            color_att.setDestinationAlphaBlendFactor_(10)

            pipeline, error = device.newRenderPipelineStateWithDescriptor_error_(
                desc, None
            )
            if pipeline is None:
                logger.error("Metal pipeline creation failed: %s", error)
                return False

            self._pipeline = pipeline
            self._command_queue = device.newCommandQueue()
            self._uniform_buffer = device.newBufferWithLength_options_(
                _UNIFORM_SIZE, 0  # MTLResourceStorageModeShared
            )

            # Insert behind everything in the parent layer
            parent_layer.insertSublayer_atIndex_(self._layer, 0)

            self._ready = True
            logger.info("Metal spinner layer ready (%.0fx%.0f @ %.1fx)",
                        frame[1][0], frame[1][1], scale)
            return True

        except Exception:
            logger.exception("Metal spinner setup failed")
            return False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def layer(self):
        return self._layer

    def set_frame(self, frame) -> None:
        """Update the layer frame (when overlay geometry changes)."""
        if self._layer is not None:
            self._layer.setFrame_(frame)

    def set_hidden(self, hidden: bool) -> None:
        """Show or hide the Metal layer."""
        if self._layer is not None:
            self._layer.setHidden_(hidden)

    def render(
        self,
        time: float,
        sweep_angle: float,
        sweep_fill: float,
        radius: float,
        center: tuple[float, float],
        size: tuple[float, float],
        hue_color: tuple[float, float, float],
        brightness: float,
    ) -> None:
        """Submit a render pass with the given uniforms."""
        if not self._ready:
            return

        try:
            # Update uniforms
            data = struct.pack(
                _UNIFORM_FMT,
                time, sweep_angle, sweep_fill, radius,
                center[0], center[1],
                size[0], size[1],
                hue_color[0], hue_color[1], hue_color[2],
                brightness,
            )
            buf_ptr = self._uniform_buffer.contents()
            ctypes.memmove(buf_ptr, data, len(data))

            # Get next drawable
            drawable = self._layer.nextDrawable()
            if drawable is None:
                return

            from Metal import MTLRenderPassDescriptor
            pass_desc = MTLRenderPassDescriptor.renderPassDescriptor()
            color_att = pass_desc.colorAttachments().objectAtIndexedSubscript_(0)
            color_att.setTexture_(drawable.texture())
            color_att.setLoadAction_(2)  # MTLLoadActionClear
            color_att.setClearColor_((0.0, 0.0, 0.0, 0.0))  # transparent
            color_att.setStoreAction_(1)  # MTLStoreActionStore

            cmd_buffer = self._command_queue.commandBuffer()
            encoder = cmd_buffer.renderCommandEncoderWithDescriptor_(pass_desc)
            encoder.setRenderPipelineState_(self._pipeline)
            encoder.setFragmentBuffer_offset_atIndex_(self._uniform_buffer, 0, 0)
            encoder.drawPrimitives_vertexStart_vertexCount_(3, 0, 6)  # triangle list
            encoder.endEncoding()

            cmd_buffer.presentDrawable_(drawable)
            cmd_buffer.commit()

        except Exception:
            logger.debug("Metal spinner render failed", exc_info=True)

    def teardown(self) -> None:
        """Remove the layer and release Metal resources."""
        if self._layer is not None:
            self._layer.removeFromSuperlayer()
            self._layer = None
        self._pipeline = None
        self._command_queue = None
        self._uniform_buffer = None
        self._device = None
        self._ready = False

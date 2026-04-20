"""Metal compute shader pipeline for optical shell warp.

Renders the capsule warp entirely on GPU — no CPU-side pixel copy.
SCK delivers frames as IOSurface-backed CVPixelBuffers.  We create
Metal texture views of those IOSurfaces, run the warp as a compute
shader, and present the result via CAMetalLayer.

The warp kernel is translated from the CIWarpKernel GLSL source
in backdrop_stream.py.
"""

from __future__ import annotations

import ctypes
import logging
import os

logger = logging.getLogger(__name__)

# Tuning constants — must stay in sync with backdrop_stream.py
_WARP_BLEED_ZONE_FRAC = 2.0
_WARP_CENTER_FLOOR = 0.80
_WARP_FIELD_EXPONENT = 0.35
_WARP_REMAP_BASE_EXP_SCALE = 0.98
_WARP_REMAP_BASE_EXP_FLOOR = 0.02
_WARP_REMAP_RIM_EXP = 0.1
_WARP_CURVEBOOST_CAP = 0.95
_WARP_CURVEBOOST_MAG_SCALE = 0.35
_WARP_CURVEBOOST_RING_DIVISOR = 240.0
_WARP_CURVEBOOST_RING_CAP = 0.55
_WARP_SPINE_PROXIMITY_BOOST = 1.5
_WARP_X_SQUEEZE = 2.5
_WARP_Y_SQUEEZE = 1.5
_WARP_EXTERIOR_MAG_STRENGTH = 0.3
_WARP_EXTERIOR_MAG_DECAY = 2.0


def _metal_shader_source() -> str:
    return f"""
#include <metal_stdlib>
using namespace metal;

struct WarpParams {{
    float width;
    float height;
    float rectWidth;
    float rectHeight;
    float cornerRadius;
    float coreMagnification;
    float bandWidth;
    float tailWidth;
    float ringAmplitudePoints;
    float tailAmplitudePoints;
}};

float sdCapsule(float2 p, float spineHalf, float radius) {{
    p.x = abs(p.x);
    float spine_dist = max(p.x - spineHalf, 0.0f);
    return length(float2(spine_dist, p.y)) - radius;
}}

float2 capsuleGradient(float2 p, float spineHalf) {{
    float clampedX = clamp(p.x, -spineHalf, spineHalf);
    float2 toP = p - float2(clampedX, 0.0f);
    float len = length(toP);
    return len > 1e-6f ? toP / len : float2(0.0f, 1.0f);
}}

float depthRemap(float inside01, float curveBoost) {{
    float x = clamp(inside01, 0.0f, 1.0f);
    float baseExp = max(1.0f - curveBoost * {_WARP_REMAP_BASE_EXP_SCALE}f, {_WARP_REMAP_BASE_EXP_FLOOR}f);
    float rimExp = mix({_WARP_REMAP_RIM_EXP}f, 1.0f, 1.0f - curveBoost);
    float exponent = mix(baseExp, rimExp, x * x);
    return pow(x, exponent);
}}

kernel void opticalShellWarp(
    texture2d<float, access::read> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant WarpParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {{
    if (gid.x >= (uint)params.width || gid.y >= (uint)params.height) return;

    float2 d = float2(gid.x, gid.y) + 0.5f;
    float2 c = float2(params.width * 0.5f, params.height * 0.5f);
    float2 p = d - c;
    float2 halfRect = float2(params.rectWidth * 0.5f, params.rectHeight * 0.5f);
    float capsuleRadius = max(halfRect.y, 1.0f);
    float spineHalf = max(halfRect.x - capsuleRadius, 0.0f);

    float capsuleSdf = sdCapsule(p, spineHalf, capsuleRadius);

    float bleedZone = capsuleRadius * {_WARP_BLEED_ZONE_FRAC}f;
    if (capsuleSdf > bleedZone) {{
        outTexture.write(inTexture.read(gid), gid);
        return;
    }}

    float curveBoost = min(
        {_WARP_CURVEBOOST_CAP}f,
        max(0.0f, (params.coreMagnification - 1.0f) * {_WARP_CURVEBOOST_MAG_SCALE}f)
            + min(params.ringAmplitudePoints / {_WARP_CURVEBOOST_RING_DIVISOR}f, {_WARP_CURVEBOOST_RING_CAP}f)
    );

    float distFromTip = max(spineHalf - abs(p.x), 0.0f);
    float spineProximity = spineHalf > 0.0f ? distFromTip / spineHalf : 0.0f;

    float rawField = clamp(1.0f + capsuleSdf / capsuleRadius, 0.0f, 1.0f);
    float localFloor = {_WARP_CENTER_FLOOR}f / (1.0f + spineProximity * {_WARP_SPINE_PROXIMITY_BOOST}f);
    float field01 = mix(localFloor, 1.0f, pow(rawField, {_WARP_FIELD_EXPONENT}f));
    float sourceField01 = 1.0f - depthRemap(1.0f - field01, curveBoost);
    float scale = sourceField01 / field01;

    float scaleX = pow(max(scale, 0.0f), {_WARP_X_SQUEEZE}f);
    float scaleY = pow(max(scale, 0.0f), {_WARP_Y_SQUEEZE}f);
    float2 warped = c + p * float2(scaleX, scaleY);

    float exteriorT = max(capsuleSdf, 0.0f);
    float seamRamp = smoothstep(0.0f, 2.0f, exteriorT);
    float magDecay = exp(-exteriorT / capsuleRadius * {_WARP_EXTERIOR_MAG_DECAY}f);
    float2 n = capsuleGradient(p, spineHalf);
    float tipDist = max(abs(p.x) - spineHalf, 0.0f);
    float tipAtten = 1.0f - smoothstep(0.0f, capsuleRadius * 0.8f, tipDist);
    float mag = {_WARP_EXTERIOR_MAG_STRENGTH}f * capsuleRadius * seamRamp * magDecay * tipAtten;
    float2 result = warped - n * mag;
    result = clamp(result, float2(0.0f), float2(params.width, params.height));

    // Bilinear sample from source
    uint2 src = uint2(clamp(result, float2(0.0f), float2(params.width - 1.0f, params.height - 1.0f)));
    outTexture.write(inTexture.read(src), gid);
}}
"""


class MetalWarpPipeline:
    """GPU-only warp pipeline using Metal compute shaders."""

    def __init__(self):
        try:
            import objc
            from Foundation import NSBundle
        except Exception as exc:
            raise RuntimeError("Metal warp requires PyObjC") from exc

        metal_bundle = NSBundle.bundleWithPath_("/System/Library/Frameworks/Metal.framework")
        objc.loadBundleFunctions(
            metal_bundle, globals(), [("MTLCreateSystemDefaultDevice", b"@")]
        )
        self._device = MTLCreateSystemDefaultDevice()
        if self._device is None:
            raise RuntimeError("No Metal device")

        self._command_queue = self._device.newCommandQueue()
        if self._command_queue is None:
            raise RuntimeError("Failed to create Metal command queue")

        # Compile shader
        source = _metal_shader_source()
        result = self._device.newLibraryWithSource_options_error_(source, None, None)
        if isinstance(result, tuple):
            library, error = result
        else:
            library, error = result, None
        if library is None:
            raise RuntimeError(f"Metal shader compilation failed: {error}")

        kernel_fn = library.newFunctionWithName_("opticalShellWarp")
        if kernel_fn is None:
            raise RuntimeError("opticalShellWarp function not found in Metal library")

        result = self._device.newComputePipelineStateWithFunction_error_(kernel_fn, None)
        if isinstance(result, tuple):
            pipeline, error = result
        else:
            pipeline, error = result, None
        if pipeline is None:
            raise RuntimeError(f"Metal compute pipeline creation failed: {error}")

        self._pipeline = pipeline
        self._params_buffer = None
        self._output_texture = None
        self._output_texture_size = None
        logger.info("Metal warp pipeline created (threadgroup=%d)", pipeline.maxTotalThreadsPerThreadgroup())

    def warp_iosurface(self, input_surface, *, width, height, shell_config):
        """Run the warp on an IOSurface and return a new IOSurface with the result."""
        import objc

        # Create input texture from IOSurface
        from Foundation import NSDictionary
        tex_desc = objc.lookUpClass("MTLTextureDescriptor").texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            80,  # MTLPixelFormatBGRA8Unorm
            width, height, False,
        )
        tex_desc.setUsage_(1 | 2)  # read | write

        input_texture = self._device.newTextureWithDescriptor_iosurface_plane_(
            tex_desc, input_surface, 0,
        )
        if input_texture is None:
            return None

        # Create or reuse output texture
        if self._output_texture_size != (width, height):
            self._output_texture = self._device.newTextureWithDescriptor_(tex_desc)
            self._output_texture_size = (width, height)

        # Params buffer
        import struct
        params_data = struct.pack(
            "10f",
            float(width),
            float(height),
            float(shell_config.get("content_width_points", width)),
            float(shell_config.get("content_height_points", height)),
            float(shell_config.get("corner_radius_points", 16.0)),
            float(shell_config.get("core_magnification", 1.0)),
            float(shell_config.get("band_width_points", 12.0)),
            float(shell_config.get("tail_width_points", 9.0)),
            float(shell_config.get("ring_amplitude_points", 12.0)),
            float(shell_config.get("tail_amplitude_points", 4.0)),
        )
        params_buffer = self._device.newBufferWithBytes_length_options_(
            params_data, len(params_data), 0,
        )

        # Encode and dispatch
        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(self._pipeline)
        encoder.setTexture_atIndex_(input_texture, 0)
        encoder.setTexture_atIndex_(self._output_texture, 1)
        encoder.setBuffer_offset_atIndex_(params_buffer, 0, 0)

        # Threadgroup size
        w = self._pipeline.threadExecutionWidth()
        h = self._pipeline.maxTotalThreadsPerThreadgroup() // w
        threadgroup_size = (w, min(h, height), 1)
        grid_size = (width, height, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
        encoder.endEncoding()

        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        return self._output_texture

    def warp_to_drawable(self, input_surface, drawable, *, width, height, shell_config):
        """Run the warp and blit the result to a CAMetalLayer drawable.

        Returns True if the drawable was presented, False on failure.
        The entire pipeline stays on GPU — no CPU pixel copy.
        """
        import objc
        import struct

        # Input texture from IOSurface
        tex_desc = objc.lookUpClass("MTLTextureDescriptor").texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            80,  # MTLPixelFormatBGRA8Unorm
            width, height, False,
        )
        tex_desc.setUsage_(1)  # read

        input_texture = self._device.newTextureWithDescriptor_iosurface_plane_(
            tex_desc, input_surface, 0,
        )
        if input_texture is None:
            return False

        # Output is the drawable's texture
        output_texture = drawable.texture()
        if output_texture is None:
            return False

        # Params
        out_w = output_texture.width()
        out_h = output_texture.height()
        params_data = struct.pack(
            "10f",
            float(out_w), float(out_h),
            float(shell_config.get("content_width_points", out_w)),
            float(shell_config.get("content_height_points", out_h)),
            float(shell_config.get("corner_radius_points", 16.0)),
            float(shell_config.get("core_magnification", 1.0)),
            float(shell_config.get("band_width_points", 12.0)),
            float(shell_config.get("tail_width_points", 9.0)),
            float(shell_config.get("ring_amplitude_points", 12.0)),
            float(shell_config.get("tail_amplitude_points", 4.0)),
        )
        params_buffer = self._device.newBufferWithBytes_length_options_(
            params_data, len(params_data), 0,
        )

        # Encode
        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(self._pipeline)
        encoder.setTexture_atIndex_(input_texture, 0)
        encoder.setTexture_atIndex_(output_texture, 1)
        encoder.setBuffer_offset_atIndex_(params_buffer, 0, 0)

        w = self._pipeline.threadExecutionWidth()
        h = self._pipeline.maxTotalThreadsPerThreadgroup() // w
        threadgroup_size = (w, min(h, out_h), 1)
        grid_size = (out_w, out_h, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
        encoder.endEncoding()

        command_buffer.presentDrawable_(drawable)
        command_buffer.commit()
        return True

    @property
    def device(self):
        return self._device

    def create_metal_layer(self, width, height, scale=2.0):
        """Create a CAMetalLayer configured for warp output."""
        import objc
        objc.loadBundle(
            "QuartzCore", globals(),
            bundle_path="/System/Library/Frameworks/QuartzCore.framework",
        )
        CAMetalLayer = objc.lookUpClass("CAMetalLayer")
        layer = CAMetalLayer.alloc().init()
        layer.setDevice_(self._device)
        layer.setPixelFormat_(80)  # MTLPixelFormatBGRA8Unorm
        layer.setFramebufferOnly_(False)
        layer.setDrawableSize_((width * scale, height * scale))
        layer.setContentsScale_(scale)
        layer.setOpaque_(False)
        return layer


_metal_warp_pipeline = None


def get_metal_warp_pipeline():
    global _metal_warp_pipeline
    if _metal_warp_pipeline is not None:
        return _metal_warp_pipeline or None
    try:
        _metal_warp_pipeline = MetalWarpPipeline()
    except Exception:
        logger.info("Metal warp pipeline unavailable", exc_info=True)
        _metal_warp_pipeline = False
    return _metal_warp_pipeline or None

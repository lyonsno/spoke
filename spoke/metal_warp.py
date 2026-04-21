"""Metal compute shader pipeline for the fullscreen optical shell compositor.

Renders the capsule warp entirely on GPU — no CPU-side pixel copy.
SCK delivers frames as IOSurface-backed CVPixelBuffers.  We create
Metal texture views of those IOSurfaces, run the warp as a compute
shader, and present the result via CAMetalLayer.

The render loop is driven by CVDisplayLink, not the SCK frame
callback.  SCK stores the latest IOSurface atomically; the display
link callback acquires nextDrawable(), dispatches the compute
shader, and presents.  This prevents buffer pool starvation — the
SCK handler queue never blocks on drawable availability.

This module owns the fullscreen Metal tuning directly. The older
CI/backdrop path remains as a fallback surface with its own tuning.
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import struct
import threading
import time

logger = logging.getLogger(__name__)

# Tuning constants for the fullscreen Metal compositor path.
_WARP_BLEED_ZONE_FRAC = 0.8
_WARP_CENTER_FLOOR = 0.94
_WARP_FIELD_EXPONENT = 0.25
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
_WARP_ALIAS_MIP_BIAS_DEADZONE = 0.12
_WARP_ALIAS_MIP_BIAS_SCALE = 1.35
_WARP_ALIAS_MIP_BIAS_MAX = 2.0

_TEMPORAL_BLEND_FACTOR = 0.25  # EMA blend: 25% new frame, 75% accumulator


def _warp_alias_mip_bias(scale_x: float, scale_y: float) -> float:
    """Return extra mip bias for strongly deformed warp regions.

    Small departures from identity should stay crisp. Once the warp stretches
    or compresses by a noticeable fraction of an octave, add a controlled mip
    bias so high-frequency source content does not alias into hard stair-steps.
    """
    sx = max(abs(float(scale_x)), 1e-4)
    sy = max(abs(float(scale_y)), 1e-4)
    warp_octaves = max(abs(math.log2(sx)), abs(math.log2(sy)))
    if warp_octaves <= _WARP_ALIAS_MIP_BIAS_DEADZONE:
        return 0.0
    bias = (warp_octaves - _WARP_ALIAS_MIP_BIAS_DEADZONE) * _WARP_ALIAS_MIP_BIAS_SCALE
    return min(bias, _WARP_ALIAS_MIP_BIAS_MAX)


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
    float centerX;      // capsule center X in pixels (0 = use width/2)
    float centerY;      // capsule center Y in pixels (0 = use height/2)
    float gridOffsetX;  // dispatch grid origin X (for bounding-box dispatch)
    float gridOffsetY;  // dispatch grid origin Y
    float temporalBlend; // EMA blend factor (0 = keep previous, 1 = fully new)
    float minBrightness; // floor for interior pixel luminance (0 = no floor)
}};

float sdStadium(float2 p, float spineHalfX, float spineHalfY, float radius) {{
    float dx = max(abs(p.x) - spineHalfX, 0.0f);
    float dy = max(abs(p.y) - spineHalfY, 0.0f);
    return length(float2(dx, dy)) - radius;
}}

float depthRemap(float inside01, float curveBoost) {{
    float x = clamp(inside01, 0.0f, 1.0f);
    float baseExp = max(1.0f - curveBoost * {_WARP_REMAP_BASE_EXP_SCALE}f, {_WARP_REMAP_BASE_EXP_FLOOR}f);
    float rimExp = mix({_WARP_REMAP_RIM_EXP}f, 1.0f, 1.0f - curveBoost);
    float exponent = mix(baseExp, rimExp, x * x);
    return pow(x, exponent);
}}

constexpr sampler bilinearSampler(
    coord::pixel,
    address::clamp_to_edge,
    filter::linear
);

constexpr sampler mipSampler(
    coord::normalized,
    address::clamp_to_edge,
    filter::linear,
    mip_filter::linear
);

kernel void opticalShellWarp(
    texture2d<float, access::sample> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    texture2d<float, access::read> accumIn [[texture(2)]],
    texture2d<float, access::write> accumOut [[texture(3)]],
    constant WarpParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {{
    // Map dispatch gid to screen pixel via grid offset
    uint2 pixel = uint2(gid.x + (uint)params.gridOffsetX, gid.y + (uint)params.gridOffsetY);
    if (pixel.x >= (uint)params.width || pixel.y >= (uint)params.height) return;

    float2 d = float2(pixel.x, pixel.y) + 0.5f;
    float2 c = float2(
        params.centerX > 0.0f ? params.centerX : params.width * 0.5f,
        params.centerY > 0.0f ? params.centerY : params.height * 0.5f
    );
    float2 p = d - c;
    float2 halfRect = float2(params.rectWidth * 0.5f, params.rectHeight * 0.5f);
    float capsuleRadius = params.cornerRadius > 0.0f
        ? min(params.cornerRadius, halfRect.y)
        : min(halfRect.y, halfRect.x * 0.35f);
    capsuleRadius = max(capsuleRadius, 1.0f);
    float spineHalfX = max(halfRect.x - capsuleRadius, 0.0f);
    float spineHalfY = max(halfRect.y - capsuleRadius, 0.0f);

    float capsuleSdf = sdStadium(p, spineHalfX, spineHalfY, capsuleRadius);

    float bleedZone = capsuleRadius * {_WARP_BLEED_ZONE_FRAC}f;
    if (capsuleSdf > bleedZone) {{
        // Outside warp zone: pass through unwarped content.
        outTexture.write(inTexture.sample(bilinearSampler, d), pixel);
        return;
    }}

    float curveBoost = min(
        {_WARP_CURVEBOOST_CAP}f,
        max(0.0f, (params.coreMagnification - 1.0f) * {_WARP_CURVEBOOST_MAG_SCALE}f)
            + min(params.ringAmplitudePoints / {_WARP_CURVEBOOST_RING_DIVISOR}f, {_WARP_CURVEBOOST_RING_CAP}f)
    );

    float distFromTip = max(spineHalfX - abs(p.x), 0.0f);
    float spineProximity = spineHalfX > 0.0f ? distFromTip / spineHalfX : 0.0f;

    float rawField = clamp(1.0f + capsuleSdf / capsuleRadius, 0.0f, 1.0f);
    float localFloor = {_WARP_CENTER_FLOOR}f / (1.0f + spineProximity * {_WARP_SPINE_PROXIMITY_BOOST}f);
    float field01 = mix(localFloor, 1.0f, pow(rawField, {_WARP_FIELD_EXPONENT}f));
    float sourceField01 = 1.0f - depthRemap(1.0f - field01, curveBoost);
    float scale = sourceField01 / field01;

    float scaleX = pow(max(scale, 0.0f), {_WARP_X_SQUEEZE}f);
    float scaleY = pow(max(scale, 0.0f), {_WARP_Y_SQUEEZE}f);
    float2 warped = c + p * float2(scaleX, scaleY);

    float2 result = warped;
    if (capsuleSdf > 0.0f) {{
        // Exterior: blend toward boundary warp scale
        float probeRaw = clamp(1.0f - 0.15f, 0.0f, 1.0f);
        float probeField = mix(localFloor, 1.0f, pow(probeRaw, {_WARP_FIELD_EXPONENT}f));
        float probeSource = 1.0f - depthRemap(1.0f - probeField, curveBoost);
        float probeScale = probeSource / probeField;
        float probeSX = pow(max(probeScale, 0.0f), {_WARP_X_SQUEEZE}f);
        float probeSY = pow(max(probeScale, 0.0f), {_WARP_Y_SQUEEZE}f);

        float t = 1.0f - smoothstep(0.0f, 40.0f, capsuleSdf);
        t = t * t;
        float2 boundaryWarped = c + p * float2(probeSX, probeSY);
        result = mix(d, boundaryWarped, t * 0.50f);
    }}
    result = clamp(result, float2(0.0f), float2(params.width, params.height));

    // Depth-dependent blur via mipmap LOD.
    // The warp capsule is inflated by capsuleRadius beyond the fill
    // boundary, so blur should only start ramping once we're past
    // that inflation zone (i.e., inside the fill pill).
    float pixelsInside = max(-capsuleSdf - capsuleRadius * 0.5f, 0.0f);
    float baseMipLod = clamp(pixelsInside / 30.0f, 0.0f, 1.0f) * 6.0f;
    float warpAliasOctaves = max(
        abs(log2(max(abs(scaleX), 1e-4f))),
        abs(log2(max(abs(scaleY), 1e-4f)))
    );
    float warpAliasBias = clamp(
        (warpAliasOctaves - {_WARP_ALIAS_MIP_BIAS_DEADZONE}f) * {_WARP_ALIAS_MIP_BIAS_SCALE}f,
        0.0f,
        {_WARP_ALIAS_MIP_BIAS_MAX}f
    );
    float mipLod = clamp(baseMipLod + warpAliasBias, 0.0f, 6.0f);

    float2 samplePt = clamp(result, float2(0.5f), float2(params.width - 0.5f, params.height - 0.5f));
    float4 warpedColor;
    if (mipLod < 0.1f) {{
        warpedColor = inTexture.sample(bilinearSampler, samplePt);
    }} else {{
        float2 normPt = samplePt / float2(params.width, params.height);
        warpedColor = inTexture.sample(mipSampler, normPt, level(mipLod));
    }}

    // Temporal accumulation: EMA blend tied to blur depth.
    // High mip LOD (deep interior) = heavy temporal smoothing.
    // Low/zero mip LOD (near boundary) = almost no temporal.
    // Exterior = no temporal at all (pass through).
    if (capsuleSdf <= 0.0f) {{
        // Interior: temporal weight scales with mip LOD (blur depth).
        // mipLod 0 (rim, no blur) → weight 1.0 (no temporal).
        // mipLod 6 (deep interior, max blur) → weight = params.temporalBlend.
        float mipFrac = clamp(mipLod / 6.0f, 0.0f, 1.0f);
        float temporalWeight = mix(1.0f, params.temporalBlend, mipFrac);
        float4 prev = accumIn.read(gid);
        float4 blended = mix(prev, warpedColor, temporalWeight);
        accumOut.write(blended, gid);
        outTexture.write(blended, pixel);
    }} else {{
        // Exterior: no temporal smoothing — fully responsive.
        accumOut.write(warpedColor, gid);
        outTexture.write(warpedColor, pixel);
    }}
}}
"""


def _create_metal_buffer(device, data: bytes):
    """Create a Metal buffer from bytes, working around PyObjC bridging issues."""
    try:
        return device.newBufferWithBytes_length_options_(data, len(data), 0)
    except (ValueError, TypeError):
        pass
    # Fallback: use newBufferWithLength then copy
    try:
        buf = device.newBufferWithLength_options_(len(data), 0)
        if buf is None:
            return None
        # Copy data into the buffer via ctypes
        contents_ptr = buf.contents()
        if contents_ptr is None:
            return None
        ctypes.memmove(contents_ptr, data, len(data))
        return buf
    except Exception:
        logger.debug("_create_metal_buffer fallback failed", exc_info=True)
        return None


_WARP_PARAMS_SIZE = struct.calcsize("16f")


def _pack_warp_params(width, height, shell_config, grid_offset_x=0.0, grid_offset_y=0.0):
    """Pack WarpParams struct for the Metal compute shader."""
    return struct.pack(
        "16f",
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
        float(shell_config.get("center_x", 0.0)),
        float(shell_config.get("center_y", 0.0)),
        float(grid_offset_x),
        float(grid_offset_y),
        float(shell_config.get("temporal_blend", _TEMPORAL_BLEND_FACTOR)),
        float(shell_config.get("min_brightness", 0.0)),
    )


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
        self._output_texture = None
        self._output_texture_size = None
        self._mip_texture = None
        self._mip_texture_size = None
        self._accum_textures = [None, None]  # ping-pong accumulation buffers
        self._accum_texture_size = None
        self._accum_index = 0
        self._accum_generation = 0  # incremented on resize for atomicity

        # Cache threadgroup dimensions — these are hardware constants
        self._thread_exec_width = pipeline.threadExecutionWidth()
        self._max_tg_height = pipeline.maxTotalThreadsPerThreadgroup() // self._thread_exec_width

        # Pre-allocate reusable params buffer (avoids alloc per frame)
        self._params_buffer = self._device.newBufferWithLength_options_(
            _WARP_PARAMS_SIZE, 0,  # MTLResourceStorageModeShared
        )

        logger.info("Metal warp pipeline created (threadgroup=%dx%d)",
                     self._thread_exec_width, self._max_tg_height)

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

        # Use separate accum textures with temporal_blend=1.0 (full replace)
        self._ensure_accum_textures(width, height)
        shell_config_copy = dict(shell_config)
        shell_config_copy["temporal_blend"] = 1.0
        params_data = _pack_warp_params(width, height, shell_config_copy)
        params_buffer = _create_metal_buffer(self._device, params_data)

        # Encode and dispatch
        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(self._pipeline)
        encoder.setTexture_atIndex_(input_texture, 0)
        encoder.setTexture_atIndex_(self._output_texture, 1)
        encoder.setTexture_atIndex_(self._accum_textures[0], 2)
        encoder.setTexture_atIndex_(self._accum_textures[1], 3)
        encoder.setBuffer_offset_atIndex_(params_buffer, 0, 0)

        threadgroup_size = (self._thread_exec_width, min(self._max_tg_height, height), 1)
        grid_size = (width, height, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
        encoder.endEncoding()

        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        return self._output_texture

    def _ensure_accum_textures(self, width, height):
        """Create or resize ping-pong accumulation textures for temporal AA.

        Uses a generation counter for atomicity — callers snapshot the
        generation before dispatching and check it hasn't changed before
        flipping the index.  This prevents a resize between dispatch and
        flip from corrupting the ping-pong state.
        """
        if self._accum_texture_size == (width, height):
            return
        import objc
        desc = objc.lookUpClass("MTLTextureDescriptor").texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            80, width, height, False,
        )
        desc.setUsage_(1 | 2)  # read | write
        self._accum_textures = [
            self._device.newTextureWithDescriptor_(desc),
            self._device.newTextureWithDescriptor_(desc),
        ]
        self._accum_texture_size = (width, height)
        self._accum_index = 0
        self._accum_generation += 1
        if self._accum_textures[0] is None or self._accum_textures[1] is None:
            logger.warning("Failed to create accumulation textures %dx%d", width, height)
            self._accum_textures = [None, None]

    def reset_temporal_state(self):
        """Force next frame to fully replace the accumulator.

        Call this when the compositor restarts so the first frame doesn't
        blend with stale content from a previous session.
        """
        self._accum_generation += 1

    def warp_to_drawable(self, input_surface, drawable, *, width, height, shell_config):
        """Run the warp and blit the result to a CAMetalLayer drawable.

        Returns True if the drawable was presented, False on failure.
        The entire pipeline stays on GPU — no CPU pixel copy.
        """
        import objc

        # Input texture from IOSurface (single-level, no mipmaps)
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

        out_w = output_texture.width()
        out_h = output_texture.height()
        in_w = input_texture.width()
        in_h = input_texture.height()

        # Dimension mismatch = skip warp, just present
        if in_w != out_w or in_h != out_h or in_w <= 0 or in_h <= 0:
            command_buffer = self._command_queue.commandBuffer()
            command_buffer.presentDrawable_(drawable)
            command_buffer.commit()
            return True

        # Create or reuse a mipmapped texture for blur LOD sampling.
        if self._mip_texture is None or self._mip_texture_size != (in_w, in_h):
            mip_desc = objc.lookUpClass("MTLTextureDescriptor").texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
                80, in_w, in_h, True,
            )
            mip_desc.setUsage_(1 | 2)
            self._mip_texture = self._device.newTextureWithDescriptor_(mip_desc)
            self._mip_texture_size = (in_w, in_h)
            if self._mip_texture is None:
                logger.warning("Failed to create mipmapped texture %dx%d", in_w, in_h)

        command_buffer = self._command_queue.commandBuffer()

        # Full-screen blit: input → output.  Guarantees the entire
        # drawable is written before presentation (no tearing on 120Hz).
        blit = command_buffer.blitCommandEncoder()
        blit.copyFromTexture_toTexture_(input_texture, output_texture)
        if self._mip_texture is not None:
            origin = (0, 0, 0)
            size = (in_w, in_h, 1)
            try:
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_(
                    input_texture, 0, 0, origin, size,
                    self._mip_texture, 0, 0, origin,
                )
            except Exception:
                try:
                    blit.copyFromTexture_toTexture_(input_texture, self._mip_texture)
                except Exception:
                    pass
            blit.generateMipmapsForTexture_(self._mip_texture)
        blit.endEncoding()

        # Pass 2: compute warp over capsule bounding box only
        cx = shell_config.get("center_x", out_w * 0.5)
        cy = shell_config.get("center_y", out_h * 0.5)
        rect_w = shell_config.get("content_width_points", out_w)
        rect_h = shell_config.get("content_height_points", out_h)
        capsule_r = max(rect_h * 0.5, 1.0)
        bleed = capsule_r * _WARP_BLEED_ZONE_FRAC
        box_x0 = max(int(cx - rect_w * 0.5 - bleed), 0)
        box_y0 = max(int(cy - rect_h * 0.5 - bleed), 0)
        box_x1 = min(int(cx + rect_w * 0.5 + bleed) + 1, out_w)
        box_y1 = min(int(cy + rect_h * 0.5 + bleed) + 1, out_h)
        box_w = box_x1 - box_x0
        box_h = box_y1 - box_y0

        if box_w > 0 and box_h > 0:
            # Accum textures are bounding-box-sized to minimize VRAM.
            self._ensure_accum_textures(box_w, box_h)
            gen_before = self._accum_generation
            accum_read = self._accum_textures[self._accum_index]
            accum_write = self._accum_textures[1 - self._accum_index]

            # First frame after accum resize: fully replace to avoid
            # blending with uninitialized texture data.
            warp_config = shell_config
            if gen_before != getattr(self, "_accum_last_used_gen", -1):
                warp_config = dict(shell_config)
                warp_config["temporal_blend"] = 1.0

            # Update reusable params buffer via memmove (no alloc per frame)
            params_data = _pack_warp_params(
                out_w, out_h, warp_config,
                grid_offset_x=float(box_x0),
                grid_offset_y=float(box_y0),
            )
            if self._params_buffer is not None:
                contents_ptr = self._params_buffer.contents()
                if contents_ptr is not None:
                    ctypes.memmove(contents_ptr, params_data, len(params_data))
                    params_buffer = self._params_buffer
                else:
                    params_buffer = _create_metal_buffer(self._device, params_data)
            else:
                params_buffer = _create_metal_buffer(self._device, params_data)

            if params_buffer is None:
                command_buffer.presentDrawable_(drawable)
                command_buffer.commit()
                return True  # blit-only frame

            encoder = command_buffer.computeCommandEncoder()
            encoder.setComputePipelineState_(self._pipeline)
            warp_input = self._mip_texture if self._mip_texture is not None else input_texture
            encoder.setTexture_atIndex_(warp_input, 0)
            encoder.setTexture_atIndex_(output_texture, 1)
            if accum_read is not None and accum_write is not None:
                encoder.setTexture_atIndex_(accum_read, 2)
                encoder.setTexture_atIndex_(accum_write, 3)
            encoder.setBuffer_offset_atIndex_(params_buffer, 0, 0)

            threadgroup_size = (self._thread_exec_width, min(self._max_tg_height, box_h), 1)
            grid_size = (box_w, box_h, 1)
            encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
            encoder.endEncoding()

            # Flip accumulation buffer — only if generation hasn't changed
            # (a resize between dispatch and here would invalidate the flip).
            if self._accum_generation == gen_before:
                self._accum_index = 1 - self._accum_index
                self._accum_last_used_gen = gen_before

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


class MetalDisplayLinkRenderer:
    """Display-link-driven Metal render loop for optical shell warp.

    Decouples SCK frame ingestion from drawable presentation:
    - SCK callback calls ``submit_iosurface()`` — atomic swap, never blocks.
    - CVDisplayLink fires at display refresh rate, acquires ``nextDrawable()``,
      dispatches the compute shader, presents.

    This eliminates buffer pool starvation: ``nextDrawable()`` is never called
    from the SCK handler queue.
    """

    def __init__(self, pipeline: MetalWarpPipeline, metal_layer):
        self._pipeline = pipeline
        self._metal_layer = metal_layer
        self._lock = threading.Lock()
        self._latest_iosurface = None
        self._latest_width = 0
        self._latest_height = 0
        self._shell_config = None
        self._running = False
        self._display_link = None
        self._frame_count = 0
        self._last_report_time = 0.0
        self._interval_frame_count = 0
        self._presented_count = 0
        self._last_drawable_size = (0, 0)

        # Configure layer for non-blocking presentation
        if hasattr(metal_layer, "setPresentsWithTransaction_"):
            metal_layer.setPresentsWithTransaction_(False)
        # Explicitly request triple buffering (default, but be explicit)
        if hasattr(metal_layer, "setMaximumDrawableCount_"):
            metal_layer.setMaximumDrawableCount_(3)

    def submit_iosurface(self, iosurface, *, width: int, height: int) -> None:
        """Called from SCK handler queue — must never block."""
        with self._lock:
            self._latest_iosurface = iosurface
            self._latest_width = width
            self._latest_height = height

    def set_shell_config(self, config: dict | None) -> None:
        with self._lock:
            self._shell_config = dict(config) if config else None

    def start(self) -> bool:
        """Start the CVDisplayLink render loop."""
        if self._running:
            return True
        try:
            self._display_link = _create_display_link(self._on_display_link)
            if self._display_link is None:
                return False
            _start_display_link(self._display_link)
            self._running = True
            self._last_report_time = time.monotonic()
            logger.info("MetalDisplayLinkRenderer: started")
            return True
        except Exception:
            logger.info("MetalDisplayLinkRenderer: failed to start", exc_info=True)
            return False

    def stop(self) -> None:
        """Stop the CVDisplayLink render loop."""
        if not self._running:
            return
        self._running = False
        dl = self._display_link
        self._display_link = None
        if dl is not None:
            try:
                _stop_display_link(dl)
            except Exception:
                logger.debug("MetalDisplayLinkRenderer: failed to stop display link", exc_info=True)
        with self._lock:
            self._latest_iosurface = None
        logger.info("MetalDisplayLinkRenderer: stopped (%d presented)", self._presented_count)

    @property
    def is_running(self) -> bool:
        return self._running

    def _on_display_link(self) -> None:
        """Called from the CVDisplayLink thread at display refresh rate."""
        if not self._running:
            return

        # Grab latest frame — non-blocking
        with self._lock:
            iosurface = self._latest_iosurface
            w = self._latest_width
            h = self._latest_height
            config = self._shell_config

        if iosurface is None or w <= 0 or h <= 0 or config is None:
            return

        self._frame_count += 1
        self._interval_frame_count += 1

        # Log FPS every 5 seconds
        now = time.monotonic()
        elapsed = now - self._last_report_time
        if elapsed >= 5.0:
            fps = self._interval_frame_count / elapsed if elapsed > 0 else 0
            logger.info(
                "Metal render: %d ticks in %.1fs (%.1f fps), %d presented",
                self._interval_frame_count, elapsed, fps, self._presented_count,
            )
            self._last_report_time = now
            self._interval_frame_count = 0

        try:
            # Update drawable size only when dimensions change
            if self._last_drawable_size != (w, h):
                self._metal_layer.setDrawableSize_((w, h))
                self._last_drawable_size = (w, h)

            # Non-blocking drawable acquisition — if all drawables are busy,
            # skip this frame.  The next vsync will try again.
            drawable = self._metal_layer.nextDrawable()
            if drawable is None:
                if self._frame_count <= 5:
                    logger.info(
                        "Metal render tick[%d]: nextDrawable returned None "
                        "(drawableSize=%s, bounds=%s, frame=%s)",
                        self._frame_count,
                        self._metal_layer.drawableSize() if hasattr(self._metal_layer, "drawableSize") else "?",
                        self._metal_layer.bounds() if hasattr(self._metal_layer, "bounds") else "?",
                        self._metal_layer.frame() if hasattr(self._metal_layer, "frame") else "?",
                    )
                return

            if self._frame_count <= 3:
                logger.info("Metal render tick[%d]: drawable acquired, dispatching warp %dx%d", self._frame_count, w, h)

            if self._pipeline.warp_to_drawable(
                iosurface, drawable, width=w, height=h, shell_config=config,
            ):
                self._presented_count += 1
            elif self._frame_count <= 5:
                logger.info("Metal render tick[%d]: warp_to_drawable returned False", self._frame_count)
        except Exception:
            if self._frame_count <= 10:
                logger.info("Metal render tick[%d] failed", self._frame_count, exc_info=True)


# ---------------------------------------------------------------------------
# CVDisplayLink helpers — ctypes wrappers around CoreVideo C API
# ---------------------------------------------------------------------------

_cv_lib = None


def _get_cv_lib():
    global _cv_lib
    if _cv_lib is not None:
        return _cv_lib
    _cv_lib = ctypes.cdll.LoadLibrary(
        "/System/Library/Frameworks/CoreVideo.framework/CoreVideo"
    )
    return _cv_lib


# CVDisplayLink callback signature:
#   CVReturn (*)(CVDisplayLinkRef, const CVTimeStamp*, const CVTimeStamp*,
#                CVOptionFlags, CVOptionFlags*, void*)
_CVDisplayLinkOutputCallback = ctypes.CFUNCTYPE(
    ctypes.c_int32,       # CVReturn
    ctypes.c_void_p,      # CVDisplayLinkRef
    ctypes.c_void_p,      # inNow
    ctypes.c_void_p,      # inOutputTime
    ctypes.c_uint64,      # flagsIn
    ctypes.POINTER(ctypes.c_uint64),  # flagsOut
    ctypes.c_void_p,      # displayLinkContext
)

# Must prevent GC of callback closures while display link is active
_active_callbacks: dict[int, tuple] = {}


def _create_display_link(python_callback) -> ctypes.c_void_p | None:
    """Create a CVDisplayLink that calls ``python_callback()`` each vsync."""
    cv = _get_cv_lib()

    dl_ref = ctypes.c_void_p()
    err = cv.CVDisplayLinkCreateWithActiveCGDisplays(ctypes.byref(dl_ref))
    if err != 0:
        logger.warning("CVDisplayLinkCreateWithActiveCGDisplays failed: %d", err)
        return None

    def _c_callback(_dl, _now, _out_time, _flags_in, _flags_out, _ctx):
        try:
            python_callback()
        except Exception:
            logger.debug("CVDisplayLink callback exception", exc_info=True)
        return 0  # kCVReturnSuccess

    c_func = _CVDisplayLinkOutputCallback(_c_callback)

    cv.CVDisplayLinkSetOutputCallback.argtypes = [
        ctypes.c_void_p, _CVDisplayLinkOutputCallback, ctypes.c_void_p,
    ]
    cv.CVDisplayLinkSetOutputCallback.restype = ctypes.c_int32
    err = cv.CVDisplayLinkSetOutputCallback(dl_ref, c_func, None)
    if err != 0:
        logger.warning("CVDisplayLinkSetOutputCallback failed: %d", err)
        return None

    # Pin the callback and closure to prevent GC
    _active_callbacks[dl_ref.value] = (c_func, _c_callback)

    return dl_ref


def _start_display_link(dl_ref: ctypes.c_void_p) -> None:
    cv = _get_cv_lib()
    cv.CVDisplayLinkStart.argtypes = [ctypes.c_void_p]
    cv.CVDisplayLinkStart.restype = ctypes.c_int32
    err = cv.CVDisplayLinkStart(dl_ref)
    if err != 0:
        raise RuntimeError(f"CVDisplayLinkStart failed: {err}")


def _stop_display_link(dl_ref: ctypes.c_void_p) -> None:
    cv = _get_cv_lib()
    cv.CVDisplayLinkStop.argtypes = [ctypes.c_void_p]
    cv.CVDisplayLinkStop.restype = ctypes.c_int32
    cv.CVDisplayLinkStop(dl_ref)
    # Release callback reference
    _active_callbacks.pop(dl_ref.value, None)


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

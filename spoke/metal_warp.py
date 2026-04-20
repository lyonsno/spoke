"""Metal compute shader pipeline for optical shell warp.

Renders the capsule warp entirely on GPU — no CPU-side pixel copy.
SCK delivers frames as IOSurface-backed CVPixelBuffers.  We create
Metal texture views of those IOSurfaces, run the warp as a compute
shader, and present the result via CAMetalLayer.

The render loop is driven by CVDisplayLink, not the SCK frame
callback.  SCK stores the latest IOSurface atomically; the display
link callback acquires nextDrawable(), dispatches the compute
shader, and presents.  This prevents buffer pool starvation — the
SCK handler queue never blocks on drawable availability.

The warp kernel is translated from the CIWarpKernel GLSL source
in backdrop_stream.py.
"""

from __future__ import annotations

import ctypes
import logging
import os
import struct
import threading
import time

logger = logging.getLogger(__name__)

# Tuning constants — must stay in sync with backdrop_stream.py
_WARP_BLEED_ZONE_FRAC = 0.5
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

    // Depth-dependent bilateral blur: sharp at boundary (0-10%% depth),
    // ramps to full blur toward center.  1px tap offsets — the warp
    // magnification amplifies the effect naturally.
    float interiorDepth = clamp(-capsuleSdf / capsuleRadius, 0.0f, 1.0f);
    // Later onset (15%), longer ramp — more blur deep inside, less near edge
    float blurT = smoothstep(0.15f, 0.50f, interiorDepth);

    float2 samplePt = clamp(result, float2(0.5f), float2(params.width - 0.5f, params.height - 0.5f));
    float4 centerColor = inTexture.read(uint2(samplePt));

    float4 finalColor;
    if (blurT < 0.01f) {{
        // Sharp — single tap at boundary
        finalColor = centerColor;
    }} else {{
        // Bilateral blur: 1px fixed offsets, depth-dependent blend.
        // Weight by color similarity to preserve edges.
        float4 acc = centerColor * 2.0f;
        float totalWeight = 2.0f;
        float colorSigma2 = 0.04f;  // tighter = more edge-preserving

        for (int dy = -1; dy <= 1; dy++) {{
            for (int dx = -1; dx <= 1; dx++) {{
                if (dx == 0 && dy == 0) continue;
                float2 sp = clamp(samplePt + float2(float(dx), float(dy)),
                                  float2(0.0f), float2(params.width - 1.0f, params.height - 1.0f));
                float4 s = inTexture.read(uint2(sp));
                float3 diff = s.rgb - centerColor.rgb;
                float colorDist = dot(diff, diff);
                float w = exp(-colorDist / colorSigma2);
                acc += s * w;
                totalWeight += w;
            }}
        }}
        float4 blurred = acc / totalWeight;
        finalColor = mix(centerColor, blurred, blurT);
    }}

    // Perceptual value clamp: keep luminance in the 15-85%% range.
    // Prevents blown-out whites and crushed blacks in the warped
    // interior.  Uses approximate sRGB luminance.
    float luma = dot(finalColor.rgb, float3(0.2126f, 0.7152f, 0.0722f));
    float clampedLuma = clamp(luma, 0.15f, 0.85f);
    if (luma > 0.001f) {{
        float lumaScale = clampedLuma / luma;
        // Blend toward clamped based on depth — full clamping at center,
        // none at boundary.
        float clampStrength = smoothstep(0.05f, 0.30f, interiorDepth);
        float scale_factor = mix(1.0f, lumaScale, clampStrength);
        finalColor.rgb *= scale_factor;
    }}

    outTexture.write(finalColor, gid);
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
            # Update drawable size to match capture dimensions
            self._metal_layer.setDrawableSize_((w, h))

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

"""Scene capture for screen context.

Captures the frontmost app's active window (or full screen as fallback),
runs OCR to extract text blocks with bounding boxes, optionally collects
shallow AX hints, and stores the result as a local SceneCapture artifact
with stable refs that downstream tools can resolve to exact text.

See docs/screen-context-v1.md for the design.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

# ── Default configuration ────────────────────────────────────────

# Linear scale factor for model-facing image (1/3 keeps enough detail for VLMs
# to read UI text reliably while cutting vision-token prefill ~9× vs full res).
_DEFAULT_SCALE = 1.0 / 3.0

# Minimum dimension (px) below which we skip downsampling.
_MIN_DOWNSAMPLE_DIM = 800

# Hard timeout for AX queries (seconds).
_AX_TIMEOUT = 0.5

# Default max cached captures.
_DEFAULT_MAX_CAPTURES = 10


# ── Dataclasses ──────────────────────────────────────────────────


@dataclass
class OCRBlock:
    """A single OCR-recognized text region."""

    ref: str
    text: str
    bbox: tuple[float, float, float, float]  # (x, y, width, height) in pixels
    confidence: float | None = None


@dataclass
class AXHint:
    """A shallow accessibility hint for the focused element."""

    ref: str
    role: str
    label: str | None = None
    value: str | None = None


@dataclass
class SceneCapture:
    """A captured scene with OCR blocks and optional AX hints."""

    scene_ref: str
    created_at: float
    scope: Literal["active_window", "screen"]
    app_name: str | None
    bundle_id: str | None
    window_title: str | None
    image_path: str
    image_size: tuple[int, int]
    model_image_size: tuple[int, int]
    ocr_text: str
    ocr_blocks: list[OCRBlock]
    ax_hints: list[AXHint]
    model_image_path: str | None = None
    model_image_media_type: str | None = None


# ── Scene ref generation ─────────────────────────────────────────


def _generate_scene_ref() -> str:
    """Generate a unique scene ref."""
    return f"scene-{uuid.uuid4().hex[:8]}"


# ── OCR extraction ───────────────────────────────────────────────


def _extract_ocr_blocks(
    observations: list,
    scene_ref: str,
    image_width: int,
    image_height: int,
) -> list[OCRBlock]:
    """Convert Vision VNRecognizedTextObservation results to OCRBlocks.

    Vision returns bounding boxes in normalized coordinates (0-1, origin
    at bottom-left). We convert to pixel coordinates (origin top-left)
    for the consumer.
    """
    blocks: list[OCRBlock] = []
    for i, obs in enumerate(observations):
        candidates = obs.topCandidates_(1)
        if not candidates:
            continue

        candidate = candidates[0]
        text = candidate.string()
        confidence = candidate.confidence()

        # Bounding box: normalized, bottom-left origin
        bbox = obs.boundingBox()
        nx, ny, nw, nh = (
            bbox.origin.x,
            bbox.origin.y,
            bbox.size.width,
            bbox.size.height,
        )

        # Convert to pixel coords, top-left origin
        px = nx * image_width
        py = (1.0 - ny - nh) * image_height  # flip Y
        pw = nw * image_width
        ph = nh * image_height

        blocks.append(
            OCRBlock(
                ref=f"{scene_ref}:block-{i}",
                text=text,
                bbox=(px, py, pw, ph),
                confidence=confidence,
            )
        )

    return blocks


def _run_ocr(
    cg_image,
    image_width: int,
    image_height: int,
    scene_ref: str,
    *,
    accurate: bool = False,
):
    """Run Vision OCR on a CGImage and return (ocr_text, ocr_blocks).

    OCR is disabled — it lags the interface noticeably (especially with
    the fullscreen compositor) and has never been reliable enough to
    justify the cost.  Returns empty results immediately.
    """
    return "", []
    try:
        from Vision import (
            VNImageRequestHandler,
            VNRecognizeTextRequest,
            VNRequestTextRecognitionLevelAccurate,
            VNRequestTextRecognitionLevelFast,
        )

        handler = VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, None
        )
        request = VNRecognizeTextRequest.alloc().init()
        if accurate:
            request.setRecognitionLevel_(VNRequestTextRecognitionLevelAccurate)
            request.setUsesLanguageCorrection_(True)
        else:
            request.setRecognitionLevel_(VNRequestTextRecognitionLevelFast)
            request.setUsesLanguageCorrection_(False)

        success, error = handler.performRequests_error_([request], None)
        if not success:
            logger.warning("Vision OCR failed: %s", error)
            return "", []

        results = request.results() or []
        blocks = _extract_ocr_blocks(results, scene_ref, image_width, image_height)
        ocr_text = " ".join(b.text for b in blocks)
        return ocr_text, blocks

    except Exception:
        logger.warning("OCR extraction failed", exc_info=True)
        return "", []


# ── AX hints ─────────────────────────────────────────────────────


def _get_focused_ax_info() -> tuple[str, str | None, str | None] | None:
    """Return (role, label, value) of the focused AX element, or None.

    Reuses the ctypes AX bindings from focus_check.py.
    """
    from spoke.focus_check import (
        _cf,
        _cfstr,
        _cfstr_to_python,
        _hi,
        _kAXErrorSuccess,
    )

    if _cf is None or _hi is None:
        return None

    system_wide = _hi.AXUIElementCreateSystemWide()
    if not system_wide:
        return None

    import ctypes

    try:
        focus_attr = _cfstr(b"AXFocusedUIElement")
        if not focus_attr:
            return None
        try:
            focused = ctypes.c_void_p()
            err = _hi.AXUIElementCopyAttributeValue(
                system_wide, focus_attr, ctypes.byref(focused)
            )
            if err != _kAXErrorSuccess or not focused.value:
                return None

            try:
                role = _ax_get_string(focused, b"AXRole")
                if not role:
                    return None
                label = _ax_get_string(focused, b"AXDescription") or _ax_get_string(
                    focused, b"AXTitle"
                )
                value = _ax_get_string(focused, b"AXValue")
                return (role, label, value)
            finally:
                _cf.CFRelease(focused)
        finally:
            _cf.CFRelease(focus_attr)
    finally:
        _cf.CFRelease(system_wide)


def _ax_get_string(element, attr_name: bytes) -> str | None:
    """Get a string attribute from an AX element. Returns None on failure."""
    import ctypes

    from spoke.focus_check import (
        _cf,
        _cfstr,
        _cfstr_to_python,
        _hi,
        _kAXErrorSuccess,
    )

    attr = _cfstr(attr_name)
    if not attr:
        return None
    try:
        value = ctypes.c_void_p()
        err = _hi.AXUIElementCopyAttributeValue(element, attr, ctypes.byref(value))
        if err != _kAXErrorSuccess or not value.value:
            return None
        try:
            return _cfstr_to_python(value)
        except Exception:
            return None
        finally:
            _cf.CFRelease(value)
    finally:
        _cf.CFRelease(attr)


def _ax_get_element(element, attr_name: bytes):
    """Get an AX element attribute. Caller must CFRelease the returned value."""
    import ctypes

    from spoke.focus_check import (
        _cf,
        _cfstr,
        _hi,
        _kAXErrorSuccess,
    )

    attr = _cfstr(attr_name)
    if not attr:
        return None
    try:
        value = ctypes.c_void_p()
        err = _hi.AXUIElementCopyAttributeValue(element, attr, ctypes.byref(value))
        if err != _kAXErrorSuccess or not value.value:
            return None
        return value
    finally:
        _cf.CFRelease(attr)


def _ax_get_pid(element) -> int | None:
    """Return the pid for an AX element, or None when unavailable."""
    import ctypes

    from spoke.focus_check import _hi, _kAXErrorSuccess

    if _hi is None or not hasattr(_hi, "AXUIElementGetPid"):
        return None

    _hi.AXUIElementGetPid.restype = ctypes.c_int32
    _hi.AXUIElementGetPid.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
    ]

    pid = ctypes.c_int32()
    err = _hi.AXUIElementGetPid(element, ctypes.byref(pid))
    if err != _kAXErrorSuccess:
        return None
    return int(pid.value)


def _get_focused_window_hint() -> tuple[int | None, str | None] | None:
    """Return (focused_app_pid, focused_window_title) from AX when available."""
    from spoke.focus_check import _cf, _hi

    if _cf is None or _hi is None:
        return None

    system_wide = _hi.AXUIElementCreateSystemWide()
    if not system_wide:
        return None

    try:
        focused_app = _ax_get_element(system_wide, b"AXFocusedApplication")
        if not focused_app:
            return None
        try:
            pid = _ax_get_pid(focused_app)
            focused_window = _ax_get_element(focused_app, b"AXFocusedWindow")
            try:
                title = _ax_get_string(focused_window, b"AXTitle") if focused_window else None
            finally:
                if focused_window:
                    _cf.CFRelease(focused_window)
            return (pid, title)
        finally:
            _cf.CFRelease(focused_app)
    finally:
        _cf.CFRelease(system_wide)


def _pick_target_window(
    window_list: list,
    *,
    preferred_pid: int | None,
    preferred_title: str | None,
    fallback_pid: int | None,
    my_pid: int,
):
    """Choose the best CG window candidate from front-to-back ordered windows."""

    def _reasonable(win) -> bool:
        bounds = win.get("kCGWindowBounds")
        return bool(
            bounds
            and bounds.get("Height", 0) > 50
            and bounds.get("Width", 0) > 50
        )

    def _for_pid(pid: int | None):
        if pid is None:
            return []
        return [
            win for win in window_list
            if win.get("kCGWindowOwnerPID") == pid and _reasonable(win)
        ]

    candidates = _for_pid(preferred_pid)
    if preferred_title:
        for win in candidates:
            if win.get("kCGWindowName") == preferred_title:
                return win
    if candidates:
        return candidates[0]

    fallback_candidates = _for_pid(fallback_pid)
    if fallback_candidates:
        return fallback_candidates[0]

    for win in window_list:
        win_pid = win.get("kCGWindowOwnerPID", 0)
        if win_pid == my_pid or not _reasonable(win):
            continue
        return win
    return None


def _collect_ax_hints(scene_ref: str, timeout: float = _AX_TIMEOUT) -> list[AXHint]:
    """Collect shallow AX hints with a hard timeout.

    Best-effort: returns [] on any failure or timeout. Must never block
    the capture.
    """
    result: list[AXHint] = []
    exc_holder: list[BaseException] = []

    def _query():
        try:
            info = _get_focused_ax_info()
            if info is not None:
                role, label, value = info
                result.append(
                    AXHint(
                        ref=f"{scene_ref}:focus",
                        role=role,
                        label=label,
                        value=value,
                    )
                )
        except Exception as e:
            exc_holder.append(e)

    t = threading.Thread(target=_query, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        logger.debug("AX hint collection timed out after %.1fs", timeout)
        return []

    if exc_holder:
        logger.debug("AX hint collection failed: %s", exc_holder[0])
        return []

    return result


# ── Downsampling ─────────────────────────────────────────────────


def _downsample_size(
    width: int, height: int, scale: float = _DEFAULT_SCALE
) -> tuple[int, int]:
    """Compute the downsampled image size.

    If the image is already small (both dimensions below _MIN_DOWNSAMPLE_DIM),
    returns the original size to avoid unnecessary quality loss.
    """
    if width <= _MIN_DOWNSAMPLE_DIM and height <= _MIN_DOWNSAMPLE_DIM:
        return (width, height)

    return (max(1, round(width * scale)), max(1, round(height * scale)))


def _downsample_image(cg_image, scale: float = _DEFAULT_SCALE):
    """Return a downsampled copy of *cg_image* using CoreGraphics.

    If the image is already small (both dims below _MIN_DOWNSAMPLE_DIM),
    returns the original unchanged.
    """
    from Quartz import (
        CGBitmapContextCreate,
        CGBitmapContextCreateImage,
        CGContextDrawImage,
        CGImageGetWidth,
        CGImageGetHeight,
        CGImageGetAlphaInfo,
        CGImageGetColorSpace,
        CGRectMake,
        kCGImageAlphaPremultipliedLast,
    )

    src_w = CGImageGetWidth(cg_image)
    src_h = CGImageGetHeight(cg_image)
    dst_w, dst_h = _downsample_size(src_w, src_h, scale)
    if (dst_w, dst_h) == (src_w, src_h):
        return cg_image

    color_space = CGImageGetColorSpace(cg_image)
    ctx = CGBitmapContextCreate(
        None, dst_w, dst_h, 8, dst_w * 4,
        color_space, kCGImageAlphaPremultipliedLast,
    )
    if ctx is None:
        logger.warning("Failed to create bitmap context for downsampling")
        return cg_image

    CGContextDrawImage(ctx, CGRectMake(0, 0, dst_w, dst_h), cg_image)
    return CGBitmapContextCreateImage(ctx)


# ── Image capture ────────────────────────────────────────────────


def _capture_active_window():
    """Capture the frontmost app's active window as a CGImage.

    Returns (cg_image, app_name, bundle_id, window_title) or None on failure.
    """
    try:
        from AppKit import NSRunningApplication, NSWorkspace
        from Quartz import (
            CGRectNull,
            CGWindowListCopyWindowInfo,
            CGWindowListCreateImage,
            kCGWindowImageBoundsIgnoreFraming,
            kCGWindowListExcludeDesktopElements,
            kCGWindowListOptionIncludingWindow,
            kCGWindowListOptionOnScreenOnly,
        )

        # Get the on-screen window list (ordered front-to-back by the
        # window server, so the first match is the topmost).
        window_list = CGWindowListCopyWindowInfo(
            kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
            0,  # kCGNullWindowID
        )
        if not window_list:
            logger.debug("CGWindowListCopyWindowInfo returned empty")
            return None

        # Try to identify the frontmost app via NSWorkspace, but prefer AX
        # focused-window hints when they exist because the workspace heuristic
        # has a standing tendency to stick on WezTerm.
        workspace_pid = None
        app_pid = None
        app_name = None
        bundle_id = None
        window_title = None
        front_app_name = None
        front_bundle_id = None
        try:
            workspace = NSWorkspace.sharedWorkspace()
            front_app = workspace.frontmostApplication()
            if front_app is not None:
                workspace_pid = front_app.processIdentifier()
                app_pid = workspace_pid
                app_name = front_app.localizedName()
                bundle_id = front_app.bundleIdentifier()
                front_app_name = app_name
                front_bundle_id = bundle_id
                logger.debug(
                    "Frontmost app: %s (pid=%s, bundle=%s)",
                    app_name, app_pid, bundle_id,
                )
        except Exception:
            logger.debug("NSWorkspace frontmost app lookup failed", exc_info=True)

        my_pid = os.getpid()
        ax_pid = None
        ax_title = None
        try:
            ax_hint = _get_focused_window_hint()
            if ax_hint is not None:
                ax_pid, ax_title = ax_hint
        except Exception:
            logger.debug("AX focused-window lookup failed", exc_info=True)

        if ax_pid is not None:
            app_pid = ax_pid

        target_window = _pick_target_window(
            window_list,
            preferred_pid=ax_pid,
            preferred_title=ax_title,
            fallback_pid=workspace_pid,
            my_pid=my_pid,
        )

        if target_window is None:
            logger.debug("No suitable window found in window list")
            return None

        window_id = target_window.get("kCGWindowNumber", 0)
        if not window_title:
            window_title = target_window.get("kCGWindowName")
        target_pid = target_window.get("kCGWindowOwnerPID")
        target_app = None
        if target_pid not in (None, workspace_pid):
            try:
                target_app = NSRunningApplication.runningApplicationWithProcessIdentifier_(
                    target_pid
                )
            except Exception:
                logger.debug(
                    "NSRunningApplication lookup failed for pid=%s",
                    target_pid,
                    exc_info=True,
                )
        if target_app is not None:
            app_name = target_app.localizedName() or target_window.get("kCGWindowOwnerName")
            bundle_id = target_app.bundleIdentifier()
        else:
            app_name = target_window.get("kCGWindowOwnerName") or app_name
        target_layer = target_window.get("kCGWindowLayer")
        bounds = target_window.get("kCGWindowBounds") or {}

        logger.info(
            "capture_context: selected window id=%s owner_pid=%s owner=%r title=%r "
            "layer=%s bounds=%sx%s workspace_pid=%s workspace_app=%r bundle=%r "
            "ax_pid=%s ax_title=%r",
            window_id,
            target_pid,
            app_name,
            window_title,
            target_layer,
            bounds.get("Width"),
            bounds.get("Height"),
            workspace_pid,
            front_app_name,
            front_bundle_id,
            ax_pid,
            ax_title,
        )

        # Capture just this window
        image = CGWindowListCreateImage(
            CGRectNull,
            kCGWindowListOptionIncludingWindow,
            window_id,
            kCGWindowImageBoundsIgnoreFraming,
        )
        if image is None:
            logger.debug("CGWindowListCreateImage returned None for window %s", window_id)
            return None

        return (image, app_name, bundle_id, window_title)

    except Exception:
        logger.warning("Active window capture failed", exc_info=True)
        return None


def _capture_screen():
    """Capture the full main screen as a CGImage.

    Returns (cg_image, None, None, None) or None on failure.
    """
    try:
        from Quartz import (
            CGRectInfinite,
            CGWindowListCreateImage,
            kCGNullWindowID,
            kCGWindowListOptionOnScreenOnly,
        )

        image = CGWindowListCreateImage(
            CGRectInfinite, kCGWindowListOptionOnScreenOnly, kCGNullWindowID, 0
        )
        if image is None:
            return None

        return (image, None, None, None)

    except Exception:
        logger.warning("Screen capture failed", exc_info=True)
        return None


def _save_image(cg_image, path: str) -> bool:
    """Save a CGImage to a PNG file. Returns True on success."""
    try:
        from Quartz import (
            CGImageDestinationAddImage,
            CGImageDestinationCreateWithURL,
            CGImageDestinationFinalize,
        )
        from Foundation import NSURL

        url = NSURL.fileURLWithPath_(path)
        dest = CGImageDestinationCreateWithURL(url, "public.png", 1, None)
        if dest is None:
            return False
        CGImageDestinationAddImage(dest, cg_image, None)
        return CGImageDestinationFinalize(dest)

    except Exception:
        logger.warning("Failed to save image to %s", path, exc_info=True)
        return False


def _image_dimensions(cg_image) -> tuple[int, int]:
    """Return (width, height) of a CGImage."""
    from Quartz import CGImageGetWidth, CGImageGetHeight

    return (CGImageGetWidth(cg_image), CGImageGetHeight(cg_image))


# ── Cache ────────────────────────────────────────────────────────


class SceneCaptureCache:
    """In-memory cache of recent SceneCapture artifacts.

    Bounded by max_captures; evicts oldest-first when full.
    """

    def __init__(self, max_captures: int = _DEFAULT_MAX_CAPTURES):
        self._max = max_captures
        self._captures: OrderedDict[str, SceneCapture] = OrderedDict()

    def store(self, capture: SceneCapture) -> None:
        """Store a capture, evicting the oldest if at capacity."""
        if capture.scene_ref in self._captures:
            self._captures.move_to_end(capture.scene_ref)
            self._captures[capture.scene_ref] = capture
            return

        while len(self._captures) >= self._max:
            evicted_ref, evicted = self._captures.popitem(last=False)
            # Clean up the image file
            try:
                if os.path.exists(evicted.image_path):
                    os.remove(evicted.image_path)
                if evicted.model_image_path and os.path.exists(evicted.model_image_path):
                    os.remove(evicted.model_image_path)
            except OSError:
                pass
            logger.debug("Evicted scene capture %s", evicted_ref)

        self._captures[capture.scene_ref] = capture

    def get(self, scene_ref: str) -> SceneCapture | None:
        """Retrieve a capture by scene_ref."""
        return self._captures.get(scene_ref)

    def list_refs(self) -> list[str]:
        """List all cached scene refs, oldest first."""
        return list(self._captures.keys())

    def resolve_block(self, block_ref: str) -> str | None:
        """Resolve a scene_block ref to its OCR text.

        block_ref format: "scene-abc:block-N"
        """
        # Split into scene ref and block part
        parts = block_ref.rsplit(":", 1)
        if len(parts) != 2:
            return None

        scene_ref = parts[0]
        capture = self._captures.get(scene_ref)
        if capture is None:
            return None

        for block in capture.ocr_blocks:
            if block.ref == block_ref:
                return block.text

        return None

    def resolve_ax_hint(self, hint_ref: str) -> str | None:
        """Resolve an ax_hint ref to its text (value preferred, then label).

        hint_ref format: "scene-abc:focus" (or other hint identifiers)
        """
        parts = hint_ref.rsplit(":", 1)
        if len(parts) != 2:
            return None

        scene_ref = parts[0]
        capture = self._captures.get(scene_ref)
        if capture is None:
            return None

        for hint in capture.ax_hints:
            if hint.ref == hint_ref:
                return hint.value or hint.label

        return None


# ── Top-level capture entry point ────────────────────────────────


def capture_context(
    scope: Literal["active_window", "screen"] = "active_window",
    cache: SceneCaptureCache | None = None,
    cache_dir: str | None = None,
    skip_ocr: bool | None = None,
) -> SceneCapture | None:
    """Capture a scene and return a SceneCapture artifact.

    Tries active_window first (if scope="active_window"), falling back
    to full screen on failure. Runs OCR and collects AX hints.

    Returns None if capture fails entirely.
    """
    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.expanduser("~/Library/Application Support/Spoke"), "scene_cache"
        )
    os.makedirs(cache_dir, exist_ok=True)

    scene_ref = _generate_scene_ref()

    # Capture
    t0 = time.perf_counter()
    result = None
    actual_scope: Literal["active_window", "screen"] = scope

    if scope == "active_window":
        result = _capture_active_window()

    if result is None:
        result = _capture_screen()
        actual_scope = "screen"

    if result is None:
        logger.warning("All capture methods failed")
        return None

    cg_image, app_name, bundle_id, window_title = result
    t_capture = time.perf_counter()
    logger.info("capture_context: screen grab %.0fms", (t_capture - t0) * 1000)

    # Dimensions and downsampling
    width, height = _image_dimensions(cg_image)
    model_w, model_h = _downsample_size(width, height)

    # Save image
    image_path = os.path.join(cache_dir, f"{scene_ref}.png")
    if not _save_image(cg_image, image_path):
        logger.warning("Failed to save capture image")
        return None

    model_image_path = os.path.join(cache_dir, f"{scene_ref}-model.png")
    model_image_media_type = "image/png"
    try:
        model_image = _downsample_image(cg_image)
        if not _save_image(model_image, model_image_path):
            logger.warning("Failed to save model-facing capture image")
            model_image_path = None
            model_image_media_type = None
    except Exception:
        logger.warning("Failed to build model-facing capture image", exc_info=True)
        model_image_path = None
        model_image_media_type = None
    t_save = time.perf_counter()
    logger.info("capture_context: image save %.0fms", (t_save - t_capture) * 1000)

    # OCR (skippable — when the model receives the image directly, OCR text
    # is redundant and its latency + token cost can be avoided).
    if skip_ocr is None:
        skip_ocr = os.environ.get("SPOKE_SKIP_OCR", "").lower() in ("1", "true", "yes")

    if skip_ocr:
        ocr_text, ocr_blocks = "", []
        logger.info("capture_context: OCR skipped")
    else:
        ocr_text, ocr_blocks = _run_ocr(cg_image, width, height, scene_ref)
        logger.info("capture_context: OCR %.0fms (%d blocks)", (time.perf_counter() - t_save) * 1000, len(ocr_blocks))
    t_ocr = time.perf_counter()

    # AX hints (best-effort, with timeout)
    ax_hints = _collect_ax_hints(scene_ref)
    t_ax = time.perf_counter()
    logger.info("capture_context: AX hints %.0fms (%d hints)", (t_ax - t_ocr) * 1000, len(ax_hints))

    capture = SceneCapture(
        scene_ref=scene_ref,
        created_at=time.time(),
        scope=actual_scope,
        app_name=app_name,
        bundle_id=bundle_id,
        window_title=window_title,
        image_path=image_path,
        image_size=(width, height),
        model_image_size=(model_w, model_h),
        ocr_text=ocr_text,
        ocr_blocks=ocr_blocks,
        ax_hints=ax_hints,
        model_image_path=model_image_path,
        model_image_media_type=model_image_media_type,
    )

    if cache is not None:
        cache.store(capture)

    return capture

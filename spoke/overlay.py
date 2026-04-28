"""Frosted transcription overlay.

A semi-transparent, system-font overlay at the bottom of the screen that shows
live transcription text during recording. Fades in when recording starts,
updates as preview transcriptions arrive (with typewriter effect), and fades
out after final injection. Text opacity breathes with voice amplitude.
"""

from __future__ import annotations

import colorsys
import logging
import math
import os
import threading

import objc
from AppKit import (
    NSBackingStoreBuffered,
    NSColor,
    NSFont,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSMutableAttributedString,
    NSPanel,
    NSScreen,
    NSScrollView,
    NSTextView,
    NSView,
    NSWindow,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskNonactivatingPanel,
)
from Foundation import NSMakeRect, NSObject, NSTimer
from Quartz import CAGradientLayer, CALayer, CAShapeLayer, CGPathCreateWithRoundedRect, CGAffineTransformIdentity

from .dedup import ontology_term_spans

logger = logging.getLogger(__name__)


def _start_overlay_fill_worker(work):
    thread = threading.Thread(target=work, name="spoke-overlay-fill", daemon=True)
    thread.start()
    return thread


def _post_overlay_result_to_main(target, selector: str, payload: dict) -> None:
    poster = getattr(target, "performSelectorOnMainThread_withObject_waitUntilDone_", None)
    if callable(poster):
        poster(selector, payload, False)
        return
    getattr(target, selector.replace(":", "_"))(payload)

def _env(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None else default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v not in {"0", "false", "False", "no", "off"}


_OVERLAY_WIDTH = 600.0
_OVERLAY_HEIGHT = 80.0
_OVERLAY_WINDOW_LEVEL = 25
_OVERLAY_BOTTOM_MARGIN = _env("SPOKE_PREVIEW_OVERLAY_BOTTOM_MARGIN", 80.0)
_OVERLAY_CORNER_RADIUS = 16.0
_OVERLAY_MAX_HEIGHT = _env("SPOKE_PREVIEW_OVERLAY_MAX_HEIGHT", 300.0)
_COMMAND_OVERLAY_BOTTOM_MARGIN = _env("SPOKE_COMMAND_OVERLAY_BOTTOM_MARGIN", 300.0)
_EXPAND_UPWARD = _env_bool("SPOKE_PREVIEW_EXPAND_UPWARD", True)
_FONT_SIZE = 16.0
_FADE_IN_S = 0.4  # fast ease-in so the overlay feels ready as soon as it appears
_FADE_OUT_S = 0.315  # 75% longer fade keeps the preview legible through fast handoff
_FADE_STEPS = 12  # number of steps for manual fade animation
_TYPEWRITER_INTERVAL = 0.02 / 0.75  # seconds between characters (~37.5 chars/sec)


def _scale_color_saturation(
    color: tuple[float, float, float], factor: float
) -> tuple[float, float, float]:
    """Scale an RGB color's saturation while keeping its hue and value stable."""
    hue, saturation, value = colorsys.rgb_to_hsv(*color)
    return colorsys.hsv_to_rgb(hue, min(max(saturation * factor, 0.0), 1.0), value)

_TEXT_ALPHA_MIN = _env("SPOKE_TEXT_ALPHA_MIN", 0.066)
_TEXT_ALPHA_MAX = _env("SPOKE_TEXT_ALPHA_MAX", 0.75)
_TEXT_ALPHA_MAX_LIGHT = 0.90
_TEXT_AMP_SATURATION = _env("SPOKE_TEXT_AMP_SATURATION", 0.05)  # sensitive but not pegged
_BG_ALPHA_MIN = _env("SPOKE_BG_ALPHA_MIN", 0.08)
_BG_ALPHA_MAX = _env("SPOKE_BG_ALPHA_MAX", 0.96)
_BG_AMP_SATURATION = _env("SPOKE_BG_AMP_SATURATION", 0.17)
_SMOOTH_RISE = _env("SPOKE_SMOOTH_RISE", 0.10)
_SMOOTH_DECAY = _env("SPOKE_SMOOTH_DECAY", 0.957)
_DARK_FILL_ADDITIVE_THRESHOLD = 0.15
_DARK_FILL_ADDITIVE_FILTER = "plusL"

# Adaptive compositing endpoints.
# On dark backgrounds: light/white fill, dark text — the overlay is a
# bright ghostly bubble that reads as additive glow.
# On light backgrounds: dark fill, text becomes transparent cutout —
# the overlay is a dark bubble with letter-shaped holes.
# Match the edge glow color on dark backgrounds — desaturated blue-white
_BG_COLOR_DARK = _scale_color_saturation((0.50, 0.59, 0.84), 0.40)
_TEXT_COLOR_DARK = (0.0, 0.0, 0.0)     # dark text on light fill
_BG_COLOR_LIGHT = (0.10, 0.10, 0.12)   # dark fill on light backgrounds
_TEXT_COLOR_LIGHT = (1.0, 1.0, 1.0)     # white text on dark fill (light backgrounds)

# Inner glow — matches screen border glow, scaled to overlay size
_GLOW_COLOR = _scale_color_saturation(
    (0.38, 0.52, 1.0), 0.13
)  # ~10% of original saturation — subtle tint, not a neon outline
_INNER_GLOW_WIDTH = 3.0  # proportional to overlay vs screen size
_INNER_GLOW_DEPTH = 30.0  # gradient extends inward — diffuse
_OUTER_FEATHER = 220.0  # glow bleed past overlay edge — wide for the stretched-exp tails
_INNER_GLOW_PEAK_TARGET = 0.50
_OUTER_GLOW_PEAK_TARGET = 0.35
_WIDE_OUTER_GLOW_SCALE = 0.56
_OVERLAY_INNER_SATURATION_SCALE = 0.70
_OVERLAY_OUTER_SATURATION_SCALE = 1.80
_POINTS_PER_CM = 72.0 / 2.54
_PREVIEW_OPTICAL_SHELL_BLEED_ZONE_FRAC = _env(
    "SPOKE_PREVIEW_OPTICAL_SHELL_BLEED_ZONE_FRAC", 0.8
)
_PREVIEW_OPTICAL_SHELL_EXTERIOR_MIX_WIDTH_POINTS = _env(
    "SPOKE_PREVIEW_OPTICAL_SHELL_EXTERIOR_MIX_WIDTH_POINTS", 20.0
)
_PREVIEW_OPTICAL_SHELL_INFLATION_X_RADII = _env(
    "SPOKE_PREVIEW_OPTICAL_SHELL_INFLATION_X_RADII", 2.0
)
_PREVIEW_OPTICAL_SHELL_INFLATION_Y_RADII = _env(
    "SPOKE_PREVIEW_OPTICAL_SHELL_INFLATION_Y_RADII", 2.0
)
_PREVIEW_OPTICAL_SHELL_CORE_MAGNIFICATION = _env(
    "SPOKE_PREVIEW_OPTICAL_SHELL_CORE_MAGNIFICATION", 1.55
)
_PREVIEW_OPTICAL_SHELL_BAND_MM = _env(
    "SPOKE_PREVIEW_OPTICAL_SHELL_BAND_MM", 4.0
)
_PREVIEW_OPTICAL_SHELL_TAIL_MM = _env(
    "SPOKE_PREVIEW_OPTICAL_SHELL_TAIL_MM", 3.0
)
_PREVIEW_OPTICAL_SHELL_RING_AMPLITUDE_POINTS = _env(
    "SPOKE_PREVIEW_OPTICAL_SHELL_RING_AMPLITUDE_POINTS",
    (_PREVIEW_OPTICAL_SHELL_BAND_MM / 10.0) * _POINTS_PER_CM,
)
_PREVIEW_OPTICAL_SHELL_TAIL_AMPLITUDE_POINTS = _env(
    "SPOKE_PREVIEW_OPTICAL_SHELL_TAIL_AMPLITUDE_POINTS",
    (_PREVIEW_OPTICAL_SHELL_TAIL_MM / 10.0) * _POINTS_PER_CM * 0.75,
)
_PREVIEW_OPTICAL_SHELL_X_SQUEEZE = _env(
    "SPOKE_PREVIEW_OPTICAL_SHELL_X_SQUEEZE", 2.5
)
_PREVIEW_OPTICAL_SHELL_Y_SQUEEZE = _env(
    "SPOKE_PREVIEW_OPTICAL_SHELL_Y_SQUEEZE", 1.5
)
_PREVIEW_OPTICAL_SHELL_CLEANUP_BLUR_RADIUS = _env(
    "SPOKE_PREVIEW_OPTICAL_SHELL_CLEANUP_BLUR_RADIUS", 0.75
)


# Recovery mode constants
_RECOVERY_BG_ALPHA = 0.35  # more opaque than recording, but still ghostly
_RECOVERY_TEXT_ALPHA = 0.80  # 80% — doesn't fully resolve
_RECOVERY_HINT_ALPHA = 0.40  # small hint text below overlay
_RECOVERY_HINT_FONT_SIZE = 11.0
_RECOVERY_DIVIDER_ALPHA = 0.18  # subtle column dividers
_RECOVERY_REJECT_DURATION = 0.3  # brief flash on Insert rejection
_RECOVERY_LABEL_FONT_SIZE = 14.0
_RECOVERY_PREVIEW_MAX_CHARS = 45  # truncate clipboard preview
_RECOVERY_HINT_MARGIN = 8.0  # gap between overlay and hint
_RECOVERY_HINT_HEIGHT = 20.0
_TRAY_CAPTURE_FLASH_ONSET_S = 0.10
_TRAY_CAPTURE_FLASH_FADE_OUT_S = 0.30


def _truncate_preview(text: str | None) -> str:
    """Truncate text for clipboard preview display."""
    if not text:
        return "(empty)"
    # Replace newlines with spaces for single-line display
    text = text.replace("\n", " ").replace("\r", " ")
    if len(text) > _RECOVERY_PREVIEW_MAX_CHARS:
        return text[:_RECOVERY_PREVIEW_MAX_CHARS] + "…"
    return text


def _lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t


def _lerp_color(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    return tuple(_lerp(s, e, t) for s, e in zip(start, end))


def _fill_compositing_filter_for_brightness(brightness: float) -> str | None:
    clamped = min(max(brightness, 0.0), 1.0)
    if clamped < _DARK_FILL_ADDITIVE_THRESHOLD:
        return _DARK_FILL_ADDITIVE_FILTER
    return None


def _compress_outer_glow_peak(opacity: float) -> float:
    """Keep low-level glow response intact while capping the outer bloom."""
    return min(opacity, _OUTER_GLOW_PEAK_TARGET)


def _overlay_layer_colors(
    base_color: tuple[float, float, float]
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Derive tighter and wider overlay glow layers from the base overlay color."""
    inner = _scale_color_saturation(base_color, _OVERLAY_INNER_SATURATION_SCALE)
    middle = base_color
    outer = _scale_color_saturation(base_color, _OVERLAY_OUTER_SATURATION_SCALE)
    return inner, middle, outer


def _max_overlay_height(screen_height: float) -> float:
    assistant_gap_cap = _COMMAND_OVERLAY_BOTTOM_MARGIN - _OVERLAY_BOTTOM_MARGIN
    capped_height = min(_OVERLAY_MAX_HEIGHT, assistant_gap_cap)
    return max(_OVERLAY_HEIGHT, capped_height)


# ── distance-field ridge ────────────────────────────────────

# Ridge shape: exponential peak at the bubble boundary that reads as a
# glowing edge rather than a stroked outline.  Interior fill preserved,
# intensity rising toward the boundary, peaking there, then falling off
# outside.

_RIDGE_FALLOFF = 1.2      # px — near-hairline peak at backing scale
_RIDGE_POWER = 16.0       # exponent — extremely sharp falloff
_RIDGE_BLOOM_FALLOFF = 8.0   # px — gentle ambient halo
_RIDGE_BLOOM_POWER = 3.0

# Crossover opacity bump — Gaussian centered at brightness 0.5
_CROSSOVER_CENTER = 0.35
_CROSSOVER_WIDTH = 0.12
_CROSSOVER_AMPLITUDE = 0.25

# Light-mode fill
_BG_ALPHA_LIGHT_BASE = 0.85
_BG_ALPHA_LIGHT_AMP = 0.10


def _overlay_rounded_rect_sdf(
    field_width: float, field_height: float,
    rect_width: float, rect_height: float,
    corner_radius: float, scale: float,
):
    """Signed distance field for a rounded rectangle centered in a larger field.

    The field covers field_width x field_height pixels; the rounded rect is
    rect_width x rect_height, centered within it.  This lets the SDF extend
    smoothly past the rect boundary into the surrounding margin.
    """
    import numpy as np

    pw, ph = int(field_width * scale), int(field_height * scale)
    rw, rh = rect_width * scale, rect_height * scale
    x = np.arange(pw, dtype=np.float32)[None, :] + 0.5
    y = np.arange(ph, dtype=np.float32)[:, None] + 0.5
    cx = x - pw * 0.5
    cy = y - ph * 0.5
    r = corner_radius * scale
    qx = np.abs(cx) - (rw * 0.5 - r)
    qy = np.abs(cy) - (rh * 0.5 - r)
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return (outside + inside - r).astype(np.float32)


def _ridge_alpha(signed_distance, falloff: float, power: float):
    """Exponential ridge: peaks at boundary (d=0), falls off symmetrically."""
    import numpy as np

    d = np.abs(signed_distance)
    return np.exp(-np.power(d / max(falloff, 1e-6), power, dtype=np.float32))


def _interior_fill_alpha(signed_distance, edge_softness: float):
    """Interior fill that fades smoothly to zero at the boundary.

    Fully opaque deep inside (negative distance), smoothly transitions
    to zero at and beyond the boundary.  edge_softness controls how many
    pixels before the boundary the fade begins.
    """
    import numpy as np

    # sigmoid-like ramp: 1 deep inside, 0 at boundary, smooth transition
    t = np.clip(-signed_distance / max(edge_softness, 1e-6), 0.0, 1.0)
    # smoothstep for a soft edge
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _glow_fill_alpha(signed_distance, width: float, interior_floor: float = 0.775):
    """Asymmetric stretched-exponential fill profile.

    Inside (negative distance): sharp cusp at boundary, drops rapidly
    then floors at interior_floor — the interior never goes below this.
    Outside (positive distance): same sharp cusp, drops all the way to
    zero with a long gradual tail.

    Both sides use exp(-sqrt(|d|/width)) for the pointed cusp and
    gradually decelerating falloff.
    """
    import numpy as np

    d = np.abs(signed_distance)
    raw = np.exp(-np.sqrt(d / max(width, 1e-6)))

    # Inside: remap so it goes from 1.0 at boundary toward interior_floor
    inside = interior_floor + (1.0 - interior_floor) * raw

    # Outside: raw curve goes all the way to zero
    return np.where(signed_distance <= 0.0, inside, raw).astype(np.float32)


def _fill_field_to_image(alpha, r: int, g: int, b: int):
    """Convert a float alpha field into a colored CGImage with per-pixel alpha.

    Unlike _alpha_field_to_image (which produces a white+alpha mask),
    this produces a colored image with the fill color baked in and
    premultiplied alpha.  Suitable for setting directly as a CALayer's
    contents without needing a separate mask layer.
    """
    import numpy as np
    from Quartz import (
        CGColorSpaceCreateDeviceRGB,
        CGDataProviderCreateWithCFData,
        CGImageCreate,
        kCGImageAlphaPremultipliedLast,
        kCGRenderingIntentDefault,
    )
    from Foundation import NSData

    mask_alpha = np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8)
    rgba = np.empty(mask_alpha.shape + (4,), dtype=np.uint8)
    # Premultiply: channel = channel_value * alpha / 255
    rgba[..., 0] = np.clip(r * alpha, 0.0, 255.0).astype(np.uint8)
    rgba[..., 1] = np.clip(g * alpha, 0.0, 255.0).astype(np.uint8)
    rgba[..., 2] = np.clip(b * alpha, 0.0, 255.0).astype(np.uint8)
    rgba[..., 3] = mask_alpha
    payload = NSData.dataWithBytes_length_(rgba.tobytes(), int(rgba.nbytes))
    provider = CGDataProviderCreateWithCFData(payload)
    image = CGImageCreate(
        alpha.shape[1],
        alpha.shape[0],
        8,
        32,
        alpha.shape[1] * 4,
        CGColorSpaceCreateDeviceRGB(),
        kCGImageAlphaPremultipliedLast,
        provider,
        None,
        False,
        kCGRenderingIntentDefault,
    )
    return image, payload


def _build_ridge_image(field_width: float, field_height: float,
                       rect_width: float, rect_height: float,
                       corner_radius: float, scale: float,
                       falloff: float, power: float):
    """Build a CGImage mask for the ridge effect at the given overlay size."""
    from .glow import _alpha_field_to_image

    sdf = _overlay_rounded_rect_sdf(field_width, field_height,
                                     rect_width, rect_height,
                                     corner_radius, scale)
    alpha = _ridge_alpha(sdf, falloff * scale, power)
    return _alpha_field_to_image(alpha)


def _window_origin_y(visible_height: float) -> float:
    base_y = _OVERLAY_BOTTOM_MARGIN - _OUTER_FEATHER
    if _EXPAND_UPWARD:
        return base_y
    return base_y - (visible_height - _OVERLAY_HEIGHT)


def _ontology_text_rgb(text_lum: float) -> tuple[float, float, float]:
    """Return a glow-blue text tint that stays legible against both fill modes."""
    if text_lum >= 0.5:
        return _scale_color_saturation((0.91, 0.95, 1.0), 1.4)
    return (0.07, 0.10, 0.19)


def _preview_warp_tuning_defaults() -> dict[str, float]:
    band_width_points = (_PREVIEW_OPTICAL_SHELL_BAND_MM / 10.0) * _POINTS_PER_CM
    tail_width_points = (_PREVIEW_OPTICAL_SHELL_TAIL_MM / 10.0) * _POINTS_PER_CM
    return {
        "inflation_x_radii": _PREVIEW_OPTICAL_SHELL_INFLATION_X_RADII,
        "inflation_y_radii": _PREVIEW_OPTICAL_SHELL_INFLATION_Y_RADII,
        "core_magnification": _PREVIEW_OPTICAL_SHELL_CORE_MAGNIFICATION,
        "band_width_points": band_width_points,
        "tail_width_points": tail_width_points,
        "ring_amplitude_points": _PREVIEW_OPTICAL_SHELL_RING_AMPLITUDE_POINTS,
        "tail_amplitude_points": _PREVIEW_OPTICAL_SHELL_TAIL_AMPLITUDE_POINTS,
        "bleed_zone_frac": _PREVIEW_OPTICAL_SHELL_BLEED_ZONE_FRAC,
        "exterior_mix_width_points": _PREVIEW_OPTICAL_SHELL_EXTERIOR_MIX_WIDTH_POINTS,
        "x_squeeze": _PREVIEW_OPTICAL_SHELL_X_SQUEEZE,
        "y_squeeze": _PREVIEW_OPTICAL_SHELL_Y_SQUEEZE,
        "cleanup_blur_radius_points": _PREVIEW_OPTICAL_SHELL_CLEANUP_BLUR_RADIUS,
    }


class TranscriptionOverlay(NSObject):
    """Manages a frosted overlay window for live transcription preview."""

    def initWithScreen_(self, screen: NSScreen | None = None):
        self = objc.super(TranscriptionOverlay, self).init()
        if self is None:
            return None

        self._screen = screen or NSScreen.mainScreen()
        self._window: NSWindow | None = None
        self._text_view: NSTextView | None = None
        self._scroll_view: NSScrollView | None = None
        self._visible = False
        self._fade_timer: NSTimer | None = None
        self._fade_step = 0
        self._fade_from = 0.0
        self._fade_direction = 0
        self._tray_mode = False

        # Typewriter state
        self._typewriter_timer: NSTimer | None = None
        self._typewriter_target = ""  # full text we're typing toward
        self._typewriter_displayed = ""  # what's currently shown

        # Text breathing — separate heavy smoothing so text doesn't flicker
        self._text_amplitude = 0.0

        # Adaptive compositing defaults dark until we get a brightness sample.
        self._brightness = 0.0
        self._brightness_target = 0.0
        self._fill_override_rgb: tuple[float, float, float] | None = None
        self._fill_override_opacity: float | None = None
        self._compositor_registry = None
        self._preview_compositor_client = None
        self._preview_compositor_identity = None
        self._preview_compositor_generation = 0
        self._preview_warp_tuning_overrides: dict[str, float] = {}

        # Recovery mode state
        self._recovery_mode = False
        self._recovery_buttons: list[NSView] = []
        self._recovery_labels: list[NSTextView] = []
        self._recovery_dividers: list[NSView] = []
        self._recovery_hint_window: NSWindow | None = None
        self._recovery_reject_timer: NSTimer | None = None
        self._tray_capture_flash_timer: NSTimer | None = None
        self._on_dismiss_callback = None
        self._on_insert_callback = None
        self._on_clipboard_toggle_callback = None

        # Performance caches — populated in setup() once we have screen state
        self._body_font = None           # NSFont — allocated once, reused forever
        self._max_overlay_height_cached = None  # screen height doesn't change
        self._ontology_spans_cache = ("", [])  # (text, spans)
        # Quantized color cache for amplitude ticks
        self._color_cache = {}
        self._last_color_key = None  # last key applied to text storage
        self._typewriter_layout_step = 0  # coalescing counter for _update_layout
        return self

    def preview_warp_tuning_snapshot(self) -> dict[str, float]:
        tuning = _preview_warp_tuning_defaults()
        tuning.update(getattr(self, "_preview_warp_tuning_overrides", {}))
        return tuning

    def set_preview_warp_tuning_value(self, key: str, value: float) -> None:
        self.update_preview_warp_tuning(**{key: value})

    def update_preview_warp_tuning(self, **updates: float) -> None:
        defaults = _preview_warp_tuning_defaults()
        overrides = dict(getattr(self, "_preview_warp_tuning_overrides", {}))
        for key, value in updates.items():
            if key not in defaults:
                continue
            numeric = float(value)
            if abs(numeric - defaults[key]) <= 1e-6:
                overrides.pop(key, None)
            else:
                overrides[key] = numeric
        self._preview_warp_tuning_overrides = overrides
        self._reapply_preview_warp_tuning()

    def reset_preview_warp_tuning(self) -> None:
        self._preview_warp_tuning_overrides = {}
        self._reapply_preview_warp_tuning()

    def _reapply_preview_warp_tuning(self) -> None:
        if (
            self._visible
            and not getattr(self, "_tray_mode", False)
            and not getattr(self, "_recovery_mode", False)
        ):
            self._publish_preview_compositor_snapshot(visible=True)

    def setup(self) -> None:
        """Create the overlay window."""
        screen_frame = self._screen.frame()
        sw = screen_frame.size.width

        # Window is oversized by _OUTER_FEATHER on each side for the feather bleed
        f = _OUTER_FEATHER
        x = (sw - _OVERLAY_WIDTH) / 2 - f
        y = _window_origin_y(_OVERLAY_HEIGHT)
        win_w = _OVERLAY_WIDTH + 2 * f
        win_h = _OVERLAY_HEIGHT + 2 * f
        frame = NSMakeRect(x, y, win_w, win_h)

        self._window = _ClickableWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, NSWindowStyleMaskNonactivatingPanel, NSBackingStoreBuffered, False
        )
        self._window.setLevel_(_OVERLAY_WINDOW_LEVEL)  # above other windows
        self._window.setOpaque_(False)
        self._window.setBackgroundColor_(NSColor.clearColor())
        self._window.setIgnoresMouseEvents_(True)
        self._window.setHasShadow_(False)
        self._window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        # Wrapper view — unclipped, holds both the dark box and the outer feather
        wrapper_frame = NSMakeRect(0, 0, win_w, win_h)
        wrapper = NSView.alloc().initWithFrame_(wrapper_frame)
        wrapper.setWantsLayer_(True)

        # Semi-transparent dark background with rounded corners (inset by feather)
        content_frame = NSMakeRect(f, f, _OVERLAY_WIDTH, _OVERLAY_HEIGHT)
        content = NSView.alloc().initWithFrame_(content_frame)
        content.setWantsLayer_(True)
        # No corner radius on the content view — the distance-field ridge
        # defines the visual boundary.  Text is inset enough not to bleed.
        content.layer().setMasksToBounds_(False)
        # Anchor point at center so scale animations expand symmetrically.
        # Setting anchorPoint shifts the layer visually, so also set position
        # to the center of the content frame to compensate.
        content.layer().setAnchorPoint_((0.5, 0.5))
        content.layer().setPosition_((f + _OVERLAY_WIDTH / 2, f + _OVERLAY_HEIGHT / 2))
        # Content view has NO background — the fill comes from a separate
        # SDF-masked layer.  Setting to None rather than clearColor ensures
        # no compositing boundary exists at the view's frame edge.
        content.layer().setBackgroundColor_(None)

        w, h = _OVERLAY_WIDTH, _OVERLAY_HEIGHT
        _, middle_rgb, _ = _overlay_layer_colors(_GLOW_COLOR)

        # Determine backing scale for SDF resolution
        self._ridge_scale = self._screen.backingScaleFactor() if hasattr(self._screen, 'backingScaleFactor') else 2.0

        # Fill layer — the interior fill is baked into a colored CGImage with
        # per-pixel alpha from the SDF smoothstep.  No mask layer needed —
        # the alpha falloff is in the image itself.  The fill color is updated
        # by rebuilding the image in update_text_amplitude.
        self._fill_layer = CALayer.alloc().init()
        self._fill_layer.setFrame_(((0, 0), (win_w, win_h)))
        self._fill_layer.setContentsGravity_("resize")

        # Build initial SDF fill image
        self._apply_ridge_masks(w, h)

        wrapper.layer().insertSublayer_below_(self._fill_layer, content.layer())

        wrapper.addSubview_(content)
        self._content_view = content

        # Scroll view with text view for scrollable transcription text
        scroll_frame = NSMakeRect(12, 8, _OVERLAY_WIDTH - 24, _OVERLAY_HEIGHT - 16)
        self._scroll_view = NSScrollView.alloc().initWithFrame_(scroll_frame)
        self._scroll_view.setHasVerticalScroller_(False)
        self._scroll_view.setHasHorizontalScroller_(False)
        self._scroll_view.setDrawsBackground_(False)
        self._scroll_view.setBorderType_(0)
        self._scroll_view.setAutoresizingMask_(18)
        # Explicitly clear the clip view's background — NSClipView can
        # draw its own background even when the scroll view doesn't.
        clip_view = self._scroll_view.contentView()
        if clip_view and hasattr(clip_view, 'setDrawsBackground_'):
            clip_view.setDrawsBackground_(False)
        if clip_view and hasattr(clip_view, 'setWantsLayer_'):
            clip_view.setWantsLayer_(True)
            clip_view.layer().setBackgroundColor_(None)

        text_frame = NSMakeRect(0, 0, _OVERLAY_WIDTH - 24, _OVERLAY_HEIGHT - 16)
        self._text_view = NSTextView.alloc().initWithFrame_(text_frame)
        self._text_view.setEditable_(False)
        self._text_view.setSelectable_(False)
        self._text_view.setDrawsBackground_(False)

        # Cache the body font once — no need to reallocate per tick
        self._body_font = NSFont.systemFontOfSize_weight_(_FONT_SIZE, 0.0)

        # Cache max overlay height — screen height doesn't change during a session
        self._max_overlay_height_cached = _max_overlay_height(
            self._screen.frame().size.height
        )

        self._current_text_color = NSColor.colorWithSRGBRed_green_blue_alpha_(
            1.0, 1.0, 1.0, _TEXT_ALPHA_MIN
        )
        ontology_r, ontology_g, ontology_b = _ontology_text_rgb(1.0)
        self._current_ontology_text_color = NSColor.colorWithSRGBRed_green_blue_alpha_(
            ontology_r, ontology_g, ontology_b, _TEXT_ALPHA_MIN
        )
        self._text_view.setTextColor_(self._current_text_color)
        self._text_view.setFont_(self._body_font)
        self._text_view.setString_("")
        self._text_view.textContainer().setWidthTracksTextView_(True)
        self._text_view.setHorizontallyResizable_(False)
        self._text_view.setVerticallyResizable_(True)

        self._scroll_view.setDocumentView_(self._text_view)
        content.addSubview_(self._scroll_view)
        self._window.setContentView_(wrapper)

        self._window.setAlphaValue_(0.0)

        logger.info("Transcription overlay created")

    def set_compositor_registry(self, registry) -> None:
        """Attach the shared optical compositor registry for preview snapshots."""
        if registry is not getattr(self, "_compositor_registry", None):
            self._release_preview_compositor_client()
        self._compositor_registry = registry

    def _preview_compositor_geometry_snapshot(self):
        from spoke.fullscreen_compositor import OpticalShellGeometrySnapshot

        scale = (
            self._screen.backingScaleFactor()
            if hasattr(self._screen, "backingScaleFactor")
            else 2.0
        )
        screen_frame = self._screen.frame()
        window_frame = self._window.frame()
        content_frame = self._content_view.frame()
        tuning = self.preview_warp_tuning_snapshot()
        visible_width = float(content_frame.size.width)
        visible_height = float(content_frame.size.height)
        shell_body_corner_r = min(_OVERLAY_CORNER_RADIUS, visible_height * 0.5)
        shell_width = visible_width + tuning["inflation_x_radii"] * shell_body_corner_r
        shell_height = visible_height + tuning["inflation_y_radii"] * shell_body_corner_r

        screen_origin_x = getattr(getattr(screen_frame, "origin", None), "x", 0.0)
        screen_origin_y = getattr(getattr(screen_frame, "origin", None), "y", 0.0)
        screen_height = getattr(getattr(screen_frame, "size", None), "height", 0.0)
        capsule_cx = (
            window_frame.origin.x
            + content_frame.origin.x
            + content_frame.size.width / 2.0
            - screen_origin_x
        )
        capsule_cy_cocoa = (
            window_frame.origin.y
            + content_frame.origin.y
            + content_frame.size.height / 2.0
        )
        capsule_cy_metal = screen_origin_y + screen_height - capsule_cy_cocoa
        return OpticalShellGeometrySnapshot(
            center_x=float(capsule_cx) * scale,
            center_y=float(capsule_cy_metal) * scale,
            content_width_points=shell_width * scale,
            content_height_points=shell_height * scale,
            corner_radius_points=shell_body_corner_r * scale,
            band_width_points=tuning["band_width_points"] * scale,
            tail_width_points=tuning["tail_width_points"] * scale,
        )

    def _preview_compositor_material_snapshot(self):
        from spoke.fullscreen_compositor import OpticalShellMaterialSnapshot

        brightness = min(max(float(getattr(self, "_brightness", 0.0)), 0.0), 1.0)
        tuning = self.preview_warp_tuning_snapshot()
        return OpticalShellMaterialSnapshot(
            initial_brightness=brightness,
            min_brightness=0.0,
            core_magnification=tuning["core_magnification"],
            ring_amplitude_points=tuning["ring_amplitude_points"],
            tail_amplitude_points=tuning["tail_amplitude_points"],
            bleed_zone_frac=tuning["bleed_zone_frac"],
            exterior_mix_width_points=tuning["exterior_mix_width_points"],
            x_squeeze=tuning["x_squeeze"],
            y_squeeze=tuning["y_squeeze"],
            cleanup_blur_radius_points=tuning["cleanup_blur_radius_points"],
            debug_visualize=False,
            debug_grid_spacing_points=18.0,
        )

    def _preview_compositor_excluded_window_ids(self) -> tuple[int, ...]:
        try:
            return (int(self._window.windowNumber()),)
        except Exception:
            return ()

    def _ensure_preview_compositor_client(self):
        registry = getattr(self, "_compositor_registry", None)
        if registry is None or self._screen is None or self._window is None:
            return None
        if getattr(self, "_content_view", None) is None:
            return None
        host = registry.host_for_screen(self._screen)
        from spoke.fullscreen_compositor import OverlayClientIdentity

        identity = OverlayClientIdentity(
            client_id="preview.transcription",
            display_id=host.display_id,
            role="preview",
        )
        client = getattr(self, "_preview_compositor_client", None)
        current_identity = getattr(self, "_preview_compositor_identity", None)
        if client is None or current_identity != identity:
            client = host.register_client(
                identity,
                window=self._window,
                content_view=self._content_view,
            )
            self._preview_compositor_client = client
        self._preview_compositor_identity = identity
        return client

    def _publish_preview_compositor_snapshot(self, *, visible: bool) -> bool:
        client = self._ensure_preview_compositor_client()
        identity = getattr(self, "_preview_compositor_identity", None)
        if client is None or identity is None:
            return False
        from spoke.fullscreen_compositor import OverlayRenderSnapshot

        self._preview_compositor_generation += 1
        snapshot = OverlayRenderSnapshot(
            identity=identity,
            generation=self._preview_compositor_generation,
            visible=bool(visible),
            geometry=self._preview_compositor_geometry_snapshot(),
            material=self._preview_compositor_material_snapshot(),
            excluded_window_ids=self._preview_compositor_excluded_window_ids(),
            z_index=0,
        )
        return bool(client.publish(snapshot))

    def _release_preview_compositor_client(self) -> None:
        client = getattr(self, "_preview_compositor_client", None)
        self._preview_compositor_client = None
        self._preview_compositor_identity = None
        if client is None:
            return
        release = getattr(client, "release", None)
        if callable(release):
            release()
            return
        stop = getattr(client, "stop", None)
        if callable(stop):
            stop()

    def _hide_and_release_preview_compositor_client(self) -> None:
        if getattr(self, "_preview_compositor_client", None) is not None:
            self._publish_preview_compositor_snapshot(visible=False)
        self._release_preview_compositor_client()

    def show(self) -> None:
        """Fade the overlay in."""
        if self._window is None:
            return
        self._cancel_tray_capture_flash()
        # If recovery mode is active, clean it up first
        if self._recovery_mode:
            self._recovery_mode = False
            self._teardown_recovery_views()
            if self._scroll_view is not None:
                self._scroll_view.setHidden_(False)
            self._window.setIgnoresMouseEvents_(True)
            # Content view stays transparent; reset the fill layer opacity
            self._content_view.layer().setBackgroundColor_(None)
            if hasattr(self, '_fill_layer') and self._fill_layer is not None:
                self._fill_layer.setOpacity_(_BG_ALPHA_MIN)
        self._cancel_fade()
        self._cancel_typewriter()
        self._visible = True
        self._tray_mode = False
        self._typewriter_target = ""
        self._typewriter_displayed = ""
        self._typewriter_hwm = 0  # furthest position typewriter has reached
        self._set_text_view_content("")
        self._content_view.layer().setBackgroundColor_(None)
        self._clear_fill_override(opacity=_BG_ALPHA_MIN)
        self._window.setAlphaValue_(0.0)

        # Reset to default size (window includes feather margin)
        screen_frame = self._screen.frame()
        sw = screen_frame.size.width
        f = _OUTER_FEATHER
        x = (sw - _OVERLAY_WIDTH) / 2 - f
        self._window.setFrame_display_animate_(
            NSMakeRect(x, _window_origin_y(_OVERLAY_HEIGHT),
                       _OVERLAY_WIDTH + 2 * f, _OVERLAY_HEIGHT + 2 * f),
            True, False
        )
        self._content_view.setFrame_(
            NSMakeRect(f, f, _OVERLAY_WIDTH, _OVERLAY_HEIGHT)
        )
        scroll_frame = NSMakeRect(12, 8, _OVERLAY_WIDTH - 24, _OVERLAY_HEIGHT - 16)
        self._scroll_view.setFrame_(scroll_frame)
        self._reset_overlay_chrome_geometry(_OVERLAY_HEIGHT)
        self._reset_text_geometry(_OVERLAY_HEIGHT - 16, scroll_to_top=True)

        self._publish_preview_compositor_snapshot(visible=True)
        self._window.orderFrontRegardless()

        # Fade in using stepped timer
        self._fade_step = 0
        self._fade_from = 0.0
        self._fade_target = 1.0
        self._fade_direction = 1  # fading in
        interval = _FADE_IN_S / _FADE_STEPS
        self._fade_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            interval, self, "fadeStep:", None, True
        )
        logger.info("Overlay show")

    def set_brightness(self, brightness: float, immediate: bool = False) -> None:
        """Set screen brightness (0.0 dark – 1.0 bright) for adaptive compositing."""
        self._brightness_target = min(max(brightness, 0.0), 1.0)
        if immediate:
            self._brightness = self._brightness_target

    def hide(self, *, fade_duration: float | None = None) -> None:
        """Fade the overlay out smoothly."""
        if self._window is None:
            return
        self._cancel_tray_capture_flash()
        self._visible = False
        self._tray_mode = False
        self._cancel_typewriter()
        self._hide_and_release_preview_compositor_client()
        self._start_fade_out(duration=fade_duration)
        logger.info("Overlay hide")

    def order_out(self) -> None:
        """Immediately remove the overlay from screen (no fade).

        Used before the AX focus check so the overlay window doesn't
        mask the underlying focused element.
        """
        if self._window is None:
            return
        self._visible = False
        self._tray_mode = False
        self._cancel_tray_capture_flash()
        self._cancel_fade()
        self._cancel_typewriter()
        self._hide_and_release_preview_compositor_client()
        self._window.setAlphaValue_(0.0)
        self._window.orderOut_(None)
        logger.info("Overlay ordered out")

    def _start_fade_out(self, *, duration: float | None = None) -> None:
        """Animate fade-out using a repeating timer for smooth steps."""
        self._cancel_fade()
        self._fade_step = 0
        self._fade_from = self._window.alphaValue()
        self._fade_target = 0.0
        self._fade_direction = -1  # fading out
        fade_duration = duration if duration is not None else _FADE_OUT_S
        interval = fade_duration / _FADE_STEPS
        self._fade_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            interval, self, "fadeStep:", None, True
        )

    def fadeStep_(self, timer) -> None:
        """One step of the fade animation."""
        self._fade_step += 1
        progress = self._fade_step / _FADE_STEPS

        if self._fade_direction == 1:
            # Fade in: ease-in (slow start, confident finish)
            eased = progress * progress
            alpha = eased
        else:
            # Fade out: ease-in (slow start, fast end)
            eased = progress * progress
            alpha = self._fade_from * (1.0 - eased)

        self._window.setAlphaValue_(alpha)

        if self._fade_step >= _FADE_STEPS:
            self._cancel_fade()
            if self._fade_direction == -1:
                self._window.setAlphaValue_(0.0)
                self._window.orderOut_(None)
            else:
                self._window.setAlphaValue_(1.0)

    def _cancel_fade(self) -> None:
        if self._fade_timer is not None:
            self._fade_timer.invalidate()
            self._fade_timer = None

    def _set_text_view_content(
        self,
        text: str,
        *,
        base_color: NSColor | None = None,
        ontology_color: NSColor | None = None,
    ) -> None:
        """Render text with ontology terms tinted in the glow-blue family.

        Full attributed string rebuild — only called when text actually changes
        (typewriter steps, snaps, show_tray).  Amplitude ticks use
        _update_text_color_inplace instead.
        """
        if self._text_view is None:
            return

        if base_color is not None:
            self._current_text_color = base_color
        elif not hasattr(self, "_current_text_color"):
            self._current_text_color = NSColor.colorWithSRGBRed_green_blue_alpha_(
                1.0, 1.0, 1.0, _TEXT_ALPHA_MIN
            )
        if ontology_color is not None:
            self._current_ontology_text_color = ontology_color
        elif not hasattr(self, "_current_ontology_text_color"):
            ontology_r, ontology_g, ontology_b = _ontology_text_rgb(1.0)
            self._current_ontology_text_color = NSColor.colorWithSRGBRed_green_blue_alpha_(
                ontology_r, ontology_g, ontology_b, _TEXT_ALPHA_MIN
            )

        if not text:
            self._text_view.setString_("")
            # Invalidate color cache so next amplitude tick re-applies colors
            self._last_color_key = None
            return

        text_storage = self._text_view.textStorage() if hasattr(self._text_view, "textStorage") else None
        if text_storage is None:
            self._text_view.setString_(text)
            if hasattr(self._text_view, "setTextColor_"):
                self._text_view.setTextColor_(self._current_text_color)
            return

        # Use cached ontology spans — recompute only when text changed
        cached_text, cached_spans = getattr(self, "_ontology_spans_cache", ("", []))
        if cached_text != text:
            cached_spans = list(ontology_term_spans(text))
            self._ontology_spans_cache = (text, cached_spans)

        # Use cached font
        font = self._body_font or NSFont.systemFontOfSize_weight_(_FONT_SIZE, 0.0)

        attr_str = NSMutableAttributedString.alloc().initWithString_(text)
        attr_str.addAttribute_value_range_(
            NSForegroundColorAttributeName,
            self._current_text_color,
            (0, len(text)),
        )
        attr_str.addAttribute_value_range_(
            NSFontAttributeName,
            font,
            (0, len(text)),
        )
        for start, end in cached_spans:
            attr_str.addAttribute_value_range_(
                NSForegroundColorAttributeName,
                self._current_ontology_text_color,
                (start, end - start),
            )
        text_storage.setAttributedString_(attr_str)
        # Invalidate color cache — text storage just replaced, key no longer valid
        self._last_color_key = None

    def _update_text_color_inplace(
        self,
        text: str,
        base_color,
        ontology_color,
        color_key: tuple,
    ) -> None:
        """Update text colors in-place on existing text storage.

        Called from update_text_amplitude when only the color changed and the
        text string is the same.  Does NOT rebuild the attributed string or
        re-run ontology_term_spans — just patches color attributes on the
        existing storage.
        """
        if self._text_view is None or not text:
            return
        if self._last_color_key == color_key:
            return  # quantized values unchanged — skip entirely

        text_storage = self._text_view.textStorage() if hasattr(self._text_view, "textStorage") else None
        if text_storage is None:
            return

        n = len(text)
        text_storage.addAttribute_value_range_(
            NSForegroundColorAttributeName, base_color, (0, n)
        )
        # Re-apply ontology span overrides
        _, cached_spans = getattr(self, "_ontology_spans_cache", ("", []))
        for start, end in cached_spans:
            text_storage.addAttribute_value_range_(
                NSForegroundColorAttributeName, ontology_color, (start, end - start)
            )

        self._current_text_color = base_color
        self._current_ontology_text_color = ontology_color
        self._last_color_key = color_key

    # ── typewriter effect ────────────────────────────────────

    def set_text(self, text: str) -> None:
        """Update the target text — typewriter effect types toward it."""
        if self._text_view is None or not self._visible:
            return

        self._typewriter_target = text

        # If the new text doesn't start with what we've displayed,
        # the transcription revised earlier words.
        if not text.startswith(self._typewriter_displayed):
            # Find divergence point
            common = 0
            for i, (a, b) in enumerate(zip(self._typewriter_displayed, text)):
                if a == b:
                    common = i + 1
                else:
                    break

            # Allow small jitter (punctuation, capitalization) near the
            # typing frontier without triggering a full snap.
            _FUZZ = 3  # chars of slack behind the high-water mark
            if common < self._typewriter_hwm - _FUZZ:
                # Divergence is well behind the high-water mark — the user
                # already saw those characters typewrite in.  Snap the full
                # text instantly so we never re-animate already-seen content.
                self._cancel_typewriter()
                self._typewriter_displayed = text
                self._typewriter_hwm = len(text)
                self._set_text_view_content(text)
                self._update_layout()
                return
            else:
                self._typewriter_displayed = text[:common]
                self._set_text_view_content(self._typewriter_displayed)

        # Start typing if not already
        if self._typewriter_timer is None and len(self._typewriter_displayed) < len(self._typewriter_target):
            self._typewriter_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                _TYPEWRITER_INTERVAL, self, "typewriterStep:", None, True
            )

    def typewriterStep_(self, timer) -> None:
        """Append one character toward the target text."""
        if len(self._typewriter_displayed) < len(self._typewriter_target):
            self._typewriter_displayed = self._typewriter_target[:len(self._typewriter_displayed) + 1]
            self._typewriter_hwm = max(self._typewriter_hwm, len(self._typewriter_displayed))
            self._set_text_view_content(self._typewriter_displayed)
            # Coalesce layout passes — at most once every 5 typewriter steps
            # (~100ms at 50 Hz).  scrollRangeToVisible_ happens every step for
            # smooth caret tracking without the full typesetter query.
            self._typewriter_layout_step = getattr(self, "_typewriter_layout_step", 0) + 1
            if self._typewriter_layout_step >= 5:
                self._typewriter_layout_step = 0
                self._update_layout()
            else:
                # Lightweight scroll-only update — no layout query
                try:
                    if self._text_view is not None:
                        end = (
                            self._text_view.string().length()
                            if hasattr(self._text_view.string(), "length")
                            else len(self._typewriter_displayed)
                        )
                        self._text_view.scrollRangeToVisible_((end, 0))
                except Exception:
                    pass
        else:
            self._cancel_typewriter()

    def _cancel_typewriter(self) -> None:
        if self._typewriter_timer is not None:
            self._typewriter_timer.invalidate()
            self._typewriter_timer = None

    # ── amplitude-reactive text ──────────────────────────────

    def update_text_amplitude(self, amplitude: float) -> None:
        """Update text opacity based on voice amplitude (0.0–1.0).

        Must be called on the main thread. Uses heavy smoothing so the
        text breathes slowly rather than flickering at 62Hz.
        """
        if self._text_view is None or not self._visible:
            return

        # Smoothing — rise same as before, decay holds longer then accelerates.
        # The fill lingers after you stop speaking, then drops away cleanly.
        if amplitude > self._text_amplitude:
            self._text_amplitude += (amplitude - self._text_amplitude) * _SMOOTH_RISE
        else:
            # Slow decay — the fill lingers after speaking but still visibly
            # moves.  Rate 0.99 when high (~70 frames to halve = ~1.2 seconds),
            # accelerates to 0.95 near zero for clean finish.
            gap = self._text_amplitude
            ease = 0.95 + 0.04 * min(gap / 0.3, 1.0)  # 0.99 when high, 0.95 near zero
            self._text_amplitude *= ease

        # Chase brightness target over roughly half a second at the live update cadence.
        _BRIGHTNESS_CHASE = 0.08
        target = getattr(self, "_brightness_target", 0.0)
        current = getattr(self, "_brightness", 0.0)
        if abs(target - current) > 0.001:
            self._brightness = current + (target - current) * _BRIGHTNESS_CHASE

        scaled = min(self._text_amplitude / _TEXT_AMP_SATURATION, 1.0)

        t = getattr(self, "_brightness", 0.0)

        # Text: anchored near-opaque.  Dark on white backgrounds, white on
        # dark backgrounds.  Text does NOT breathe with amplitude — it stays
        # legible and stable.  The SDF fill breathes instead.
        # Text alpha: on dark backgrounds, anchored at 0.88 (no RMS link).
        # On light backgrounds, slight RMS waiver: floor 0.80, ceiling 1.0.
        if t > 0.15:
            _TEXT_ANCHOR_ALPHA = _lerp(0.80, 1.0, scaled)
        else:
            _TEXT_ANCHOR_ALPHA = 0.88
        # Text contrasts against the fill: light fill (dark bg) → dark text,
        # dark fill (light bg) → white text.
        bg_r, bg_g, bg_b = _lerp_color(_BG_COLOR_DARK, _BG_COLOR_LIGHT, t)
        bg_lum = 0.299 * bg_r + 0.587 * bg_g + 0.114 * bg_b
        target_text_lum = 0.0 if bg_lum > 0.5 else 1.0

        # Ease-out snap: chase the target with a fast-start, slow-finish curve.
        # ~200ms at 60Hz = ~12 frames.  The ease-out makes the snap feel
        # satisfying — it commits immediately then settles.
        current_text_lum = getattr(self, '_text_lum', target_text_lum)
        _TEXT_SNAP_SPEED = 0.35  # ease-out: big initial jump, then settles
        current_text_lum += (target_text_lum - current_text_lum) * _TEXT_SNAP_SPEED
        self._text_lum = current_text_lum
        tr = tg = tb = current_text_lum
        ontology_r, ontology_g, ontology_b = _ontology_text_rgb(current_text_lum)

        # Quantize color components to ~2 decimal places so minor floating-point
        # drift from smoothing does not trigger unnecessary NSColor allocations.
        _Q = 100.0
        color_key = (
            round(tr * _Q),
            round(tg * _Q),
            round(tb * _Q),
            round(_TEXT_ANCHOR_ALPHA * _Q),
            round(ontology_r * _Q),
            round(ontology_g * _Q),
            round(ontology_b * _Q),
            round(_TEXT_ANCHOR_ALPHA * _Q),
        )

        color_cache = getattr(self, "_color_cache", {})
        if color_key not in color_cache:
            base_color = NSColor.colorWithSRGBRed_green_blue_alpha_(
                tr, tg, tb, _TEXT_ANCHOR_ALPHA
            )
            ontology_color = NSColor.colorWithSRGBRed_green_blue_alpha_(
                ontology_r, ontology_g, ontology_b, _TEXT_ANCHOR_ALPHA
            )
            color_cache[color_key] = (base_color, ontology_color)
            # Bound cache size — very few distinct quantized values in practice
            if len(color_cache) > 32:
                del color_cache[next(iter(color_cache))]
            self._color_cache = color_cache
        else:
            base_color, ontology_color = color_cache[color_key]

        displayed = getattr(self, "_typewriter_displayed", "")
        # Update color in-place — do NOT rebuild the full attributed string on
        # amplitude ticks where only the color changes and the text is stable.
        self._update_text_color_inplace(displayed, base_color, ontology_color, color_key)

        # SDF fill breathes with amplitude.  On light backgrounds the fill
        # is relatively MORE assertive (because the glow is dimming the same
        # area).  On dark backgrounds the fill is subtle (the glow carries
        # the visual).  Phase shift: fill uses squared response so it leads
        # the glow — visible before the glow builds up, and at low RMS the
        # fill is already present against the undimmed background.
        # On dark backgrounds use linear response so soft sounds register.
        # On light backgrounds use squared so the fill leads the glow.
        fill_drive = _lerp(scaled, scaled * scaled, t)
        fill_min = _lerp(0.06, 0.84, t)   # light: 2x rest presence — much more material
        fill_max = _lerp(0.92, 0.99, t)   # saturates near-full on both backgrounds
        fill_opacity = _lerp(fill_min, fill_max, fill_drive)
        if hasattr(self, '_fill_layer') and self._fill_layer is not None:
            self._fill_layer.setOpacity_(min(fill_opacity, 0.96))
            # Rebuild the fill image when brightness changes enough to
            # affect the baked color.
            last_t = getattr(self, '_fill_image_brightness', -1.0)
            if abs(t - last_t) > 0.03:
                self._fill_image_brightness = t
                # Recompute the full SDF + fill image at the current overlay
                # size.  Using _update_fill_image with a stale SDF caused
                # corner distortion when the overlay had resized since the
                # SDF was last computed.
                content = getattr(self, '_content_view', None)
                if content:
                    try:
                        cf = content.frame()
                        self._apply_ridge_masks(cf.size.width, cf.size.height)
                    except Exception:
                        pass

    def update_glow_amplitude(self, opacity: float, cap_factor: float = 1.0) -> None:
        """Update inner and outer glow opacity to match the screen glow.

        opacity should be the screen glow's current opacity (0.0–1.0).
        cap_factor scales the glow down during the recording cap countdown
        (1.0 = full, ramps toward 0.25 near the cap).

        Has its own smoothing (60% of screen glow's attack speed) so the
        overlay glow responds more gently than the screen edge.
        """
        if not self._visible:
            return
        # Independent smoothing — 60% of the screen glow's attack
        _OVERLAY_GLOW_RISE = 0.54   # 60% of screen glow's 0.90
        _OVERLAY_GLOW_DECAY = 0.70  # 60% blend toward screen glow's 0.50
        if not hasattr(self, '_smoothed_glow_opacity'):
            self._smoothed_glow_opacity = 0.0
        if opacity > self._smoothed_glow_opacity:
            self._smoothed_glow_opacity += (opacity - self._smoothed_glow_opacity) * _OVERLAY_GLOW_RISE
        else:
            self._smoothed_glow_opacity += (opacity - self._smoothed_glow_opacity) * (1.0 - _OVERLAY_GLOW_DECAY)
        opacity = self._smoothed_glow_opacity

        # Apply recording-cap countdown scaling
        if cap_factor < 1.0:
            cap_floor = 0.25
            scale = cap_floor + (1.0 - cap_floor) * cap_factor
            opacity *= scale
        # Ridge layer removed — the SDF fill edge carries the boundary.

    # ── layout helpers ───────────────────────────────────────

    def _reset_text_geometry(self, visible_height: float, scroll_to_top: bool = False) -> None:
        """Keep the document view and clip view in sync with the current overlay size."""
        if self._text_view is None or self._scroll_view is None:
            return

        doc_frame = NSMakeRect(0, 0, _OVERLAY_WIDTH - 24, visible_height)
        self._text_view.setFrame_(doc_frame)

        container = self._text_view.textContainer()
        if container is not None and hasattr(container, "setContainerSize_"):
            container.setContainerSize_((_OVERLAY_WIDTH - 24, 1.0e7))

        clip_view = self._scroll_view.contentView() if hasattr(self._scroll_view, "contentView") else None
        if clip_view is not None and scroll_to_top:
            if hasattr(clip_view, "scrollToPoint_"):
                clip_view.scrollToPoint_((0, 0))
            elif hasattr(clip_view, "setBoundsOrigin_"):
                clip_view.setBoundsOrigin_((0, 0))
            if hasattr(self._scroll_view, "reflectScrolledClipView_"):
                self._scroll_view.reflectScrolledClipView_(clip_view)

    def _apply_ridge_masks(self, width: float, height: float) -> None:
        """Compute SDF and apply ridge + bloom masks for the given overlay size.

        The SDF is computed for the content rect (width x height) and embedded
        in a larger field covering the full window (content + feather margin)
        so the outer falloff bleeds into the feather zone.

        The SDF is cached by geometry (width, height, scale).  When only
        brightness changes the cache is hit and the expensive numpy
        computation is skipped — only the colored fill image is rebuilt.
        """
        f = _OUTER_FEATHER
        scale = getattr(self, '_ridge_scale', 2.0)
        total_w = width + 2 * f
        total_h = height + 2 * f

        geom_key = (width, height, scale)
        brightness = getattr(self, "_brightness", 0.0)
        fill_override_rgb = getattr(self, "_fill_override_rgb", None)
        appearance_key = (
            round(float(total_w), 3),
            round(float(total_h), 3),
            round(float(scale), 3),
            round(float(brightness) * 50.0) / 50.0,
            fill_override_rgb,
        )
        self._desired_fill_image_signature = appearance_key
        if (
            getattr(self, "_fill_image_signature", None) == appearance_key
            and getattr(self, "_fill_payload", None) is not None
        ):
            return
        pending = getattr(self, "_pending_fill_image_signature", None)
        if pending is not None:
            if pending != appearance_key:
                self._queued_fill_request = (width, height)
            return
        if hasattr(self, "_fill_layer") and self._fill_layer is not None:
            self._fill_layer.setFrame_(((0, 0), (total_w, total_h)))

        self._pending_fill_image_signature = appearance_key
        cached_sdf = (
            getattr(self, "_fill_sdf", None)
            if getattr(self, "_sdf_cache_key", None) == geom_key
            else None
        )

        def build() -> None:
            try:
                sdf = cached_sdf
                if sdf is None:
                    sdf = _overlay_rounded_rect_sdf(
                        total_w, total_h, width, height,
                        _OVERLAY_CORNER_RADIUS, scale,
                    )
                floor = _lerp(0.55, 0.775, brightness)
                fill_alpha = _glow_fill_alpha(
                    sdf, width=2.5 * scale, interior_floor=floor
                )
                if fill_override_rgb is None:
                    bg_r, bg_g, bg_b = _lerp_color(
                        _BG_COLOR_DARK, _BG_COLOR_LIGHT, brightness
                    )
                else:
                    bg_r, bg_g, bg_b = fill_override_rgb
                result = {
                    "signature": appearance_key,
                    "sdf": sdf,
                    "geom_key": geom_key,
                    "scale": scale,
                    "total_w": total_w,
                    "total_h": total_h,
                    "fill_override_rgb": fill_override_rgb,
                }
                try:
                    fill_image, payload = _fill_field_to_image(
                        fill_alpha,
                        int(bg_r * 255), int(bg_g * 255), int(bg_b * 255),
                    )
                    result["image"] = fill_image
                    result["payload"] = payload
                except Exception as exc:
                    result["error"] = repr(exc)
            except Exception as exc:
                result = {"signature": appearance_key, "error": repr(exc)}
            _post_overlay_result_to_main(self, "fillImageReady:", result)

        _start_overlay_fill_worker(build)

    def _update_fill_image(self, total_w: float, total_h: float) -> None:
        """Rebuild the colored fill image from the stashed SDF and current fill color."""
        content_width = max(total_w - 2 * _OUTER_FEATHER, 1.0)
        content_height = max(total_h - 2 * _OUTER_FEATHER, 1.0)
        self._apply_ridge_masks(content_width, content_height)

    def fillImageReady_(self, payload: dict) -> None:
        signature = payload.get("signature")
        if getattr(self, "_pending_fill_image_signature", None) == signature:
            self._pending_fill_image_signature = None
        if signature != getattr(self, "_desired_fill_image_signature", None):
            queued = getattr(self, "_queued_fill_request", None)
            if queued is not None:
                self._queued_fill_request = None
                self._apply_ridge_masks(*queued)
            return
        if "geom_key" in payload:
            self._fill_sdf = payload.get("sdf")
            self._fill_scale = payload.get("scale", getattr(self, "_fill_scale", 2.0))
            self._sdf_cache_key = payload.get("geom_key")
        error = payload.get("error")
        if error:
            logger.debug("Overlay fill image generation failed: %s", error)
            return
        if not hasattr(self, "_fill_layer") or self._fill_layer is None:
            return
        self._fill_payload = payload.get("payload")
        self._fill_layer.setContents_(payload.get("image"))
        self._fill_layer.setFrame_(((0, 0), (payload["total_w"], payload["total_h"])))
        self._fill_image_signature = signature
        if hasattr(self._fill_layer, "setCompositingFilter_"):
            fill_override_rgb = payload.get("fill_override_rgb")
            filter_name = (
                None
                if fill_override_rgb is not None
                else _fill_compositing_filter_for_brightness(getattr(self, "_brightness", 0.0))
            )
            self._fill_layer.setCompositingFilter_(filter_name)
        queued = getattr(self, "_queued_fill_request", None)
        if queued is not None:
            self._queued_fill_request = None
            self._apply_ridge_masks(*queued)

    def _reset_overlay_chrome_geometry(self, visible_height: float) -> None:
        """Keep height-dependent overlay layers in sync with the current overlay size."""
        w = _OVERLAY_WIDTH
        self._apply_ridge_masks(w, visible_height)

    def _set_fill_override(
        self,
        rgb: tuple[float, float, float],
        opacity: float,
    ) -> None:
        self._fill_override_rgb = rgb
        self._fill_override_opacity = opacity
        if hasattr(self, "_fill_layer") and self._fill_layer is not None:
            self._fill_layer.setOpacity_(opacity)
        if getattr(self, "_content_view", None) is not None:
            content_frame = self._content_view.frame()
            self._apply_ridge_masks(content_frame.size.width, content_frame.size.height)

    def _clear_fill_override(self, *, opacity: float | None = None) -> None:
        self._fill_override_rgb = None
        self._fill_override_opacity = None
        if hasattr(self, "_fill_layer") and self._fill_layer is not None and opacity is not None:
            self._fill_layer.setOpacity_(opacity)
        if getattr(self, "_content_view", None) is not None:
            content_frame = self._content_view.frame()
            self._apply_ridge_masks(content_frame.size.width, content_frame.size.height)

    def _update_layout(self) -> None:
        """Resize window and scroll to bottom after text change."""
        try:
            layout = self._text_view.layoutManager()
            container = self._text_view.textContainer()
            if layout and container:
                layout.ensureLayoutForTextContainer_(container)
                text_rect = layout.usedRectForTextContainer_(container)
                text_height = text_rect.size.height
            else:
                text_height = _OVERLAY_HEIGHT - 16

            # Use cached max height — screen height does not change mid-session.
            # Fall back to a live query if setup() was not called (e.g. tests that
            # directly call _update_layout on a bare instance).
            max_height = (
                self._max_overlay_height_cached
                if getattr(self, "_max_overlay_height_cached", None) is not None
                else _max_overlay_height(self._screen.frame().size.height)
            )
            new_height = min(max(_OVERLAY_HEIGHT, text_height + 24), max_height)

            f = _OUTER_FEATHER
            new_win_h = new_height + 2 * f
            # Check the resize guard BEFORE querying the live window frame, so we
            # skip the window-frame round-trip on the common no-resize case.
            win_frame = self._window.frame()
            if abs(win_frame.size.height - new_win_h) > 4:
                win_frame.origin.y = _window_origin_y(new_height)
                win_frame.size.height = new_win_h
                self._window.setFrame_display_animate_(win_frame, True, False)
                self._content_view.setFrame_(
                    NSMakeRect(f, f, _OVERLAY_WIDTH, new_height)
                )
                self._scroll_view.setFrame_(NSMakeRect(12, 8, _OVERLAY_WIDTH - 24, new_height - 16))
                self._reset_overlay_chrome_geometry(new_height)

            self._reset_text_geometry(max(new_height - 16, text_height))
            end = self._text_view.string().length() if hasattr(self._text_view.string(), 'length') else len(self._typewriter_displayed)
            self._text_view.scrollRangeToVisible_((end, 0))
            if (
                self._visible
                and not getattr(self, "_tray_mode", False)
                and not getattr(self, "_recovery_mode", False)
            ):
                self._publish_preview_compositor_snapshot(visible=True)
        except Exception:
            pass

    # ── tray mode ──────────────────────────────────────────────

    def show_tray(self, text: str, *, owner: str = "user") -> None:
        """Show the tray overlay with the given text.

        Displays the text immediately (no typewriter effect) in the
        normal overlay style. No buttons, no interactive elements.
        The tray gesture vocabulary handles all interaction.
        """
        if self._window is None:
            return
        self._cancel_tray_capture_flash()
        self._hide_and_release_preview_compositor_client()
        self._tray_mode = True

        # Clean up any existing recovery state
        if self._recovery_mode:
            self._recovery_mode = False
            self._teardown_recovery_views()
            self._window.setIgnoresMouseEvents_(True)

        self._cancel_fade()
        self._cancel_typewriter()

        # Show normal scroll view
        if self._scroll_view is not None:
            self._scroll_view.setHidden_(False)

        # Reset background to the owner's color language.
        if owner == "assistant":
            fill_rgb = (0.10, 0.13, 0.19)
            text_color = NSColor.colorWithSRGBRed_green_blue_alpha_(
                0.88, 0.93, 1.0, _RECOVERY_TEXT_ALPHA
            )
        else:
            fill_rgb = (0.1, 0.1, 0.12)
            text_color = NSColor.colorWithSRGBRed_green_blue_alpha_(
                1.0, 1.0, 1.0, _RECOVERY_TEXT_ALPHA
            )
        self._content_view.layer().setBackgroundColor_(None)

        # Reset to default height
        screen_frame = self._screen.frame()
        sw = screen_frame.size.width
        f = _OUTER_FEATHER
        x = (sw - _OVERLAY_WIDTH) / 2 - f
        self._window.setFrame_display_animate_(
            NSMakeRect(x, _OVERLAY_BOTTOM_MARGIN - f,
                       _OVERLAY_WIDTH + 2 * f, _OVERLAY_HEIGHT + 2 * f),
            True, False
        )
        self._content_view.setFrame_(
            NSMakeRect(f, f, _OVERLAY_WIDTH, _OVERLAY_HEIGHT)
        )
        self._reset_overlay_chrome_geometry(_OVERLAY_HEIGHT)
        self._set_fill_override(fill_rgb, _RECOVERY_BG_ALPHA)

        # Set text immediately (no typewriter)
        self._typewriter_target = text
        self._typewriter_displayed = text
        self._typewriter_hwm = len(text)
        ontology_r, ontology_g, ontology_b = _ontology_text_rgb(1.0)
        if owner == "assistant":
            ontology_color = NSColor.colorWithSRGBRed_green_blue_alpha_(
                0.78, 0.84, 1.0, _RECOVERY_TEXT_ALPHA
            )
        else:
            ontology_color = NSColor.colorWithSRGBRed_green_blue_alpha_(
                ontology_r, ontology_g, ontology_b, _RECOVERY_TEXT_ALPHA
            )
        if self._text_view is not None:
            self._set_text_view_content(
                text,
                base_color=text_color,
                ontology_color=ontology_color,
            )
        self._update_layout()

        # Show overlay
        self._visible = True
        self._window.setAlphaValue_(1.0)
        self._window.orderFrontRegardless()

        # Entrance pop
        self._pop_entrance()

        logger.info(
            "Tray overlay shown (%s): %r",
            owner,
            text[:50] if text else "",
        )

    def flash_notice(self, text: str, hold: float = 3.0, fade: float = 1.5) -> None:
        """Show a transient notice on the overlay, then auto-fade.

        The overlay appears with the given text, holds for *hold* seconds,
        then fades out over *fade* seconds. No click required to dismiss.
        """
        if self._window is None:
            return
        self.show()
        self.set_text(text)
        self._cancel_notice_timer()
        self._notice_fade = fade

        self._notice_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            hold, self, "noticeTimerDone:", None, False,
        )

    def noticeTimerDone_(self, timer) -> None:
        self._notice_timer = None
        fade = getattr(self, "_notice_fade", 1.5)
        self.hide(fade_duration=fade)

    def _cancel_notice_timer(self) -> None:
        if getattr(self, "_notice_timer", None) is not None:
            self._notice_timer.invalidate()
            self._notice_timer = None

    def flash_tray_capture(self, text: str, *, owner: str = "user") -> None:
        """Briefly acknowledge a silent tray save, then vanish."""
        if self._window is None:
            return
        self.show_tray(text, owner=owner)
        self._pulse_tray_capture_ack()
        self._cancel_tray_capture_flash()
        self._tray_capture_flash_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            _TRAY_CAPTURE_FLASH_ONSET_S,
            self,
            "trayCaptureFlashDone:",
            None,
            False,
        )

    def _pulse_tray_capture_ack(self) -> None:
        """Quick ease-in/ease-out pulse for silent tray-save feedback."""
        if self._window is None:
            return
        from Quartz import CABasicAnimation, CAMediaTimingFunction

        content_layer = self._content_view.layer()
        pulse = CABasicAnimation.animationWithKeyPath_("transform.scale")
        pulse.setFromValue_(1.0)
        pulse.setToValue_(1.02)
        pulse.setDuration_(_TRAY_CAPTURE_FLASH_ONSET_S)
        pulse.setAutoreverses_(True)
        pulse.setTimingFunction_(
            CAMediaTimingFunction.functionWithName_("easeInEaseOut")
        )
        pulse.setRemovedOnCompletion_(True)
        content_layer.addAnimation_forKey_(pulse, "tray_capture_ack")

    def trayCaptureFlashDone_(self, timer) -> None:
        self._tray_capture_flash_timer = None
        self.hide(fade_duration=_TRAY_CAPTURE_FLASH_FADE_OUT_S)

    def _cancel_tray_capture_flash(self) -> None:
        if self._tray_capture_flash_timer is not None:
            self._tray_capture_flash_timer.invalidate()
            self._tray_capture_flash_timer = None

    # ── recovery mode ────────────────────────────────────────

    def show_recovery(self, text: str, on_dismiss=None, on_insert=None,
                      on_clipboard_toggle=None) -> None:
        """Enter recovery mode: three-column button layout.

        The overlay becomes interactive (accepts mouse events) and shows
        three equal columns: Dismiss | Insert | Clipboard.

        Parameters
        ----------
        text : str
            The transcribed text (displayed in Clipboard column after toggle).
        on_dismiss : callable, optional
            Called when Dismiss column is clicked.
        on_insert : callable, optional
            Called when Insert column is clicked.
        on_clipboard_toggle : callable, optional
            Called when Clipboard column is clicked.
        """
        if self._window is None:
            return
        self._cancel_tray_capture_flash()
        self._hide_and_release_preview_compositor_client()
        self._tray_mode = False

        # Clean up any existing recovery state
        self._teardown_recovery_views()
        self._cancel_fade()
        self._cancel_typewriter()

        self._recovery_mode = True
        self._on_dismiss_callback = on_dismiss
        self._on_insert_callback = on_insert
        self._on_clipboard_toggle_callback = on_clipboard_toggle

        # Hide the normal text/scroll views
        if self._scroll_view is not None:
            self._scroll_view.setHidden_(True)

        # Make window interactive
        self._window.setIgnoresMouseEvents_(False)

        # Reset to default height
        screen_frame = self._screen.frame()
        sw = screen_frame.size.width
        f = _OUTER_FEATHER
        x = (sw - _OVERLAY_WIDTH) / 2 - f
        self._window.setFrame_display_animate_(
            NSMakeRect(x, _OVERLAY_BOTTOM_MARGIN - f,
                       _OVERLAY_WIDTH + 2 * f, _OVERLAY_HEIGHT + 2 * f),
            True, False
        )
        self._content_view.setFrame_(
            NSMakeRect(f, f, _OVERLAY_WIDTH, _OVERLAY_HEIGHT)
        )
        self._reset_overlay_chrome_geometry(_OVERLAY_HEIGHT)

        # Keep the content view transparent and tint the existing SDF fill instead.
        self._content_view.layer().setBackgroundColor_(None)
        self._set_fill_override((0.1, 0.1, 0.12), _RECOVERY_BG_ALPHA)

        # Create three column buttons
        col_w = _OVERLAY_WIDTH / 3.0
        labels = ["Dismiss", "Insert", "Clipboard"]
        callbacks = [self._dismiss_clicked, self._insert_clicked,
                     self._clipboard_clicked]

        for i, (label, callback) in enumerate(zip(labels, callbacks)):
            btn_frame = NSMakeRect(col_w * i, 0, col_w, _OVERLAY_HEIGHT)
            btn = _RecoveryButton.alloc().initWithFrame_callback_(btn_frame, callback)
            btn.setWantsLayer_(True)
            btn.layer().setBackgroundColor_(
                NSColor.clearColor().CGColor()
            )

            # Label text view (centered in column, click-through)
            label_frame = NSMakeRect(4, (_OVERLAY_HEIGHT - 24) / 2, col_w - 8, 24)
            label_view = _ClickThroughTextView.alloc().initWithFrame_(label_frame)
            label_view.setEditable_(False)
            label_view.setSelectable_(False)
            label_view.setDrawsBackground_(False)
            label_view.setAlignment_(1)  # NSTextAlignmentCenter
            label_view.setTextColor_(
                NSColor.colorWithSRGBRed_green_blue_alpha_(
                    1.0, 1.0, 1.0, _RECOVERY_TEXT_ALPHA
                )
            )
            label_view.setFont_(
                NSFont.systemFontOfSize_weight_(_RECOVERY_LABEL_FONT_SIZE, 0.3)
            )
            label_view.setString_(label)

            btn.addSubview_(label_view)
            self._content_view.addSubview_(btn)
            self._recovery_buttons.append(btn)
            self._recovery_labels.append(label_view)

            # Add subtle divider between columns (not after last)
            if i < len(labels) - 1:
                div_x = col_w * (i + 1)
                div_frame = NSMakeRect(div_x - 0.5, 8, 1, _OVERLAY_HEIGHT - 16)
                div = NSView.alloc().initWithFrame_(div_frame)
                div.setWantsLayer_(True)
                div.layer().setBackgroundColor_(
                    NSColor.colorWithSRGBRed_green_blue_alpha_(
                        1.0, 1.0, 1.0, _RECOVERY_DIVIDER_ALPHA
                    ).CGColor()
                )
                self._content_view.addSubview_(div)
                self._recovery_dividers.append(div)

        # Show hint window below overlay: "press spacebar to dismiss"
        self._show_hint_window(sw)

        # Make overlay visible immediately (no fade-in for recovery)
        self._visible = True
        self._window.setAlphaValue_(1.0)
        self._window.orderFrontRegardless()

        # Entrance pop: expand slightly then ease back to normal size.
        # Communicates "I just appeared" on first show, and "paste failed,
        # I'm back" on re-entry after a failed retry.
        self._pop_entrance()

        logger.info("Recovery overlay shown")

    def bounce(self) -> None:
        """Subtle scale bounce — shrink to 97% then back to 100%.

        Used when spacebar retry fails (no text field). The overlay
        briefly contracts and snaps back, signalling "I tried but
        there's nowhere to paste."
        """
        if self._window is None or not self._recovery_mode:
            return

        from Quartz import CABasicAnimation

        content_layer = self._content_view.layer()

        shrink = CABasicAnimation.animationWithKeyPath_("transform.scale")
        shrink.setFromValue_(1.0)
        shrink.setToValue_(0.97)
        shrink.setDuration_(0.08)
        shrink.setAutoreverses_(True)
        shrink.setRemovedOnCompletion_(True)
        content_layer.addAnimation_forKey_(shrink, "bounce")

    def _pop_entrance(self) -> None:
        """Entrance pop: expand ~1mm on each side then ease back to normal.

        A quick overshoot that says "I just arrived" on first appearance,
        or "paste failed, I'm back" on re-entry. The ease-in on the
        return makes it feel like the overlay settles into place.
        """
        if self._window is None:
            return

        from Quartz import CABasicAnimation, CAMediaTimingFunction

        content_layer = self._content_view.layer()

        pop = CABasicAnimation.animationWithKeyPath_("transform.scale")
        pop.setFromValue_(1.015)  # ~1mm overshoot on a 600px overlay
        pop.setToValue_(1.0)
        pop.setDuration_(0.2)
        pop.setTimingFunction_(
            CAMediaTimingFunction.functionWithName_("easeIn")
        )
        pop.setRemovedOnCompletion_(True)
        content_layer.addAnimation_forKey_(pop, "pop_entrance")

    def start_insert_windup(self) -> None:
        """Shrink animation signaling an impending insert at cursor."""
        if self._window is None:
            return
        from Quartz import CABasicAnimation, CAMediaTimingFunction

        content_layer = self._content_view.layer()
        shrink = CABasicAnimation.animationWithKeyPath_("transform.scale")
        shrink.setFromValue_(1.0)
        shrink.setToValue_(0.97)
        shrink.setDuration_(0.35)
        shrink.setTimingFunction_(
            CAMediaTimingFunction.functionWithName_("easeIn")
        )
        shrink.setFillMode_("forwards")
        shrink.setRemovedOnCompletion_(False)
        content_layer.addAnimation_forKey_(shrink, "insert_windup")

    def cancel_insert_windup(self) -> None:
        """Cancel the wind-up and snap back to normal scale."""
        if self._window is None or self._content_view is None:
            return
        self._content_view.layer().removeAnimationForKey_("insert_windup")

    def dismiss_recovery(self) -> None:
        """Exit recovery mode and hide the overlay."""
        if not self._recovery_mode:
            return
        self._recovery_mode = False
        self._teardown_recovery_views()

        # Restore normal overlay state
        if self._scroll_view is not None:
            self._scroll_view.setHidden_(False)
        self._window.setIgnoresMouseEvents_(True)

        # Restore the standard preview fill and clear any recovery-specific tint.
        self._content_view.layer().setBackgroundColor_(None)
        self._clear_fill_override(opacity=_BG_ALPHA_MIN)

        self.order_out()

    def flash_insert_reject(self) -> None:
        """Brief visual signal that Insert was rejected (no text field)."""
        if not self._recovery_mode or len(self._recovery_buttons) < 2:
            return

        insert_btn = self._recovery_buttons[1]
        insert_label = self._recovery_labels[1]

        # Flash the insert column red briefly
        insert_btn.layer().setBackgroundColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(
                0.8, 0.2, 0.2, 0.15
            ).CGColor()
        )

        # Schedule reset
        def _reset_flash(timer):
            if insert_btn is not None:
                insert_btn.layer().setBackgroundColor_(
                    NSColor.clearColor().CGColor()
                )

        self._recovery_reject_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            _RECOVERY_REJECT_DURATION,
            _TimerCallback.alloc().initWithCallback_(_reset_flash),
            "fire:",
            None,
            False,
        )

    def set_clipboard_preview(self, preview_text: str) -> None:
        """Update the Clipboard column label to show a preview of swapped contents."""
        if not self._recovery_mode or len(self._recovery_labels) < 3:
            return
        clipboard_label = self._recovery_labels[2]
        truncated = _truncate_preview(preview_text)
        clipboard_label.setString_(truncated)
        clipboard_label.setFont_(
            NSFont.systemFontOfSize_weight_(_RECOVERY_LABEL_FONT_SIZE - 2, 0.0)
        )

    def reset_clipboard_label(self) -> None:
        """Reset the Clipboard column label back to 'Clipboard'."""
        if not self._recovery_mode or len(self._recovery_labels) < 3:
            return
        clipboard_label = self._recovery_labels[2]
        clipboard_label.setString_("Clipboard")
        clipboard_label.setFont_(
            NSFont.systemFontOfSize_weight_(_RECOVERY_LABEL_FONT_SIZE, 0.3)
        )

    # ── recovery internal helpers ────────────────────────────

    def _dismiss_clicked(self) -> None:
        if self._on_dismiss_callback is not None:
            self._on_dismiss_callback()

    def _insert_clicked(self) -> None:
        if self._on_insert_callback is not None:
            self._on_insert_callback()

    def _clipboard_clicked(self) -> None:
        if self._on_clipboard_toggle_callback is not None:
            self._on_clipboard_toggle_callback()

    def _show_hint_window(self, screen_width: float) -> None:
        """Create a small hint window below the overlay."""
        hint_y = _OVERLAY_BOTTOM_MARGIN - _RECOVERY_HINT_MARGIN - _RECOVERY_HINT_HEIGHT
        hint_x = (screen_width - _OVERLAY_WIDTH) / 2
        hint_frame = NSMakeRect(hint_x, hint_y, _OVERLAY_WIDTH, _RECOVERY_HINT_HEIGHT)

        self._recovery_hint_window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            hint_frame, 0, NSBackingStoreBuffered, False
        )
        self._recovery_hint_window.setLevel_(_OVERLAY_WINDOW_LEVEL)
        self._recovery_hint_window.setOpaque_(False)
        self._recovery_hint_window.setBackgroundColor_(NSColor.clearColor())
        self._recovery_hint_window.setIgnoresMouseEvents_(True)
        self._recovery_hint_window.setHasShadow_(False)
        self._recovery_hint_window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        hint_text = NSTextView.alloc().initWithFrame_(
            NSMakeRect(0, 0, _OVERLAY_WIDTH, _RECOVERY_HINT_HEIGHT)
        )
        hint_text.setEditable_(False)
        hint_text.setSelectable_(False)
        hint_text.setDrawsBackground_(False)
        hint_text.setAlignment_(1)  # center
        hint_text.setTextColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(
                1.0, 1.0, 1.0, _RECOVERY_HINT_ALPHA
            )
        )
        hint_text.setFont_(
            NSFont.systemFontOfSize_weight_(_RECOVERY_HINT_FONT_SIZE, 0.0)
        )
        hint_text.setString_("spacebar to retry  ·  shift+space to dismiss")

        wrapper = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, _OVERLAY_WIDTH, _RECOVERY_HINT_HEIGHT)
        )
        wrapper.addSubview_(hint_text)
        self._recovery_hint_window.setContentView_(wrapper)
        self._recovery_hint_window.setAlphaValue_(1.0)
        self._recovery_hint_window.orderFrontRegardless()

    def _teardown_recovery_views(self) -> None:
        """Remove all recovery-mode subviews and hint window."""
        for btn in self._recovery_buttons:
            btn.removeFromSuperview()
        self._recovery_buttons.clear()
        self._recovery_labels.clear()

        for div in self._recovery_dividers:
            div.removeFromSuperview()
        self._recovery_dividers.clear()

        if self._recovery_reject_timer is not None:
            self._recovery_reject_timer.invalidate()
            self._recovery_reject_timer = None

        if self._recovery_hint_window is not None:
            self._recovery_hint_window.orderOut_(None)
            self._recovery_hint_window = None

        self._on_dismiss_callback = None
        self._on_insert_callback = None
        self._on_clipboard_toggle_callback = None


# ── helper NSView subclass for clickable recovery columns ────


class _ClickableWindow(NSPanel):
    """Non-activating panel that accepts mouse events without stealing focus.

    Uses NSWindowStyleMaskNonactivatingPanel so clicks on the overlay
    do not activate the Spoke application or deactivate the target app.
    The user's text field keeps first responder status while they click
    recovery buttons.
    """

    def canBecomeKeyWindow(self):
        return False

    def canBecomeMainWindow(self):
        return False


class _RecoveryButton(NSView):
    """A transparent NSView that intercepts mouse clicks for recovery buttons."""

    def initWithFrame_callback_(self, frame, callback):
        self = objc.super(_RecoveryButton, self).initWithFrame_(frame)
        if self is None:
            return None
        self._callback = callback
        return self

    def acceptsFirstMouse_(self, event):
        return True

    def mouseDown_(self, event):
        if self._callback is not None:
            self._callback()


class _ClickThroughTextView(NSTextView):
    """NSTextView that passes mouse events through to its superview."""

    def hitTest_(self, point):
        return None


class _TimerCallback(NSObject):
    """NSObject wrapper for NSTimer → Python callable bridge."""

    def initWithCallback_(self, callback):
        self = objc.super(_TimerCallback, self).init()
        if self is None:
            return None
        self._callback = callback
        return self

    def fire_(self, timer):
        self._callback(timer)

"""Native presentation observations, distinct from GPU submission receipts."""

from __future__ import annotations

import ctypes
import math
import threading
from functools import lru_cache


@lru_cache(maxsize=1)
def _mach_timebase():
    class Timebase(ctypes.Structure):
        _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]

    info = Timebase()
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    function = library.mach_timebase_info
    function.argtypes = [ctypes.POINTER(Timebase)]
    function.restype = ctypes.c_int
    if function(ctypes.byref(info)) != 0 or not info.numer or not info.denom:
        raise RuntimeError("Mach timebase unavailable")
    return info.numer, info.denom


_registered_drawable_classes = set()
_metadata_lock = threading.Lock()


def source_display_seconds(attachments, key, *, numerator, denominator):
    if not attachments or key is None or numerator <= 0 or denominator <= 0:
        return None
    try:
        ticks = attachments[0].get(key)
        if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks <= 0:
            return None
        return ticks * numerator / denominator / 1_000_000_000
    except (TypeError, KeyError, IndexError, AttributeError):
        return None


def capture_display_seconds(sample_buffer, bridge):
    """Read SCK's mach-absolute display time; absence never becomes receipt time."""
    try:
        getter = bridge.get("CMSampleBufferGetSampleAttachmentsArray")
        key = bridge.get("SCStreamFrameInfoDisplayTime")
        if getter is None or key is None:
            return None
        attachments = getter(sample_buffer, False)

        numerator, denominator = _mach_timebase()
        return source_display_seconds(attachments, key, numerator=numerator, denominator=denominator)
    except Exception:
        return None


def _attach_presented_handler(drawable, callback):
    method = getattr(drawable, "addPresentedHandler_", None)
    if method is None:
        raise RuntimeError("Drawable presentation observer unavailable")
    # Spoke loads Metal manually, rather than importing pyobjc-framework-Metal.
    # The SDK's MTLDrawablePresentedHandler is void (^)(id<MTLDrawable>).
    if hasattr(method, "__metadata__"):
        import objc

        class_name = type(drawable).__name__.encode("ascii")
        with _metadata_lock:
            if class_name not in _registered_drawable_classes:
                objc.registerMetaDataForSelector(
                    class_name, b"addPresentedHandler:",
                    {"arguments": {2: {"callable": {
                        "retval": {"type": b"v"},
                        "arguments": {0: {"type": b"^v"}, 1: {"type": b"@"}},
                    }}}},
                )
                _registered_drawable_classes.add(class_name)
        method = drawable.addPresentedHandler_
    method(callback)


class PresentationTiming:
    def __init__(self, emit):
        self._emit = emit
        self._lock = threading.Lock()
        self._epoch = 0
        self.reset()

    def reset(self):
        with self._lock:
            self._epoch += 1
            self._displayed = self._not_presented = self._errors = 0
            self._first = self._last = None
            self._age_total = self._age_count = 0

    def observe(self, drawable, metadata):
        metadata = dict(metadata)
        with self._lock:
            epoch = self._epoch

        def presented(current):
            try:
                at = float(current.presentedTime())
                if not math.isfinite(at) or at < 0:
                    raise ValueError("Invalid native presentation timestamp")
                outcome = "displayed" if at > 0 else "not_presented"
                source = metadata.get("source_display_seconds")
                age = None
                if outcome == "displayed" and isinstance(source, (int, float)) and not isinstance(source, bool):
                    if math.isfinite(source) and 0 < source <= at:
                        age = (at - source) * 1000
                with self._lock:
                    stale = epoch != self._epoch
                    if not stale:
                        if outcome == "displayed":
                            self._displayed += 1
                            self._first = at if self._first is None else min(self._first, at)
                            self._last = at if self._last is None else max(self._last, at)
                            if age is not None:
                                self._age_total += age
                                self._age_count += 1
                        else:
                            self._not_presented += 1
                self._emit({**metadata, "outcome": outcome, "presented_host_seconds": at,
                            "source_age_ms": age, "stale_epoch": stale, "timing_epoch": epoch})
            except Exception as error:
                self._observer_error(metadata, epoch, error)

        try:
            _attach_presented_handler(drawable, presented)
            return True
        except Exception as error:
            self._observer_error(metadata, epoch, error)
            return False

    def _observer_error(self, metadata, epoch, error):
        with self._lock:
            if epoch == self._epoch:
                self._errors += 1
        try:
            self._emit({**metadata, "outcome": "observer_unavailable", "timing_epoch": epoch,
                        "error": f"{type(error).__name__}: {error}"})
        except Exception:
            pass

    def snapshot(self):
        with self._lock:
            span = self._last - self._first if self._first is not None else 0
            return {
                "displayed_frames": self._displayed,
                "displayed_fps": (self._displayed - 1) / span if span > 0 else 0.0,
                "not_presented_callbacks": self._not_presented,
                "presentation_observer_errors": self._errors,
                "source_age_samples": self._age_count,
                "avg_source_age_ms": self._age_total / self._age_count if self._age_count else None,
            }

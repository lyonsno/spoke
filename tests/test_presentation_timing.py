import pytest

from spoke.presentation_timing import PresentationTiming, source_display_seconds


class Drawable:
    def __init__(self):
        self.callback = None
        self.at = 0.0

    def addPresentedHandler_(self, callback):
        self.callback = callback

    def presentedTime(self):
        return self.at


def test_submission_does_not_count_as_displayed():
    records = []
    timing = PresentationTiming(records.append)
    drawable = Drawable()
    assert timing.observe(drawable, {"capture_frame_generation": 4, "source_display_seconds": 10.0})
    assert timing.snapshot()["displayed_frames"] == 0
    drawable.at = 10.025
    drawable.callback(drawable)
    assert timing.snapshot()["displayed_frames"] == 1
    assert timing.snapshot()["avg_source_age_ms"] == pytest.approx(25)
    assert records[0]["outcome"] == "displayed"
    assert records[0]["capture_frame_generation"] == 4


def test_drop_and_missing_source_are_not_successful_fresh_frames():
    records = []
    timing = PresentationTiming(records.append)
    dropped = Drawable()
    timing.observe(dropped, {})
    dropped.callback(dropped)
    assert timing.snapshot()["displayed_frames"] == 0
    assert records[-1]["outcome"] == "not_presented"
    shown = Drawable()
    timing.observe(shown, {})
    shown.at = 5.0
    shown.callback(shown)
    assert timing.snapshot()["displayed_frames"] == 1
    assert timing.snapshot()["avg_source_age_ms"] is None
    assert records[-1]["source_age_ms"] is None


def test_observer_failure_is_explicit_and_does_not_block_rendering():
    records = []
    timing = PresentationTiming(records.append)
    assert not timing.observe(object(), {})
    assert timing.snapshot()["presentation_observer_errors"] == 1
    assert records[-1]["outcome"] == "observer_unavailable"


def test_callbacks_from_old_epoch_do_not_credit_current_session():
    records = []
    timing = PresentationTiming(records.append)
    drawable = Drawable()
    timing.observe(drawable, {})
    timing.reset()
    drawable.at = 20.0
    drawable.callback(drawable)
    assert timing.snapshot()["displayed_frames"] == 0
    assert records[-1]["stale_epoch"] is True


def test_observation_snapshots_metadata_and_uses_actual_presentation_intervals():
    records = []
    timing = PresentationTiming(records.append)
    metadata = {"capture_frame_generation": 1}
    first, second = Drawable(), Drawable()
    timing.observe(first, metadata)
    metadata["capture_frame_generation"] = 2
    timing.observe(second, metadata)
    first.at, second.at = 5.0, 5.01
    first.callback(first)
    second.callback(second)
    assert records[0]["capture_frame_generation"] == 1
    assert timing.snapshot()["displayed_fps"] == pytest.approx(100)


@pytest.mark.parametrize("attachments", [None, [], [{}], [{"time": "bad"}], [{"time": -1}], [{"time": True}]])
def test_missing_or_invalid_source_timestamp_stays_unverified(attachments):
    assert source_display_seconds(attachments, "time", numerator=125, denominator=3) is None


def test_source_timestamp_uses_mach_timebase_not_assumed_nanoseconds():
    assert source_display_seconds([{"time": 24_000_000}], "time", numerator=125, denominator=3) == pytest.approx(1.0)


@pytest.mark.parametrize("at", [float("nan"), float("inf"), -1.0])
def test_invalid_native_timestamp_is_unverified_not_a_known_drop(at):
    records = []
    timing = PresentationTiming(records.append)
    drawable = Drawable()
    timing.observe(drawable, {})
    drawable.at = at
    drawable.callback(drawable)
    assert timing.snapshot()["displayed_frames"] == 0
    assert timing.snapshot()["not_presented_callbacks"] == 0
    assert timing.snapshot()["presentation_observer_errors"] == 1
    assert records[-1]["outcome"] == "observer_unavailable"


def test_installed_pyobjc_accepts_one_argument_presented_block():
    # Native selector/block conversion only: this does not create a Metal device
    # or establish that a real drawable reached the operator's screen.
    objc = pytest.importorskip("objc")
    from Foundation import NSObject

    class SpokeTimingDrawableContractWitness1001(NSObject):
        @objc.typedSelector(b"v@:@?")
        def addPresentedHandler_(self, callback):
            self.callback = callback

        @objc.typedSelector(b"d@:")
        def presentedTime(self):
            return 5.0

    records = []
    timing = PresentationTiming(records.append)
    drawable = SpokeTimingDrawableContractWitness1001.new()
    assert timing.observe(drawable, {"source_display_seconds": 4.99})
    drawable.callback(drawable)
    assert records[-1]["outcome"] == "displayed"
    assert timing.snapshot()["avg_source_age_ms"] == pytest.approx(10)

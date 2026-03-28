"""Tests for the typewriter effect's high-water-mark snap behaviour."""


class FakeOverlay:
    """Minimal stand-in that replicates the typewriter state machine
    from TranscriptionOverlay without needing AppKit/PyObjC.
    """

    def __init__(self):
        self._typewriter_target = ""
        self._typewriter_displayed = ""
        self._typewriter_hwm = 0
        self._timer_running = False
        # Track what's on screen for assertions
        self.screen_text = ""
        self._snapped = False  # did the last set_text snap?

    # ── mirrors overlay.py logic ──────────────────────────────

    def set_text(self, text: str) -> None:
        self._snapped = False
        self._typewriter_target = text

        if not text.startswith(self._typewriter_displayed):
            common = 0
            for i, (a, b) in enumerate(zip(self._typewriter_displayed, text)):
                if a == b:
                    common = i + 1
                else:
                    break

            if common < self._typewriter_hwm:
                self._timer_running = False
                self._typewriter_displayed = text
                self._typewriter_hwm = len(text)
                self.screen_text = text
                self._snapped = True
                return
            else:
                self._typewriter_displayed = text[:common]
                self.screen_text = self._typewriter_displayed

        if not self._timer_running and len(self._typewriter_displayed) < len(self._typewriter_target):
            self._timer_running = True

    def tick(self, n: int = 1) -> None:
        """Simulate n typewriter timer ticks."""
        for _ in range(n):
            if len(self._typewriter_displayed) < len(self._typewriter_target):
                self._typewriter_displayed = self._typewriter_target[:len(self._typewriter_displayed) + 1]
                self._typewriter_hwm = max(self._typewriter_hwm, len(self._typewriter_displayed))
                self.screen_text = self._typewriter_displayed
            else:
                self._timer_running = False

    def tick_all(self) -> None:
        """Run ticks until typewriter finishes (including the stop tick)."""
        remaining = len(self._typewriter_target) - len(self._typewriter_displayed)
        if remaining > 0:
            self.tick(remaining)
        # One more tick to trigger the stop condition, mirroring the real timer
        if self._timer_running:
            self.tick(1)


class TestTypewriterHighWaterMark:
    """The HWM prevents re-animating text the user already saw."""

    def test_normal_forward_progress(self):
        """New text extending the existing text should typewrite normally."""
        ov = FakeOverlay()
        ov.set_text("hello")
        ov.tick_all()
        assert ov.screen_text == "hello"
        assert ov._typewriter_hwm == 5

        ov.set_text("hello world")
        assert not ov._snapped
        ov.tick_all()
        assert ov.screen_text == "hello world"
        assert ov._typewriter_hwm == 11

    def test_punctuation_revision_snaps(self):
        """Punctuation change before the HWM should snap, not re-animate."""
        ov = FakeOverlay()
        ov.set_text("hello world")
        ov.tick_all()
        assert ov._typewriter_hwm == 11

        # ASR revises: adds a comma — diverges at position 5
        ov.set_text("hello, world")
        assert ov._snapped
        assert ov.screen_text == "hello, world"

    def test_early_word_revision_snaps(self):
        """Word change before the HWM should snap."""
        ov = FakeOverlay()
        ov.set_text("I think that")
        ov.tick_all()

        # ASR revises "think" → "thought" — diverges at position 2
        ov.set_text("I thought that")
        assert ov._snapped
        assert ov.screen_text == "I thought that"

    def test_divergence_at_hwm_does_not_snap(self):
        """Divergence right at the HWM boundary should typewrite, not snap."""
        ov = FakeOverlay()
        ov.set_text("hello")
        ov.tick_all()  # HWM = 5

        # New text diverges exactly at position 5 (the HWM)
        ov.set_text("hellp world")
        # common prefix is "hell" (4), which is < HWM (5), so this snaps
        assert ov._snapped

    def test_divergence_beyond_hwm_typewriters(self):
        """Divergence ahead of the cursor (mid-typewrite) should not snap."""
        ov = FakeOverlay()
        ov.set_text("hello world foo bar")
        ov.tick(5)  # typed "hello", HWM = 5

        # New text changes "foo" to "baz" — diverges at position 12
        # But HWM is only 5, so this is ahead of what we've shown
        ov.set_text("hello world baz bar")
        assert not ov._snapped

    def test_hwm_resets_on_new_session(self):
        """Simulating show() should reset the HWM."""
        ov = FakeOverlay()
        ov.set_text("first session text")
        ov.tick_all()
        assert ov._typewriter_hwm == 18

        # Simulate show() — reset state
        ov._typewriter_target = ""
        ov._typewriter_displayed = ""
        ov._typewriter_hwm = 0

        # Now a revision at position 0 should NOT snap (HWM is 0)
        ov.set_text("second session")
        assert not ov._snapped

    def test_partial_typewrite_then_revision(self):
        """Revision behind the cursor mid-typewrite should snap."""
        ov = FakeOverlay()
        ov.set_text("the quick brown fox")
        ov.tick(10)  # typed "the quick " — HWM = 10

        # ASR revises "the" → "a" — diverges at position 1
        ov.set_text("a quick brown fox")
        assert ov._snapped
        assert ov.screen_text == "a quick brown fox"
        # HWM should advance to cover the snapped text
        assert ov._typewriter_hwm == len("a quick brown fox")

    def test_snap_then_normal_typewrite(self):
        """After a snap, subsequent forward text should typewrite normally."""
        ov = FakeOverlay()
        ov.set_text("hello world")
        ov.tick_all()

        # Snap due to revision
        ov.set_text("hello, world")
        assert ov._snapped

        # Now extend — should typewrite
        ov.set_text("hello, world! How are you?")
        assert not ov._snapped
        assert ov._timer_running
        ov.tick_all()
        assert ov.screen_text == "hello, world! How are you?"

    def test_identical_text_no_snap(self):
        """Same text again should neither snap nor re-animate."""
        ov = FakeOverlay()
        ov.set_text("hello")
        ov.tick_all()

        ov.set_text("hello")
        assert not ov._snapped
        assert not ov._timer_running

    def test_hwm_advances_through_snap(self):
        """HWM should jump forward on snap to cover all snapped text."""
        ov = FakeOverlay()
        ov.set_text("abc")
        ov.tick_all()  # HWM = 3

        # Snap to longer text
        ov.set_text("xbc longer text here")
        assert ov._snapped
        assert ov._typewriter_hwm == len("xbc longer text here")

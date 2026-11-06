"""Window-global WezTerm lanes sidebar HUD using Spoke House choreography.

Pairs with WezTerm's window-global `overrides.window_padding = { left = '252pt' }`
so the terminal grid squeezes 252pt to the right across all tabs simultaneously
(zero per-tab split-panes, zero tab-switch SIGWINCH resizes, and zero occluded text).
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any

from . import house_optical_primitive as house
from .optical_lifecycle import retarget_progress_for_dismiss

logger = logging.getLogger(__name__)

WEZTERM_BIN = "/Applications/WezTerm.app/Contents/MacOS/wezterm"
WEZTERM_BUNDLE_ID = "com.github.wez.wezterm"
DEFAULT_STATE_PATH = Path.home() / ".local/state/epistaxis/wezterm-lanes-state.json"
DEFAULT_ENABLED_PATH = Path.home() / ".local/state/epistaxis/wezterm-sidebar-enabled"
AGY_TITLES_PATH = Path.home() / ".local/state/epistaxis/wezterm-agy-pane-titles.json"
AGY_CLI_DIR = Path.home() / ".gemini/antigravity-cli"
DAEMON_LOCK_PATH = Path("/tmp/spoke-wezterm-lanes-hud.lock")

SIDEBAR_WIDTH_PT = 252.0
TOP_RAIL_HEIGHT_PT = 25.0
FULLSCREEN_TOP_CLEARANCE_PT = 43.0
SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


@dataclass(frozen=True)
class LaneRow:
    index: int
    tab_id: int
    pane_id: str
    name: str
    state: str  # "working" | "done" | "idle"
    badge: str
    active: bool


_state_cache: dict[str, tuple[int, list[LaneRow], int]] = {}
_snapshot_enabled: dict[str, bool] = {}


def load_wezterm_lanes_state(
    state_path: Path | None = None,
) -> tuple[list[LaneRow], int]:
    path = state_path or DEFAULT_STATE_PATH
    key = str(path)
    prev = _state_cache.get(key)
    if not path.exists():
        if prev is not None:
            return prev[1], prev[2]
        return [], 0
    try:
        mtime_ns = path.stat().st_mtime_ns
    except Exception:
        mtime_ns = -1
    if prev is not None and mtime_ns != -1 and mtime_ns == prev[0] and prev[1]:
        return prev[1], prev[2]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        if prev is not None:
            return prev[1], prev[2]
        return [], 0
    if not isinstance(raw, dict):
        if prev is not None:
            return prev[1], prev[2]
        return [], 0
    scoped_enabled = raw.get("sidebar_enabled")
    if isinstance(scoped_enabled, bool):
        _snapshot_enabled[key] = scoped_enabled
    elif "sidebar_enabled" in raw:
        if _snapshot_enabled.get(key) is not False:
            logger.warning("Invalid sidebar_enabled in window snapshot; hiding sidebar")
        _snapshot_enabled[key] = False
    raw_lanes = raw.get("lanes")
    if not isinstance(raw_lanes, list):
        if "sidebar_enabled" in raw:
            _snapshot_enabled[key] = False
        return (prev[1], prev[2]) if prev is not None else ([], 0)
    try:
        lanes: list[LaneRow] = []
        for item in raw_lanes:
            if not isinstance(item, dict):
                continue
            lanes.append(
                LaneRow(
                    index=int(item.get("index", len(lanes) + 1)),
                    tab_id=int(item.get("tab_id", 0)),
                    pane_id=str(item.get("pane_id", "?")),
                    name=str(item.get("name", "untitled")),
                    state=str(item.get("state") or "idle"),
                    badge=str(item.get("badge") or ""),
                    active=bool(item.get("active", False)),
                )
            )
        updated_at = int(raw.get("updated_at", 0))
    except (TypeError, ValueError, OverflowError):
        if "sidebar_enabled" in raw:
            _snapshot_enabled[key] = False
        return (prev[1], prev[2]) if prev is not None else ([], 0)
    if "sidebar_enabled" not in raw:
        _snapshot_enabled.pop(key, None)

    if not lanes and prev is not None and prev[1]:
        return prev[1], prev[2]
    if lanes:
        _state_cache[key] = (mtime_ns, lanes, updated_at)
    return lanes, updated_at


_enabled_cache: dict[str, tuple[int, bool]] = {}


def is_wezterm_sidebar_enabled(enabled_path: Path | None = None) -> bool:
    if enabled_path is None:
        # The focused producer supplies rows and window-owned visibility in
        # one atomic snapshot. Keep the legacy flag for older producers only.
        load_wezterm_lanes_state()
        scoped = _snapshot_enabled.get(str(DEFAULT_STATE_PATH))
        if scoped is not None:
            return scoped
    path = enabled_path or DEFAULT_ENABLED_PATH
    key = str(path)
    try:
        st = path.stat()
    except OSError:
        _enabled_cache.pop(key, None)
        return False
    prev = _enabled_cache.get(key)
    if prev is not None and prev[0] == st.st_mtime_ns:
        return prev[1]
    try:
        val = path.read_text(encoding="utf-8").strip() == "1"
        _enabled_cache[key] = (st.st_mtime_ns, val)
        return val
    except Exception:
        return prev[1] if prev is not None else False


def compute_sidebar_frame_for_wezterm_bounds(
    wezterm_bounds: dict[str, float],
    screen_height: float,
    *,
    width_pt: float = SIDEBAR_WIDTH_PT,
    top_rail_pt: float = TOP_RAIL_HEIGHT_PT,
) -> tuple[float, float, float, float]:
    """Convert Quartz top-left window bounds into AppKit bottom-left sidebar frame."""
    wx = float(wezterm_bounds.get("X", 0.0))
    wy = float(wezterm_bounds.get("Y", 0.0))
    wh = float(wezterm_bounds.get("Height", 900.0))
    appkit_y = float(screen_height) - (wy + wh)
    effective_top_rail = (
        max(float(top_rail_pt), FULLSCREEN_TOP_CLEARANCE_PT - wy)
        if wy < 10.0
        else float(top_rail_pt)
    )
    sidebar_h = max(120.0, wh - effective_top_rail)
    return (wx, appkit_y, float(width_pt), sidebar_h)


class SidebarHouseTransition:
    """Wall-clock House choreography for summoning and dismissing the sidebar."""

    def __init__(self) -> None:
        self.phase: str = "hidden"
        self.started_at: float = 0.0
        self.start_progress: float = 0.0
        self._dismiss_text_ceiling: dict[str, float] | None = None
        self._last_exposed_text: dict[str, float] = {
            "width_frac": 0.0,
            "height_frac": 0.0,
            "alpha": 0.0,
        }

    def _position(self, now: float) -> tuple[str, float]:
        elapsed = max(0.0, float(now) - self.started_at)
        if self.phase == "summon":
            progress = house.clamp01(
                self.start_progress + elapsed / house.OPTICAL_MATERIALIZATION_S
            )
            return ("rest" if progress >= 1.0 else "summon", progress)
        if self.phase == "dismiss":
            remaining = (
                self.start_progress * house.OPTICAL_MATERIALIZATION_DISMISS_S - elapsed
            )
            if remaining > 0.0:
                return (
                    "dismiss",
                    house.clamp01(remaining / house.OPTICAL_MATERIALIZATION_DISMISS_S),
                )
            return ("hidden", 0.0)
        return (self.phase, 1.0 if self.phase == "rest" else 0.0)

    def summon(self, now: float) -> None:
        phase, progress = self._position(now)
        if phase == "dismiss":
            self.start_progress = retarget_progress_for_dismiss(progress).start_progress
        elif phase == "summon":
            self.start_progress = progress
        else:
            self.start_progress = 0.0
        self.phase = "summon"
        self.started_at = float(now)
        self._dismiss_text_ceiling = None

    def dismiss(self, now: float) -> None:
        phase, progress = self._position(now)
        if phase in {"hidden", "dismiss"}:
            return
        if self.phase == "summon":
            self._dismiss_text_ceiling = dict(self._last_exposed_text)
        else:
            self._dismiss_text_ceiling = None
        self.phase = "dismiss"
        self.started_at = float(now)
        self.start_progress = progress

    def sample(self, now: float) -> dict[str, float | str]:
        phase, progress = self._position(now)
        self.phase = phase
        if phase in {"dismiss", "hidden"}:
            body = house.house_dismiss_materialization_fill_state(progress)
            text = house.house_dismiss_text_collapse_state(progress)
            if self._dismiss_text_ceiling is not None:
                text = {
                    k: min(v, self._dismiss_text_ceiling.get(k, v))
                    for k, v in text.items()
                }
        else:
            body = house.house_materialization_fill_state(progress)
            text = {
                "width_frac": 1.0,
                "height_frac": float(body["height_frac"]),
                "alpha": float(
                    house.optical_entrance_text_ready(
                        progress, float(body["height_frac"])
                    )
                ),
            }
        self._last_exposed_text = dict(text)
        return {
            "phase": phase,
            "progress": progress,
            "height_frac": float(body["height_frac"]),
            "width_frac": float(text["width_frac"]),
            "opacity": float(body["opacity"]),
            "text_alpha": float(text["alpha"]),
        }


def _run_subprocess(cmd: list[str]) -> int:
    res = subprocess.run(
        cmd,
        capture_output=True,
        timeout=2.5,
        check=False,
    )
    return int(res.returncode)


_activate_lock = threading.Lock()


def activate_wezterm_lane(pane_id: str, *, sync: bool = False) -> bool:
    if not pane_id or not str(pane_id).isdigit():
        return False
    cmd = [WEZTERM_BIN, "cli", "activate-pane", "--pane-id", str(pane_id)]

    def _worker() -> bool:
        if not _activate_lock.acquire(blocking=False):
            return False
        try:
            return _run_subprocess(cmd) == 0
        except Exception:
            return False
        finally:
            _activate_lock.release()

    if sync:
        return _worker()
    threading.Thread(target=_worker, daemon=True).start()
    return True


_agy_titles_lock = threading.Lock()
_pid_to_pane_cache: dict[int, str] = {}
_last_agy_dir_sig: tuple[int, int, tuple[str, ...]] | None = None
_last_agy_scan_mono: float = 0.0


def refresh_live_agy_pane_titles(
    known_pane_ids: tuple[str, ...] = (),
    *,
    force: bool = False,
) -> None:
    """Discover each running agy pane's live active conversation title."""
    global _last_agy_dir_sig, _last_agy_scan_mono
    if not _agy_titles_lock.acquire(blocking=False):
        return
    try:
        presence_dir = AGY_CLI_DIR / "presence"
        annotations_dir = AGY_CLI_DIR / "annotations"
        p_mtime = presence_dir.stat().st_mtime_ns if presence_dir.exists() else 0
        a_mtime = annotations_dir.stat().st_mtime_ns if annotations_dir.exists() else 0
        sig = (p_mtime, a_mtime, known_pane_ids)
        now_mono = time.monotonic()
        if (
            not force
            and _last_agy_dir_sig == sig
            and (now_mono - _last_agy_scan_mono) < 60.0
            and AGY_TITLES_PATH.exists()
        ):
            return
        _last_agy_dir_sig = sig
        _last_agy_scan_mono = now_mono

        ps_out = subprocess.check_output(
            ["/bin/ps", "-axo", "pid,command"],
            text=True,
            timeout=1.5,
            stderr=subprocess.DEVNULL,
        )
        agy_pids = [
            int(m.group(1))
            for line in ps_out.splitlines()
            if (m := re.match(r"^\s*(\d+)\s+(agy\b.*)$", line))
        ]
        if not agy_pids:
            return

        lsof_out = subprocess.check_output(
            ["/usr/sbin/lsof", "-p", ",".join(map(str, agy_pids))],
            text=True,
            timeout=2.0,
            stderr=subprocess.DEVNULL,
        )
        pid_locks: dict[int, str] = {}
        pid_brains: dict[int, list[str]] = {}
        for line in lsof_out.splitlines():
            if m_lock := re.search(
                r"^agy\s+(\d+)\s+.*?/presence/([0-9a-f-]{36})\.lock", line
            ):
                pid_locks[int(m_lock.group(1))] = m_lock.group(2)
            if m_brain := re.search(
                r"^agy\s+(\d+)\s+.*?/brain/([0-9a-f-]{36})(?:/|$)", line
            ):
                pid_brains.setdefault(int(m_brain.group(1)), []).append(
                    m_brain.group(2)
                )

        meta_path = AGY_CLI_DIR / "cache/conversation_metadata.json"
        meta = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.exists()
            else {}
        )

        pane_titles: dict[str, str] = {}
        for pid in agy_pids:
            pane_id = _pid_to_pane_cache.get(pid)
            if not pane_id:
                env_out = subprocess.check_output(
                    ["/bin/ps", "eww", "-p", str(pid)],
                    text=True,
                    timeout=1.0,
                    stderr=subprocess.DEVNULL,
                )
                if pm := re.search(r"\bWEZTERM_PANE=(\d+)", env_out):
                    pane_id = pm.group(1)
                    _pid_to_pane_cache[pid] = pane_id
            if not pane_id:
                continue

            cid = pid_locks.get(pid)
            if not cid and pid in pid_brains:
                cids = list(set(pid_brains[pid]))
                cids.sort(
                    key=lambda c: (AGY_CLI_DIR / "brain" / c).stat().st_mtime
                    if (AGY_CLI_DIR / "brain" / c).exists()
                    else 0.0,
                    reverse=True,
                )
                cid = cids[0]

            if not cid:
                pane_titles[pane_id] = "agy (new)"
                continue

            title: str | None = None
            ann_path = AGY_CLI_DIR / "annotations" / f"{cid}.pbtxt"
            if ann_path.exists():
                if tm := re.search(
                    r'title:\s*"([^"]+)"',
                    ann_path.read_text(encoding="utf-8", errors="ignore"),
                ):
                    title = tm.group(1)
            if not title and isinstance(meta.get(cid), dict):
                title = meta[cid].get("title")
            if title:
                clean = re.sub(r'["\\\r\n\t]+', " ", title).strip()
                if clean:
                    pane_titles[pane_id] = clean

        AGY_TITLES_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = AGY_TITLES_PATH.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(pane_titles, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        tmp_path.replace(AGY_TITLES_PATH)
    except Exception:
        pass
    finally:
        _agy_titles_lock.release()


def query_wezterm_window_bounds() -> dict[str, float] | None:
    """Return the frontmost on-screen WezTerm window bounds in Quartz coordinates."""
    try:
        import Quartz

        win_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly
            | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID,
        )
        # Quartz returns windows from front to back. The focused producer owns
        # the rows, so geometry must follow that window even when it is smaller.
        for w in win_list or ():
            owner = str(w.get("kCGWindowOwnerName") or "").lower()
            if "wezterm" not in owner:
                continue
            if int(w.get("kCGWindowLayer", 0)) != 0:
                continue
            bounds = w.get("kCGWindowBounds")
            if not bounds:
                continue
            width = float(bounds.get("Width", 0.0))
            height = float(bounds.get("Height", 0.0))
            if width >= 400.0 and height >= 250.0:
                return {
                    "X": float(bounds.get("X", 0.0)),
                    "Y": float(bounds.get("Y", 0.0)),
                    "Width": width,
                    "Height": height,
                }
        return None
    except Exception:
        return None


def is_wezterm_frontmost() -> bool:
    try:
        from AppKit import NSWorkspace

        front = NSWorkspace.sharedWorkspace().frontmostApplication()
        if front is None:
            return False
        bid = str(front.bundleIdentifier() or "")
        name = str(front.localizedName() or "").lower()
        return bid == WEZTERM_BUNDLE_ID or "wezterm" in name
    except Exception:
        return True


def build_sidebar_hud_controller() -> Any:
    """Build the AppKit WezTermLanesSidebarHUD NSObject controller."""
    import objc
    from AppKit import (
        NSBackingStoreBuffered,
        NSButton,
        NSColor,
        NSFont,
        NSPanel,
        NSScreen,
        NSScrollView,
        NSTextField,
        NSView,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSWindowCollectionBehaviorStationary,
        NSWindowStyleMaskNonactivatingPanel,
    )
    from Foundation import NSMakeRect, NSObject, NSRunLoop, NSTimer
    from Quartz import (
        CALayer,
        CAShapeLayer,
        CATransaction,
        CGPathCreateWithRoundedRect,
    )

    _NSWindowStyleMaskBorderless = 0
    _WINDOW_LEVEL_ACTIVE = 26
    _HEADER_H = 44.0
    _FOOTER_H = 26.0
    _ROW_H = 28.0
    _PAD_X = 10.0

    def _make_label(
        text: str,
        frame: Any,
        *,
        size: float = 12.0,
        color: Any,
        bold: bool = False,
        align: int = 0,
    ) -> Any:
        lbl = NSTextField.alloc().initWithFrame_(frame)
        lbl.setStringValue_(text)
        lbl.setBezeled_(False)
        lbl.setDrawsBackground_(False)
        lbl.setEditable_(False)
        lbl.setSelectable_(False)
        lbl.setFont_(
            NSFont.monospacedSystemFontOfSize_weight_(size, 0.4 if bold else 0.0)
        )
        lbl.setTextColor_(color)
        lbl.setAlignment_(align)
        lbl.setLineBreakMode_(4)  # NSLineBreakByTruncatingTail
        return lbl

    class _NonActivatingSidebarPanel(NSPanel):
        def canBecomeKeyWindow(self) -> bool:
            return False

        def canBecomeMainWindow(self) -> bool:
            return False

    class WezTermLanesSidebarHUD(NSObject):
        def init(self) -> Any:
            self = objc.super(WezTermLanesSidebarHUD, self).init()
            if self is None:
                return None
            self._panel: Any = None
            self._body_layer: Any = None
            self._foreground: Any = None
            self._text_mask: Any = None
            self._header_title: Any = None
            self._header_counts: Any = None
            self._scroll_view: Any = None
            self._document_view: Any = None
            self._footer_label: Any = None
            self._transition = SidebarHouseTransition()
            self._anim_timer: Any = None
            self._poll_timer: Any = None
            self._logical_visible = False
            self._poll_tick = 0
            self._spin_tick = 0
            self._last_signature: Any = None
            self._spinner_labels: list[Any] = []
            self._last_agy_scan_mono = 0.0
            self._last_frame_tuple: tuple[float, float, float, float] | None = None
            self._last_plane_sample: dict[str, float | str] | None = None
            return self

        @objc.python_method
        def setup(self) -> None:
            if self._panel is not None:
                return
            screen = NSScreen.mainScreen()
            s_frame = (
                screen.frame() if screen is not None else NSMakeRect(0, 0, 1708, 1117)
            )
            init_rect = NSMakeRect(
                0.0, 0.0, SIDEBAR_WIDTH_PT, float(s_frame.size.height) - TOP_RAIL_HEIGHT_PT
            )
            style = _NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
            panel = _NonActivatingSidebarPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                init_rect,
                style,
                NSBackingStoreBuffered,
                False,
            )
            panel.setLevel_(_WINDOW_LEVEL_ACTIVE)
            panel.setOpaque_(False)
            panel.setHasShadow_(False)
            panel.setBackgroundColor_(NSColor.clearColor())
            panel.setFloatingPanel_(True)
            panel.setBecomesKeyOnlyIfNeeded_(True)
            panel.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces
                | NSWindowCollectionBehaviorStationary
                | NSWindowCollectionBehaviorFullScreenAuxiliary
            )

            root = panel.contentView()
            root.setWantsLayer_(True)
            self._body_layer = CALayer.layer()
            self._body_layer.setFrame_(root.bounds())
            self._body_layer.setBackgroundColor_(
                NSColor.colorWithSRGBRed_green_blue_alpha_(
                    0.055, 0.062, 0.075, 0.98
                ).CGColor()
            )
            root.layer().addSublayer_(self._body_layer)

            fg = NSView.alloc().initWithFrame_(root.bounds())
            fg.setWantsLayer_(True)
            root.addSubview_(fg)
            self._foreground = fg
            self._text_mask = CAShapeLayer.layer()
            fg.layer().setMask_(self._text_mask)

            h = float(init_rect.size.height)
            w = SIDEBAR_WIDTH_PT

            self._header_title = _make_label(
                "LANES",
                NSMakeRect(_PAD_X, h - 24.0, w - 2 * _PAD_X, 18.0),
                size=11.5,
                bold=True,
                color=NSColor.colorWithSRGBRed_green_blue_alpha_(
                    0.51, 0.55, 0.97, 1.0
                ),
            )
            fg.addSubview_(self._header_title)

            self._header_counts = _make_label(
                "",
                NSMakeRect(_PAD_X, h - 40.0, w - 2 * _PAD_X, 15.0),
                size=10.5,
                bold=False,
                color=NSColor.colorWithSRGBRed_green_blue_alpha_(
                    0.62, 0.68, 0.75, 1.0
                ),
            )
            fg.addSubview_(self._header_counts)

            scroll_h = max(60.0, h - _HEADER_H - _FOOTER_H)
            sv = NSScrollView.alloc().initWithFrame_(
                NSMakeRect(0.0, _FOOTER_H, w, scroll_h)
            )
            sv.setHasVerticalScroller_(False)
            sv.setDrawsBackground_(False)
            doc = NSView.alloc().initWithFrame_(NSMakeRect(0.0, 0.0, w, scroll_h))
            sv.setDocumentView_(doc)
            fg.addSubview_(sv)
            self._scroll_view = sv
            self._document_view = doc

            self._footer_label = _make_label(
                "● live · tab (pane) · ⌘B hide",
                NSMakeRect(_PAD_X, 6.0, w - 2 * _PAD_X, 15.0),
                size=10.0,
                color=NSColor.colorWithSRGBRed_green_blue_alpha_(
                    0.45, 0.52, 0.60, 1.0
                ),
            )
            fg.addSubview_(self._footer_label)

            self._panel = panel
            self._apply_planes(
                {"phase": "hidden", "height_frac": 0.0, "width_frac": 1.0, "opacity": 0.0, "text_alpha": 0.0}
            )

        @objc.python_method
        def start(self) -> None:
            self.setup()
            self._poll_timer = (
                NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                    0.12, self, "pollTick:", None, True
                )
            )
            NSRunLoop.currentRunLoop().addTimer_forMode_(
                self._poll_timer, "NSRunLoopCommonModes"
            )
            self.pollTick_(None)

        @objc.python_method
        def cleanup(self) -> None:
            self._stop_anim_timer()
            if self._poll_timer is not None:
                self._poll_timer.invalidate()
                self._poll_timer = None
            if self._panel is not None:
                self._panel.orderOut_(None)

        @objc.python_method
        def _sync_geometry(self) -> bool:
            if self._panel is None:
                return False
            screen = NSScreen.mainScreen()
            screen_h = float(screen.frame().size.height) if screen is not None else 1117.0
            wb = query_wezterm_window_bounds()
            if wb is None:
                return False
            frame_tuple = compute_sidebar_frame_for_wezterm_bounds(
                wb, screen_h, width_pt=SIDEBAR_WIDTH_PT, top_rail_pt=TOP_RAIL_HEIGHT_PT
            )
            if frame_tuple != self._last_frame_tuple:
                self._last_frame_tuple = frame_tuple
                x, y, w, h = frame_tuple
                self._panel.setFrame_display_(NSMakeRect(x, y, w, h), True)
                root_bounds = NSMakeRect(0.0, 0.0, w, h)
                self._body_layer.setFrame_(root_bounds)
                self._foreground.setFrame_(root_bounds)
                self._header_title.setFrame_(
                    NSMakeRect(_PAD_X, h - 24.0, w - 2 * _PAD_X, 18.0)
                )
                self._header_counts.setFrame_(
                    NSMakeRect(_PAD_X, h - 40.0, w - 2 * _PAD_X, 15.0)
                )
                scroll_h = max(60.0, h - _HEADER_H - _FOOTER_H)
                self._scroll_view.setFrame_(NSMakeRect(0.0, _FOOTER_H, w, scroll_h))
                self._last_signature = None
                if self._last_plane_sample is not None:
                    self._apply_planes(self._last_plane_sample)
            return True

        @objc.python_method
        def _apply_planes(self, sample: dict[str, float | str]) -> None:
            if self._foreground is None or self._body_layer is None:
                return
            self._last_plane_sample = dict(sample)
            CATransaction.begin()
            CATransaction.setDisableActions_(True)
            try:
                h_frac = max(0.001, float(sample["height_frac"]))
                w_frac = max(0.001, float(sample["width_frac"]))
                opacity = float(sample["opacity"])
                t_alpha = float(sample["text_alpha"])
                self._body_layer.setOpacity_(opacity)
                size = self._foreground.bounds().size
                w = float(size.width) * w_frac
                h = float(size.height) * h_frac
                rect = (((float(size.width) - w) / 2.0, (float(size.height) - h) / 2.0), (w, h))
                self._text_mask.setPath_(
                    CGPathCreateWithRoundedRect(rect, min(6.0, h / 2.0), min(6.0, h / 2.0), None)
                )
                self._foreground.layer().setOpacity_(min(opacity, t_alpha))
            finally:
                CATransaction.commit()

        @objc.python_method
        def _ensure_anim_timer(self) -> None:
            if self._anim_timer is not None:
                return
            self._anim_timer = (
                NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                    1.0 / 60.0, self, "animStep:", None, True
                )
            )
            for mode in ("NSRunLoopCommonModes", "NSEventTrackingRunLoopMode"):
                NSRunLoop.currentRunLoop().addTimer_forMode_(self._anim_timer, mode)

        @objc.python_method
        def _stop_anim_timer(self) -> None:
            if self._anim_timer is not None:
                self._anim_timer.invalidate()
                self._anim_timer = None

        def animStep_(self, _timer: Any) -> None:
            sample = self._transition.sample(time.perf_counter())
            self._apply_planes(sample)
            phase = sample["phase"]
            if phase == "rest":
                self._stop_anim_timer()
            elif phase == "hidden":
                self._stop_anim_timer()
                if self._panel is not None:
                    self._panel.orderOut_(None)

        def pollTick_(self, _timer: Any) -> None:
            self._poll_tick += 1
            want_enabled = is_wezterm_sidebar_enabled()
            frontmost = is_wezterm_frontmost() if want_enabled or self._logical_visible else False
            want_visible = want_enabled and frontmost

            if want_visible and not self._logical_visible:
                if self._sync_geometry():
                    self._logical_visible = True
                    self._transition.phase = "rest"
                    self._apply_planes(
                        {
                            "phase": "rest",
                            "height_frac": 1.0,
                            "width_frac": 1.0,
                            "opacity": 1.0,
                            "text_alpha": 1.0,
                        }
                    )
                    lanes, _updated = load_wezterm_lanes_state()
                    self._render_lanes(lanes)
                    self._panel.setIgnoresMouseEvents_(False)
                    self._panel.orderFrontRegardless()
                return
            elif not want_visible and self._logical_visible:
                self._logical_visible = False
                self._transition.phase = "hidden"
                self._stop_anim_timer()
                self._panel.setIgnoresMouseEvents_(True)
                self._panel.orderOut_(None)
                return

            if not self._logical_visible and self._transition.phase == "hidden":
                return

            if self._poll_tick % 4 != 0:
                return

            self._sync_geometry()
            lanes, _updated = load_wezterm_lanes_state()
            now_mono = time.monotonic()
            if now_mono - self._last_agy_scan_mono >= 4.0:
                self._last_agy_scan_mono = now_mono
                pane_sig = tuple(sorted(ln.pane_id for ln in lanes))
                threading.Thread(
                    target=refresh_live_agy_pane_titles,
                    args=(pane_sig,),
                    daemon=True,
                ).start()

            self._render_lanes(lanes)

        @objc.python_method
        def _render_lanes(self, lanes: list[LaneRow]) -> None:
            if self._document_view is None:
                return
            any_working = any(ln.state == "working" for ln in lanes)
            if any_working:
                self._spin_tick += 1
            spin_char = SPINNER_FRAMES[self._spin_tick % len(SPINNER_FRAMES)]

            sig = tuple(
                (ln.index, ln.pane_id, ln.name, ln.state, ln.badge, ln.active)
                for ln in lanes
            )
            if sig == self._last_signature:
                if any_working:
                    for lbl in self._spinner_labels:
                        lbl.setStringValue_(spin_char)
                return

            self._last_signature = sig
            self._spinner_labels = []
            for sub in list(self._document_view.subviews()):
                sub.removeFromSuperview()

            working_n = sum(1 for ln in lanes if ln.state == "working")
            done_n = sum(1 for ln in lanes if ln.state == "done")
            self._header_title.setStringValue_(f"LANES ({len(lanes)})")
            self._header_counts.setStringValue_(
                f"⟳ {working_n} working   ✔ {done_n} done"
            )

            w = SIDEBAR_WIDTH_PT
            scroll_h = float(self._scroll_view.contentSize().height)
            doc_h = max(scroll_h, len(lanes) * _ROW_H + 8.0)
            self._document_view.setFrame_(NSMakeRect(0.0, 0.0, w, doc_h))

            for idx, ln in enumerate(lanes):
                y = doc_h - (idx + 1) * _ROW_H
                if ln.active:
                    bg = NSView.alloc().initWithFrame_(
                        NSMakeRect(4.0, y + 1.0, w - 8.0, _ROW_H - 2.0)
                    )
                    bg.setWantsLayer_(True)
                    bg.layer().setCornerRadius_(5.0)
                    bg.layer().setBackgroundColor_(
                        NSColor.colorWithSRGBRed_green_blue_alpha_(
                            0.14, 0.23, 0.20, 1.0
                        ).CGColor()
                    )
                    self._document_view.addSubview_(bg)

                prefix = f"{'▸' if ln.active else ' '}{ln.index:>2} ({ln.pane_id})"
                self._document_view.addSubview_(
                    _make_label(
                        prefix,
                        NSMakeRect(6.0, y + 5.0, 62.0, 18.0),
                        size=10.5,
                        bold=ln.active,
                        color=NSColor.colorWithSRGBRed_green_blue_alpha_(
                            0.55, 0.62, 0.70, 1.0
                        ),
                    )
                )

                if ln.state == "working":
                    icon_txt = spin_char
                    icon_col = NSColor.colorWithSRGBRed_green_blue_alpha_(
                        0.96, 0.62, 0.04, 1.0
                    )
                elif ln.state == "done":
                    icon_txt = "✔"
                    icon_col = NSColor.colorWithSRGBRed_green_blue_alpha_(
                        0.23, 0.51, 0.96, 1.0
                    )
                else:
                    icon_txt = "·"
                    icon_col = NSColor.colorWithSRGBRed_green_blue_alpha_(
                        0.40, 0.45, 0.52, 1.0
                    )

                icon_lbl = _make_label(
                    icon_txt,
                    NSMakeRect(68.0, y + 5.0, 16.0, 18.0),
                    size=11.5,
                    bold=True,
                    color=icon_col,
                )
                if ln.state == "working":
                    self._spinner_labels.append(icon_lbl)
                self._document_view.addSubview_(icon_lbl)

                badge_w = 26.0 if ln.badge else 0.0
                name_w = w - 86.0 - badge_w - 6.0
                name_col = (
                    NSColor.colorWithSRGBRed_green_blue_alpha_(0.56, 0.96, 0.80, 1.0)
                    if ln.active
                    else NSColor.colorWithSRGBRed_green_blue_alpha_(
                        0.88, 0.91, 0.94, 1.0
                    )
                    if ln.state != "idle"
                    else NSColor.colorWithSRGBRed_green_blue_alpha_(
                        0.52, 0.58, 0.66, 1.0
                    )
                )
                self._document_view.addSubview_(
                    _make_label(
                        ln.name,
                        NSMakeRect(84.0, y + 5.0, name_w, 18.0),
                        size=11.5,
                        bold=ln.active,
                        color=name_col,
                    )
                )

                if ln.badge:
                    self._document_view.addSubview_(
                        _make_label(
                            ln.badge,
                            NSMakeRect(w - 32.0, y + 5.0, 26.0, 18.0),
                            size=10.0,
                            bold=True,
                            align=2,
                            color=NSColor.colorWithSRGBRed_green_blue_alpha_(
                                0.22, 0.74, 0.97, 1.0
                            ),
                        )
                    )

                btn = NSButton.buttonWithTitle_target_action_("", self, "selectLane:")
                btn.setFrame_(NSMakeRect(0.0, y, w, _ROW_H))
                btn.setBordered_(False)
                btn.setIdentifier_(str(ln.pane_id))
                self._document_view.addSubview_(btn)

                if ln.active:
                    self._document_view.scrollRectToVisible_(
                        NSMakeRect(0.0, y, w, _ROW_H)
                    )

        def selectLane_(self, sender: Any) -> None:
            pane_id = str(sender.identifier() or "")
            activate_wezterm_lane(pane_id)

    return WezTermLanesSidebarHUD.alloc().init()


def ensure_daemon_running() -> None:
    """Spawn the single-instance WezTermLanesSidebarHUD daemon if not already running."""
    try:
        fd = os.open(str(DAEMON_LOCK_PATH), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            os.close(fd)
            return
        os.close(fd)
    except Exception:
        return

    subprocess.Popen(
        [sys.executable, "-m", "spoke.wezterm_lanes_sidebar"],
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def main() -> int:
    if "--ensure-daemon" in sys.argv:
        ensure_daemon_running()
        return 0

    lock_fd = os.open(str(DAEMON_LOCK_PATH), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return 0

    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    hud = build_sidebar_hud_controller()
    hud.start()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

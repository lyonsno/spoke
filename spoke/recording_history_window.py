"""Native recording library. Metadata refresh and replay never own live capture."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
import json
import logging
import threading
import time

import objc
import AppKit as AK
from Foundation import NSData, NSIndexSet, NSMakeRect, NSObject, NSTimer, NSURL

from .recording_history import delivery_label, duration_label, first_line, run_retranscription
from .window_focus import WindowFocus, restore_window_focus, window_owns_key
from .house_window import HouseWindowShell

logger = logging.getLogger(__name__)


def _label(text, frame, size=13, secondary=False):
    view = AK.NSTextField.labelWithString_(text)
    view.setFrame_(frame)
    view.setFont_(AK.NSFont.systemFontOfSize_(size))
    if secondary:
        view.setTextColor_(AK.NSColor.secondaryLabelColor())
    view.setLineBreakMode_(4)
    return view


def _date_label(value):
    try:
        return datetime.fromisoformat(value).astimezone().strftime("%b %d, %I:%M:%S %p")
    except (TypeError, ValueError):
        return "Unknown capture time"


class RecordingHistoryWindow(HouseWindowShell):
    def initWithDelegate_(self, delegate):
        self = objc.super(RecordingHistoryWindow, self).init()
        if self is None:
            return None
        self._delegate = delegate
        self._spool = delegate._audio_spool
        self._window = None
        self._records = []
        self._filtered = []
        self._selected_id = None
        self._attempt_id = None
        self._refreshing = False
        self._loading_partial = False
        self._load_failed = False
        self._timer = None
        self._sound = None
        self._previous_app = None
        self._focus_return = None
        self._replays = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spoke-history")
        self._busy = {}
        self._replay_outcomes = {}
        self._replay_generation = 0
        self._load_generation = 0
        self._record_overrides = {}
        return self

    def setup(self):
        if self._window is not None:
            return
        window = AK.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 980, 650), 1 | 2 | 4 | 8, AK.NSBackingStoreBuffered, False,
        )
        window.setTitle_("Spoke Recordings")
        window.setMinSize_((860, 540))
        window.setReleasedWhenClosed_(False)
        window.setDelegate_(self)
        window.center()
        content = window.contentView()
        self._search = AK.NSSearchField.alloc().initWithFrame_(NSMakeRect(18, 600, 290, 28))
        self._search.setPlaceholderString_("Search recordings")
        self._search.setDelegate_(self)
        self._search.setAutoresizingMask_(8)
        content.addSubview_(self._search)

        self._table = AK.NSTableView.alloc().initWithFrame_(NSMakeRect(0, 0, 318, 540))
        column = AK.NSTableColumn.alloc().initWithIdentifier_("recording")
        column.setWidth_(310)
        self._table.addTableColumn_(column)
        self._table.setHeaderView_(None)
        self._table.setRowHeight_(66)
        self._table.setIntercellSpacing_((0, 1))
        self._table.setDataSource_(self)
        self._table.setDelegate_(self)
        self._table.setAllowsEmptySelection_(True)
        self._table.setColumnAutoresizingStyle_(1)
        self._table.setStyle_(2)
        scroll = AK.NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 48, 328, 542))
        scroll.setDocumentView_(self._table)
        scroll.setHasVerticalScroller_(True)
        scroll.setAutoresizingMask_(16)
        content.addSubview_(scroll)

        divider = AK.NSBox.alloc().initWithFrame_(NSMakeRect(328, 0, 1, 650))
        divider.setBoxType_(2)
        divider.setAutoresizingMask_(16)
        content.addSubview_(divider)
        self._count = _label("Loading recordings", NSMakeRect(18, 17, 292, 20), 11, True)
        self._count.setToolTip_(f"Retention: manual\n{self._spool.config.root}")
        content.addSubview_(self._count)
        self._title = _label("Recordings", NSMakeRect(350, 603, 520, 26), 19)
        self._title.setFont_(AK.NSFont.boldSystemFontOfSize_(19))
        self._title.setAutoresizingMask_(2 | 8)
        content.addSubview_(self._title)
        self._meta = _label("", NSMakeRect(350, 575, 605, 21), 11, True)
        self._meta.setAutoresizingMask_(2 | 8)
        content.addSubview_(self._meta)

        self._attempts = AK.NSPopUpButton.alloc().initWithFrame_pullsDown_(NSMakeRect(350, 523, 300, 30), False)
        self._attempts.setTarget_(self)
        self._attempts.setAction_("selectAttempt:")
        self._attempts.setAutoresizingMask_(2 | 8)
        content.addSubview_(self._attempts)
        self._play = self._tool(content, "play.fill", "Play original audio", "playAudio:", 658, 502, 8 | 1)
        self._copy = self._tool(content, "doc.on.doc", "Copy selected transcript", "copyText:", 734, 502, 8 | 1)
        self._insert = self._tool(content, "arrow.up.doc", "Insert selected transcript", "insertText:", 810, 502, 8 | 1)
        self._trash = self._tool(content, "trash", "Move recording and attempts to Trash", "trashRecording:", 886, 502, 8 | 1)

        text_scroll = AK.NSScrollView.alloc().initWithFrame_(NSMakeRect(350, 208, 608, 294))
        text_scroll.setHasVerticalScroller_(True)
        text_scroll.setAutoresizingMask_(2 | 16)
        self._text = AK.NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, 608, 330))
        self._text.setEditable_(False)
        self._text.setSelectable_(True)
        self._text.setFont_(AK.NSFont.systemFontOfSize_(15))
        self._text.setTextContainerInset_((8, 10))
        self._text.setVerticallyResizable_(True)
        self._text.setHorizontallyResizable_(False)
        self._text.setAutoresizingMask_(2)
        self._text.textContainer().setWidthTracksTextView_(True)
        text_scroll.setDocumentView_(self._text)
        content.addSubview_(text_scroll)

        self._route = _label("", NSMakeRect(350, 146, 608, 54), 11, True)
        self._route.setMaximumNumberOfLines_(3)
        self._route.setAutoresizingMask_(2)
        content.addSubview_(self._route)
        content.addSubview_(_label("Re-transcription model", NSMakeRect(350, 113, 290, 20), 12, True))
        self._models = AK.NSPopUpButton.alloc().initWithFrame_pullsDown_(NSMakeRect(350, 75, 320, 32), False)
        self._models.setAutoresizingMask_(2)
        self._models.setTarget_(self)
        self._models.setAction_("selectModel:")
        content.addSubview_(self._models)
        self._retry = AK.NSButton.buttonWithTitle_target_action_("Re-transcribe", self, "retranscribe:")
        self._retry.setFrame_(NSMakeRect(684, 76, 142, 32))
        self._retry.setAutoresizingMask_(1)
        self._retry.setImage_(AK.NSImage.imageWithSystemSymbolName_accessibilityDescription_("arrow.clockwise", "Re-transcribe"))
        self._retry.setImagePosition_(2)
        self._retry.setToolTip_("Re-transcribe original audio with the selected recovery model")
        content.addSubview_(self._retry)
        image = AK.NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            "stop.circle", "Stop current work and re-transcribe",
        )
        self._force_retry = AK.NSButton.buttonWithTitle_target_action_("Stop & Retry", self, "forceRetranscribe:")
        self._force_retry.setImage_(image)
        self._force_retry.setImagePosition_(2)
        self._force_retry.setFrame_(NSMakeRect(838, 76, 122, 32))
        self._force_retry.setAutoresizingMask_(1)
        self._force_retry.setToolTip_("Stop current transcription work and re-transcribe this recording with the selected model")
        content.addSubview_(self._force_retry)
        self._status = _label("", NSMakeRect(350, 19, 608, 40), 11, True)
        self._status.setMaximumNumberOfLines_(2)
        self._status.setAutoresizingMask_(2)
        self._status.setStringValue_("Retention: manual" if self._spool.config.enabled else "Recording history disabled")
        content.addSubview_(self._status)
        self._window = window
        self._init_window_shell(window, "spoke.recordings")
        self._populate_models()
        self._render_detail()

    @objc.python_method
    def _tool(self, parent, symbol, tooltip, action, x, y, mask):
        image = AK.NSImage.imageWithSystemSymbolName_accessibilityDescription_(symbol, tooltip)
        if image is not None:
            configuration = AK.NSImageSymbolConfiguration.configurationWithPointSize_weight_scale_(
                44, AK.NSFontWeightMedium, AK.NSImageSymbolScaleMedium,
            )
            image = image.imageWithSymbolConfiguration_(configuration)
            image.setSize_((56, 56))
        button = AK.NSButton.buttonWithImage_target_action_(
            image, self, action,
        )
        button.setBezelStyle_(AK.NSBezelStyleCircular)
        button.setControlSize_(AK.NSControlSizeLarge)
        button.setFrame_(NSMakeRect(x, y, 72, 72))
        button.setToolTip_(tooltip)
        button.setAutoresizingMask_(mask)
        parent.addSubview_(button)
        return button

    def _populate_models(self):
        self._models.removeAllItems()
        selected = self._delegate._history_model
        self._model_routes = self._delegate._history_model_options()
        for index, item in enumerate(self._model_routes):
            self._models.addItemWithTitle_(item["label"])
            self._models.itemAtIndex_(index).setEnabled_(item["available"])
            if item["route"] == selected:
                self._models.selectItemAtIndex_(index)
        if not any(item["route"] == selected for item in self._model_routes):
            self._models.addItemWithTitle_(f"Unavailable: {selected.get('model', 'unknown')}")
            self._models.selectItemAtIndex_(len(self._model_routes))

    @objc.python_method
    def show(self, previous_app=None, previous_focus=None):
        self.setup()
        if (previous_focus is not None or previous_app is not None
                or not getattr(self, "_popup_visible", self._window.isVisible())):
            self._focus_return = previous_focus or WindowFocus.capture(previous_app)
            self._previous_app = self._focus_return.app if self._focus_return is not None else None
        self._show_window_shell()
        self._window.makeKeyAndOrderFront_(None)
        AK.NSApp().activateIgnoringOtherApps_(True)
        self.refresh_(None)
        if self._timer is None:
            self._timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                1.5, self, "refresh:", None, True,
            )

    @objc.python_method
    def dismiss(self, restore_focus=True):
        self._restore_focus = restore_focus
        self._return_focus_on_close = window_owns_key(self._window)
        if self._dismiss_window_shell(restore_focus):
            self.windowWillClose_(None)
        else:
            self._window.close()

    def windowShouldClose_(self, sender):
        self.dismiss()
        return False

    def windowWillClose_(self, notification):
        alert = getattr(self, "_trash_alert", None)
        self._trash_alert = None
        if alert is not None:
            self._window.endSheet_returnCode_(alert.window(), 1001)
        if getattr(self, "visible", False):
            self._dismiss_window_shell(False)
        was_key = getattr(self, '_return_focus_on_close', None)
        self._return_focus_on_close = None
        if self._timer is not None:
            self._timer.invalidate()
            self._timer = None
        self._stop_audio()
        on_close = getattr(self, "_on_close", None)
        if on_close is not None:
            on_close(self)
        if not getattr(self, '_cleaning_up', False) and getattr(self, '_restore_focus', True):
            restore_window_focus(self, self._window, was_key=was_key)
        self._restore_focus = True

    def cleanup(self):
        self._cleaning_up = True
        if self._window is not None:
            self.dismiss(restore_focus=False)
        self._replays.shutdown(wait=False, cancel_futures=False)

    def refresh_(self, sender):
        if self._refreshing:
            return
        self._refreshing = True
        self._load_generation = getattr(self, "_load_generation", 0) + 1
        generation = self._load_generation
        mailbox = {"lock": threading.Lock(), "payload": None, "error": None, "queued": False}

        def publish(payload):
            # Superseded prefixes must not queue archive-sized UI rebuilds.
            with mailbox["lock"]:
                mailbox["error" if "error" in payload else "payload"] = payload
                if mailbox["queued"]:
                    return
                mailbox["queued"] = True
            self.performSelectorOnMainThread_withObject_waitUntilDone_("drainLoad:", mailbox, False)

        def load():
            records = []
            try:
                last_publication = time.monotonic()
                for record in self._spool.iter_recordings():
                    records.append(record)
                    now = time.monotonic()
                    if len(records) == 1 or now - last_publication >= 0.1:
                        publish({"records": list(records), "complete": False,
                                 "load_generation": generation})
                        last_publication = now
                payload = {"records": records, "complete": True, "load_generation": generation}
            except Exception as exc:
                publish({"records": records, "complete": False, "load_generation": generation})
                payload = {"error": str(exc), "load_generation": generation}
            publish(payload)
        try:
            threading.Thread(target=load, name="spoke-history-list", daemon=True).start()
        except Exception as exc:
            self.loaded_({"error": str(exc), "load_generation": generation})

    def drainLoad_(self, mailbox):
        with mailbox["lock"]:
            payload, error = mailbox["payload"], mailbox["error"]
            mailbox.update(payload=None, error=None, queued=False)
        if payload is not None:
            self.loaded_(payload)
        if error is not None:
            self.loaded_(error)

    def loaded_(self, payload):
        generation = payload.get("load_generation")
        if generation != getattr(self, "_load_generation", 0):
            return
        if "error" not in payload and ("records" not in payload or "complete" not in payload):
            logger.warning("Ignored incomplete History scan callback")
            return
        complete = payload.get("complete", True)
        self._refreshing = not complete
        if "error" in payload:
            self._status.setStringValue_(f"History unavailable: {payload['error']}")
            self._count.setStringValue_(f"History unavailable | {len(self._records)} loaded")
            self._loading_partial = False
            self._load_failed = True
            return
        was_incomplete = (getattr(self, "_loading_partial", False)
                          or getattr(self, "_load_failed", False))
        self._loading_partial = not complete
        self._load_failed = False
        records = {r["capture_id"]: r for r in payload["records"]}
        if not complete:
            # A checked prefix is not evidence that older cached rows vanished.
            records.update((r["capture_id"], r) for r in self._records
                           if r["capture_id"] not in records)
        for capture_id, (observed_generation, record) in list(getattr(self, "_record_overrides", {}).items()):
            if generation <= observed_generation or not complete:
                records[capture_id] = record
            elif complete:
                self._record_overrides.pop(capture_id, None)
        records = [self._merge_replay_outcomes(record) for record in records.values()]
        if records != self._records:
            self._records = records
            self._filter()
        elif complete and (was_incomplete or not self._records):
            self._filter()
        if not complete:
            self._count.setStringValue_(f"Loading recordings | {len(self._records)} loaded")
        elif str(self._status.stringValue()).startswith("History unavailable:"):
            self._status.setStringValue_("Retention: manual")
        self._update_controls()

    def controlTextDidChange_(self, notification):
        self._filter()

    def _filter(self):
        query = str(self._search.stringValue()).casefold().strip()
        self._filtered = [r for r in self._records if not query or query in (
            r["capture_id"] + " " + " ".join(str(a.get("text") or "") for a in r["attempts"])
        ).casefold()]
        index = next((i for i, r in enumerate(self._filtered) if r["capture_id"] == self._selected_id), 0)
        selected = self._filtered[index]["capture_id"] if self._filtered else None
        if selected != self._selected_id:
            self._attempt_id = None
            if getattr(self, "_sound", None) is not None:
                self._stop_audio()
        self._selected_id = selected
        self._reloading_table = True
        try:
            self._table.reloadData()
            if self._filtered:
                self._table.selectRowIndexes_byExtendingSelection_(NSIndexSet.indexSetWithIndex_(index), False)
        finally:
            self._reloading_table = False
        size = sum(r.get("byte_count", 0) or 0 for r in self._records)
        if getattr(self, "_loading_partial", False):
            count = f"Loading recordings | {len(self._records)} loaded"
        elif getattr(self, "_load_failed", False):
            count = f"History unavailable | {len(self._records)} loaded"
        else:
            count = f"{len(self._filtered)} recordings  |  {size / 1048576:.1f} MB retained"
        self._count.setStringValue_(count)
        self._render_detail()

    def numberOfRowsInTableView_(self, table):
        return len(self._filtered)

    def tableView_viewForTableColumn_row_(self, table, column, row):
        record = self._filtered[row]
        width = float(column.width())
        view = AK.NSTableCellView.alloc().initWithFrame_(NSMakeRect(0, 0, width, 66))
        view.addSubview_(_label(_date_label(record.get("created_at")), NSMakeRect(8, 42, width - 16, 18), 12, True))
        view.addSubview_(_label(first_line(record), NSMakeRect(8, 21, width - 16, 20), 13))
        status = "Audio missing" if not record["audio_available"] else f"Transcription: {record['status'].replace('_', ' ')}"
        view.addSubview_(_label(f"{duration_label(record.get('duration_seconds'))}  |  {status}", NSMakeRect(8, 3, width - 16, 17), 10, True))
        return view

    def tableViewSelectionDidChange_(self, notification):
        if getattr(self, "_reloading_table", False):
            return
        index = int(self._table.selectedRow())
        if 0 <= index < len(self._filtered):
            selected = self._filtered[index]["capture_id"]
            if selected != self._selected_id:
                self._attempt_id = None
                self._stop_audio()
            self._selected_id = selected
        self._render_detail()

    def _selected(self):
        return next((r for r in self._filtered if r["capture_id"] == self._selected_id), None)

    def _attempt(self):
        return next((a for a in self._selected_attempts() if a["attempt_id"] == self._attempt_id), None)

    def _merge_replay_outcomes(self, record):
        outcomes = getattr(self, "_replay_outcomes", {}).get(record["capture_id"], {})
        if not outcomes:
            return record
        attempts = list(record["attempts"])
        for attempt_id, outcome in outcomes.items():
            if not outcome["metadata_pending"]:
                continue
            index = next((i for i, attempt in enumerate(attempts) if attempt["attempt_id"] == attempt_id), None)
            if index is not None and attempts[index].get("status") in self._replay_terminal_labels:
                outcome.update(metadata_pending=False, message=self._replay_terminal_labels[attempts[index]["status"]])
            elif index is None:
                attempts.append(outcome["unverified_attempt"])
            else:
                attempts[index] = outcome["unverified_attempt"]
        return {**record, "attempts": attempts,
                "status": "unverified" if attempts and attempts[-1].get("status") == "unverified"
                else record.get("status", "not_transcribed")}

    def _selected_attempts(self):
        record = self._selected()
        if record is None:
            return []
        attempts = list(self._merge_replay_outcomes(record)["attempts"])
        job = self._busy.get(record["capture_id"])
        if job and not any(a["attempt_id"] == job["attempt_id"] for a in attempts):
            attempts.append({"attempt_id": job["attempt_id"], "kind": "retranscription",
                             "requested": job["route"], "status": "pending"})
        return attempts

    def _update_replay_status(self):
        job = self._busy.get(self._selected_id)
        outcomes = getattr(self, "_replay_outcomes", {})
        outcome = outcomes.get(self._selected_id, {}).get(self._attempt_id)
        if job:
            self._status.setStringValue_(self._replay_phase_label(job["phase"]))
        elif outcome and outcome["attempt_id"] == self._attempt_id:
            self._status.setStringValue_(outcome["message"])
        elif self._busy:
            self._status.setStringValue_("Re-transcribing another recording")
        elif str(self._status.stringValue()) in {
            "Re-transcription queued", "Re-transcribing another recording",
            *self._replay_phase_labels.values(),
            *(outcome["message"] for capture in outcomes.values() for outcome in capture.values()),
        }:
            self._status.setStringValue_("Retention: manual")

    _replay_phase_labels = {
        "queued": "Waiting for the re-transcription worker",
        "validating": "Checking original audio",
        "preparing": "Preparing re-transcription backend",
        "waiting": "Waiting for inference",
        "transcribing": "Re-transcribing...",
        "saving": "Saving re-transcription",
    }
    _replay_terminal_labels = {
        "success": "Re-transcription finished; original retained",
        "failed": "Re-transcription failed; original retained",
        "cancelled": "Re-transcription stopped; original retained",
        "blank": "Re-transcription returned no text; original retained",
    }

    def _replay_phase_label(self, phase):
        return self._replay_phase_labels.get(phase, "Re-transcription state unavailable")

    def _render_detail(self):
        record = self._selected()
        attempts = self._selected_attempts() if record else []
        if self._attempt_id not in {a["attempt_id"] for a in attempts}:
            self._attempt_id = attempts[-1]["attempt_id"] if attempts else None
        job = self._busy.get(self._selected_id)
        snapshot = (record, attempts, self._attempt_id, job.get("phase") if job else None,
                    getattr(self, "_loading_partial", False) if record is None else None,
                    getattr(self, "_load_failed", False) if record is None else None,
                    bool(self._records) if record is None else None)
        if snapshot == getattr(self, "_detail_snapshot", None):
            self._update_replay_status()
            self._update_controls()
            return
        self._detail_snapshot = deepcopy(snapshot)
        self._attempts.removeAllItems()
        if record is None:
            self._update_replay_status()
            self._title.setStringValue_("Recordings")
            self._meta.setStringValue_("")
            if getattr(self, "_loading_partial", False):
                message = "No matching recordings yet"
            elif getattr(self, "_load_failed", False):
                message = "History unavailable"
            else:
                message = "No recordings" if not self._records else "No matching recordings"
            self._text.setString_(message)
            self._route.setStringValue_("")
            self._update_controls()
            return
        self._title.setStringValue_(_date_label(record.get("created_at")))
        pathway = record.get("pathway", "Unverified pathway")
        self._meta.setStringValue_(f"{duration_label(record.get('duration_seconds'))}  |  Pathway: {pathway}")
        self._meta.setToolTip_(record["capture_id"])
        for index, attempt in enumerate(attempts):
            kind = "Original" if attempt.get("kind") == "live" else "Re-transcription"
            label = f"{kind} {index + 1}  |  {attempt.get('requested', {}).get('model', 'Unknown model')}"
            self._attempts.addItemWithTitle_(label)
            if attempt["attempt_id"] == self._attempt_id:
                self._attempts.selectItemAtIndex_(index)
        self._render_attempt()

    def selectAttempt_(self, sender):
        record = self._selected()
        index = int(sender.indexOfSelectedItem())
        attempts = self._selected_attempts()
        if record and 0 <= index < len(attempts):
            self._attempt_id = attempts[index]["attempt_id"]
            self._render_attempt()

    def _render_attempt(self):
        record, attempt = self._selected(), self._attempt()
        self._update_replay_status()
        if attempt:
            message = attempt.get("text") or attempt.get("error") or {
                "pending": "No completed result yet.", "blank": "The model returned an empty transcript.",
            }.get(attempt.get("status"), "No transcript available.")
            job = self._busy.get(record["capture_id"])
            if attempt.get("status") == "pending" and job and job["attempt_id"] == attempt["attempt_id"]:
                message = self._replay_phase_label(job["phase"])
            self._text.setString_(message)
            effective = attempt.get("effective", {})
            model = effective.get("model") or effective.get("model_id") or effective.get("model_path") or "Unverified model"
            duration = attempt.get("wall_seconds")
            timing = f"{duration:.2f}s" if isinstance(duration, (int, float)) else "Unfinished"
            client = effective.get("client", "No effective route yet").split(".")[-1]
            self._route.setStringValue_(f"{client}  |  {model}\nTranscription: {attempt['status']}  |  {timing}\n{delivery_label(attempt)}")
            self._route.setToolTip_(json.dumps(attempt, indent=2, sort_keys=True))
        else:
            self._text.setString_(record.get("error") or "No original transcript was retained." if record else "")
            self._route.setStringValue_(record.get("audio_error", "Original audio preserved") if record else "")
        self._update_controls()

    def _update_controls(self):
        record, attempt = self._selected(), self._attempt()
        has_text = bool(attempt and isinstance(attempt.get("text"), str) and attempt["text"].strip())
        self._copy.setEnabled_(has_text)
        live_busy = self._delegate._history_live_busy()
        self._insert.setEnabled_(has_text and not live_busy)
        self._play.setEnabled_(bool(record and record["audio_available"]))
        self._trash.setEnabled_(bool(record and record["capture_id"] not in self._busy and not live_busy))
        route = self._delegate._history_model
        available = any(item["route"] == route and item["available"] for item in self._model_routes)
        self._retry.setEnabled_(bool(record and record["audio_available"] and available and not live_busy
                                    and record["capture_id"] not in self._busy and not record.get("error")))
        can_take_over = getattr(self._delegate, "_history_can_take_over", lambda: False)()
        self._force_retry.setEnabled_(bool(record and record["audio_available"] and available
                                          and can_take_over and (live_busy or self._busy)
                                          and not record.get("error")))

    def selectModel_(self, sender):
        index = int(sender.indexOfSelectedItem())
        if 0 <= index < len(self._model_routes):
            route = self._model_routes[index]["route"]
            self._delegate._set_history_model(route["backend"], route["model"])
        self._update_controls()

    def retranscribe_(self, sender):
        self._start_retranscription(origin="retry_action")

    def forceRetranscribe_(self, sender):
        record = self._selected()
        route = self._delegate._history_model
        available = any(item["route"] == route and item["available"] for item in self._model_routes)
        if not record or not record["audio_available"] or record.get("error") or not available:
            return
        if not self._delegate._cancel_transcription_for_recovery():
            self._status.setStringValue_("Current work cannot be stopped here; active recording and paste are protected")
            return
        self._replay_generation += 1
        self._busy.clear()
        self._start_retranscription(origin="stop_and_retry_action")

    @objc.python_method
    def _start_retranscription(self, *, origin="direct_call"):
        record = self._selected()
        if not record or record["capture_id"] in self._busy or self._delegate._history_live_busy():
            return
        route = dict(self._delegate._history_model)
        capture_id = record["capture_id"]
        try:
            attempt_id = self._spool.start_attempt(capture_id, requested=route, kind="retranscription")
        except Exception as exc:
            self._status.setStringValue_(str(exc))
            return
        self._attempt_id = attempt_id
        generation = self._replay_generation
        control = getattr(self._delegate, "_inference_control", None)
        epoch = control.epoch if control is not None else None
        self._busy[capture_id] = {"attempt_id": attempt_id, "route": route, "phase": "queued",
                                  "generation": generation}
        try:
            event = AK.NSApp().currentEvent()
            event_type = int(event.type()) if event is not None else None
        except Exception:
            event_type = None
        logger.info("History replay requested: capture=%s attempt=%s generation=%s origin=%s event_type=%s model=%s",
                    capture_id, attempt_id, generation, origin, event_type, route.get("model"))
        self._render_detail()
        submission_resolved = threading.Event()
        accepted = False
        def replay():
            submission_resolved.wait()
            if not accepted:
                return
            try:
                def progress(phase):
                    self.performSelectorOnMainThread_withObject_waitUntilDone_("replayPhase:", {
                        "capture_id": capture_id, "attempt_id": attempt_id,
                        "generation": generation, "phase": phase}, False)
                run_retranscription(self._spool, capture_id, attempt_id, route,
                                    self._delegate._build_history_client,
                                    lambda client: control.context(self._delegate._local_inference_lock, client, epoch=epoch)
                                    if control is not None else self._delegate._local_inference_context(client),
                                    cancel_check=lambda: generation != self._replay_generation,
                                    on_phase=progress)
                payload = {"capture_id": capture_id, "attempt_id": attempt_id, "generation": generation,
                           "record": self._spool.read_recording_metadata(capture_id)}
            except Exception as exc:
                logger.exception("Could not finish History replay/update")
                payload = {"capture_id": capture_id, "attempt_id": attempt_id,
                           "generation": generation, "error": str(exc)}
            self.performSelectorOnMainThread_withObject_waitUntilDone_("replayFinished:", payload, False)
        try:
            self._replays.submit(replay)
            accepted = True
        except Exception as exc:
            # CPython can enqueue before worker startup raises. That exact
            # rejected closure must remain inert if a later submit starts it.
            payload = {"capture_id": capture_id, "attempt_id": attempt_id, "generation": generation,
                       "error": f"Could not queue re-transcription: {exc}"}
            try:
                self._spool.finish_attempt(capture_id, attempt_id, text=None, effective={}, wall_seconds=0,
                    error=f"Could not queue replay: {exc}", evidence={"phase": "queue_submission"})
            except Exception:
                logger.exception("Could not persist History queue failure")
            try:
                payload["record"] = self._spool.read_recording_metadata(capture_id)
            except Exception:
                logger.exception("Could not read History queue failure")
            self.replayFinished_(payload)
            return
        finally:
            submission_resolved.set()
        self.refresh_(None)
        self._update_controls()

    def replayPhase_(self, payload):
        job = self._busy.get(payload.get("capture_id"))
        if (payload.get("generation") != self._replay_generation or job is None
                or payload.get("attempt_id") != job["attempt_id"]
                or payload.get("phase") not in self._replay_phase_labels):
            return
        job["phase"] = payload["phase"]
        self._update_replay_status()
        if self._selected_id == payload["capture_id"] and self._attempt_id == job["attempt_id"]:
            self._render_attempt()

    def replayFinished_(self, payload):
        if payload.get("generation") != self._replay_generation:
            return
        job = self._busy.get(payload.get("capture_id"))
        if job is None or payload.get("attempt_id") != job["attempt_id"]:
            return
        self._busy.pop(payload["capture_id"], None)
        record = payload.get("record")
        valid_record = isinstance(record, dict) and record.get("capture_id") == payload["capture_id"]
        terminal = next((a for a in record.get("attempts", []) if a["attempt_id"] == payload["attempt_id"]), {}) if valid_record else {}
        metadata_pending = terminal.get("status") not in self._replay_terminal_labels
        if metadata_pending:
            prior = record if valid_record else next((r for r in self._records if r["capture_id"] == payload["capture_id"]), {})
            record = {**prior, "capture_id": payload["capture_id"], "status": "unverified",
                      "audio_available": prior.get("audio_available", False),
                      "attempts": [a for a in prior.get("attempts", []) if a["attempt_id"] != payload["attempt_id"]] + [{
                          "attempt_id": payload["attempt_id"], "kind": "retranscription", "status": "unverified",
                          "requested": job["route"], "error": payload.get("error") or "Re-transcription result unavailable"}]}
        attempt = next((a for a in record.get("attempts", [])
                        if a["attempt_id"] == payload["attempt_id"]), {})
        message = self._replay_terminal_labels.get(attempt.get("status"), "Re-transcription result unavailable")
        if not hasattr(self, "_replay_outcomes"):
            self._replay_outcomes = {}
        self._replay_outcomes.setdefault(payload["capture_id"], {})[payload["attempt_id"]] = {
            "attempt_id": payload["attempt_id"], "metadata_pending": metadata_pending,
            "unverified_attempt": attempt if metadata_pending else None,
            "message": payload.get("error") or message}
        record = self._merge_replay_outcomes(record)
        overrides = getattr(self, "_record_overrides", None)
        if overrides is None:
            overrides = self._record_overrides = {}
        overrides[payload["capture_id"]] = (getattr(self, "_load_generation", 0), record)
        records = {r["capture_id"]: r for r in self._records}
        records[payload["capture_id"]] = record
        self._records = list(records.values())
        self._filter()
        self.refresh_(None)

    def copyText_(self, sender):
        attempt = self._attempt()
        if not attempt or not attempt.get("text"):
            return
        from .inject import set_pasteboard_only
        try:
            set_pasteboard_only(attempt["text"])
        except Exception as exc:
            logger.exception("Could not copy selected transcript")
            self._status.setStringValue_(f"Copy failed: {exc}")
            return
        self._spool.record_delivery(self._selected_id, self._attempt_id, state="copied")
        self._status.setStringValue_("Copied")

    def insertText_(self, sender):
        attempt = self._attempt()
        if not attempt or not attempt.get("text") or self._delegate._history_live_busy():
            return
        self.dismiss()
        self._delegate._inject_result_text(
            attempt["text"], "History insertion requested",
            history={"capture_id": self._selected_id, "attempt_id": self._attempt_id},
        )

    def _stop_audio(self):
        if self._sound is not None:
            self._sound.stop()
            self._sound = None

    def playAudio_(self, sender):
        if self._sound is not None and self._sound.isPlaying():
            self._stop_audio()
            return
        record = self._selected()
        if not record:
            return
        try:
            audio = self._spool.read_recording_audio(record["capture_id"])
            data = NSData.dataWithBytes_length_(audio, len(audio))
            self._sound = AK.NSSound.alloc().initWithData_(data)
            if self._sound is None or not self._sound.play():
                raise RuntimeError("Audio playback failed")
        except Exception as exc:
            self._status.setStringValue_(str(exc))

    def trashRecording_(self, sender):
        record = self._selected()
        if (not record or record["capture_id"] in self._busy or self._delegate._history_live_busy()
                or getattr(self, "_trash_alert", None) is not None):
            return
        alert = AK.NSAlert.alloc().init()
        alert.setMessageText_("Move this recording to Trash?")
        alert.setInformativeText_("The original audio and all transcription attempts will move together.")
        alert.addButtonWithTitle_("Move to Trash")
        alert.addButtonWithTitle_("Cancel")
        self._trash_alert = alert

        def confirmed(response):
            if self._trash_alert is not alert:
                return
            self._trash_alert = None
            if (response == 1000 and record["capture_id"] not in self._busy
                    and not self._delegate._history_live_busy()):
                self._trash_recording(record["capture_id"])

        alert.beginSheetModalForWindow_completionHandler_(self._window, confirmed)

    def _trash_recording(self, capture_id):
        self._stop_audio()
        base = self._spool._capture_path(capture_id)
        paths = [base.with_suffix(".wav"), base.with_suffix(".json"),
                 self._spool.config.root / "attempts" / capture_id]
        urls = [NSURL.fileURLWithPath_(str(path)) for path in paths if path.exists()]
        def complete(mapping, error):
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "trashFinished:", {"error": str(error) if error else None}, False,
            )
        AK.NSWorkspace.sharedWorkspace().recycleURLs_completionHandler_(urls, complete)

    def trashFinished_(self, payload):
        self._status.setStringValue_(f"Trash operation incomplete: {payload['error']}" if payload.get("error") else "Moved to Trash")
        self.refresh_(None)

"""Native recording library. Metadata refresh and replay never own live capture."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import logging
import threading

import objc
import AppKit as AK
from Foundation import NSData, NSIndexSet, NSMakeRect, NSObject, NSTimer, NSURL

from .recording_history import delivery_label, duration_label, first_line, run_retranscription

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


class RecordingHistoryWindow(NSObject):
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
        self._timer = None
        self._sound = None
        self._previous_app = None
        self._replays = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spoke-history")
        self._busy = set()
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

        self._attempts = AK.NSPopUpButton.alloc().initWithFrame_pullsDown_(NSMakeRect(350, 533, 348, 30), False)
        self._attempts.setTarget_(self)
        self._attempts.setAction_("selectAttempt:")
        self._attempts.setAutoresizingMask_(2 | 8)
        content.addSubview_(self._attempts)
        self._play = self._tool(content, "play.fill", "Play original audio", "playAudio:", 706, 514, 8 | 1)
        self._copy = self._tool(content, "doc.on.doc", "Copy selected transcript", "copyText:", 770, 514, 8 | 1)
        self._insert = self._tool(content, "arrow.up.doc", "Insert selected transcript", "insertText:", 834, 514, 8 | 1)
        self._trash = self._tool(content, "trash", "Move recording and attempts to Trash", "trashRecording:", 898, 514, 8 | 1)

        text_scroll = AK.NSScrollView.alloc().initWithFrame_(NSMakeRect(350, 208, 608, 306))
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
        self._models = AK.NSPopUpButton.alloc().initWithFrame_pullsDown_(NSMakeRect(350, 75, 454, 32), False)
        self._models.setAutoresizingMask_(2)
        self._models.setTarget_(self)
        self._models.setAction_("selectModel:")
        content.addSubview_(self._models)
        self._retry = AK.NSButton.buttonWithTitle_target_action_("Re-transcribe", self, "retranscribe:")
        self._retry.setFrame_(NSMakeRect(818, 76, 142, 32))
        self._retry.setAutoresizingMask_(1)
        self._retry.setImage_(AK.NSImage.imageWithSystemSymbolName_accessibilityDescription_("arrow.clockwise", "Re-transcribe"))
        self._retry.setImagePosition_(2)
        self._retry.setToolTip_("Re-transcribe original audio with the selected recovery model")
        content.addSubview_(self._retry)
        self._status = _label("", NSMakeRect(350, 19, 608, 40), 11, True)
        self._status.setMaximumNumberOfLines_(2)
        self._status.setAutoresizingMask_(2)
        self._status.setStringValue_("Retention: manual" if self._spool.config.enabled else "Recording history disabled")
        content.addSubview_(self._status)
        self._window = window
        self._populate_models()
        self._render_detail()

    @objc.python_method
    def _tool(self, parent, symbol, tooltip, action, x, y, mask):
        image = AK.NSImage.imageWithSystemSymbolName_accessibilityDescription_(symbol, tooltip)
        if image is not None:
            configuration = AK.NSImageSymbolConfiguration.configurationWithPointSize_weight_scale_(
                34, AK.NSFontWeightMedium, AK.NSImageSymbolScaleMedium,
            )
            image = image.imageWithSymbolConfiguration_(configuration)
            image.setSize_((42, 42))
        button = AK.NSButton.buttonWithImage_target_action_(
            image, self, action,
        )
        button.setFrame_(NSMakeRect(x, y, 60, 60))
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

    def show(self):
        self.setup()
        if not self._window.isVisible():
            self._previous_app = AK.NSWorkspace.sharedWorkspace().frontmostApplication()
        self._window.makeKeyAndOrderFront_(None)
        AK.NSApp().activateIgnoringOtherApps_(True)
        self.refresh_(None)
        if self._timer is None:
            self._timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                1.5, self, "refresh:", None, True,
            )

    def windowWillClose_(self, notification):
        if self._timer is not None:
            self._timer.invalidate()
            self._timer = None
        self._stop_audio()

    def cleanup(self):
        if self._window is not None:
            self._window.close()
        self._replays.shutdown(wait=False, cancel_futures=False)

    def refresh_(self, sender):
        if self._refreshing:
            return
        self._refreshing = True
        def load():
            try:
                payload = {"records": self._spool.list_recordings()}
            except Exception as exc:
                payload = {"error": str(exc)}
            self.performSelectorOnMainThread_withObject_waitUntilDone_("loaded:", payload, False)
        threading.Thread(target=load, name="spoke-history-list", daemon=True).start()

    def loaded_(self, payload):
        self._refreshing = False
        if payload.get("error"):
            self._status.setStringValue_(f"History unavailable: {payload['error']}")
            return
        if payload["records"] != self._records:
            self._records = payload["records"]
            self._filter()
        self._update_controls()

    def controlTextDidChange_(self, notification):
        self._filter()

    def _filter(self):
        query = str(self._search.stringValue()).casefold().strip()
        self._filtered = [r for r in self._records if not query or query in (
            r["capture_id"] + " " + " ".join(str(a.get("text") or "") for a in r["attempts"])
        ).casefold()]
        self._table.reloadData()
        index = next((i for i, r in enumerate(self._filtered) if r["capture_id"] == self._selected_id), 0)
        if self._filtered:
            self._table.selectRowIndexes_byExtendingSelection_(NSIndexSet.indexSetWithIndex_(index), False)
            self._selected_id = self._filtered[index]["capture_id"]
        else:
            self._selected_id = None
        size = sum(r.get("byte_count", 0) or 0 for r in self._records)
        self._count.setStringValue_(f"{len(self._filtered)} recordings  |  {size / 1048576:.1f} MB retained")
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
        record = self._selected()
        return next((a for a in record["attempts"] if a["attempt_id"] == self._attempt_id), None) if record else None

    def _render_detail(self):
        record = self._selected()
        self._attempts.removeAllItems()
        if record is None:
            self._title.setStringValue_("Recordings")
            self._meta.setStringValue_("")
            self._text.setString_("No recordings" if not self._records else "No matching recordings")
            self._route.setStringValue_("")
            self._update_controls()
            return
        self._title.setStringValue_(_date_label(record.get("created_at")))
        pathway = record.get("pathway", "Unverified pathway")
        self._meta.setStringValue_(f"{duration_label(record.get('duration_seconds'))}  |  Pathway: {pathway}")
        self._meta.setToolTip_(record["capture_id"])
        attempts = record["attempts"]
        if self._attempt_id not in {a["attempt_id"] for a in attempts}:
            self._attempt_id = attempts[-1]["attempt_id"] if attempts else None
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
        if record and 0 <= index < len(record["attempts"]):
            self._attempt_id = record["attempts"][index]["attempt_id"]
            self._render_attempt()

    def _render_attempt(self):
        record, attempt = self._selected(), self._attempt()
        if attempt:
            message = attempt.get("text") or attempt.get("error") or {
                "pending": "No completed result yet.", "blank": "The model returned an empty transcript.",
            }.get(attempt.get("status"), "No transcript available.")
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

    def selectModel_(self, sender):
        index = int(sender.indexOfSelectedItem())
        if 0 <= index < len(self._model_routes):
            route = self._model_routes[index]["route"]
            self._delegate._set_history_model(route["backend"], route["model"])
        self._update_controls()

    def retranscribe_(self, sender):
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
        self._busy.add(capture_id)
        self._status.setStringValue_("Re-transcription queued")
        def replay():
            try:
                run_retranscription(self._spool, capture_id, attempt_id, route,
                                    self._delegate._build_history_client, self._delegate._local_inference_context)
                payload = {"capture_id": capture_id}
            except Exception as exc:
                logger.exception("Could not persist re-transcription result")
                payload = {"capture_id": capture_id, "error": str(exc)}
            self.performSelectorOnMainThread_withObject_waitUntilDone_("replayFinished:", payload, False)
        self._replays.submit(replay)
        self.refresh_(None)
        self._update_controls()

    def replayFinished_(self, payload):
        self._busy.discard(payload["capture_id"])
        self._status.setStringValue_(payload.get("error") or "Re-transcription finished; original retained")
        self.refresh_(None)

    def copyText_(self, sender):
        attempt = self._attempt()
        if not attempt or not attempt.get("text"):
            return
        from .inject import set_pasteboard_only
        set_pasteboard_only(attempt["text"])
        self._spool.record_delivery(self._selected_id, self._attempt_id, state="copied")
        self._status.setStringValue_("Copied")

    def insertText_(self, sender):
        attempt = self._attempt()
        if not attempt or not attempt.get("text") or self._delegate._history_live_busy():
            return
        self.windowWillClose_(None)
        self._window.orderOut_(None)
        if self._previous_app is not None:
            self._previous_app.activateWithOptions_(1 << 1)
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
        if not record or record["capture_id"] in self._busy or self._delegate._history_live_busy():
            return
        alert = AK.NSAlert.alloc().init()
        alert.setMessageText_("Move this recording to Trash?")
        alert.setInformativeText_("The original audio and all transcription attempts will move together.")
        alert.addButtonWithTitle_("Move to Trash")
        alert.addButtonWithTitle_("Cancel")
        if alert.runModal() != 1000:
            return
        self._stop_audio()
        base = self._spool._capture_path(record["capture_id"])
        paths = [base.with_suffix(".wav"), base.with_suffix(".json"),
                 self._spool.config.root / "attempts" / record["capture_id"]]
        urls = [NSURL.fileURLWithPath_(str(path)) for path in paths if path.exists()]
        def complete(mapping, error):
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "trashFinished:", {"error": str(error) if error else None}, False,
            )
        AK.NSWorkspace.sharedWorkspace().recycleURLs_completionHandler_(urls, complete)

    def trashFinished_(self, payload):
        self._status.setStringValue_(f"Trash operation incomplete: {payload['error']}" if payload.get("error") else "Moved to Trash")
        self.refresh_(None)

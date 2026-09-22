"""Native, non-modal inbox for explicitly requested interactive smokes."""

from __future__ import annotations

from datetime import datetime
import logging
import threading

import objc
from AppKit import (
    NSApplication, NSApplicationActivateIgnoringOtherApps, NSButton, NSColor,
    NSFont, NSImage, NSImageLeading, NSMakeRect, NSPanel, NSScrollView, NSTableColumn, NSTableView,
    NSTextField, NSTextView, NSWindowStyleMaskClosable, NSWindowStyleMaskTitled,
    NSWorkspace,
)
from Foundation import NSObject, NSURL

from .diaulos_switcher import EpistaxisDiaulosClient
from .smoke_requests import DirectoryWatch, SmokeRequests, default_queue, notify_requests, return_response

logger = logging.getLogger(__name__)


def _return_affordance(row):
    if not row or not row.get("response"):
        return "Send Reply", bool(row and row.get("status") == "pending"), ""
    state = (row.get("delivery") or {}).get("state")
    if state == "delivered":
        return "Send Reply", False, "Response handed to agent. Interpretation is not yet confirmed."
    if state == "sending":
        return "Retry Return", True, "Response saved; return in progress or interrupted. Retry is idempotent."
    if state == "unconfirmed":
        return "Retry Return", True, "Response saved; return unconfirmed. You can retry the same return."
    return "Return Saved Reply", True, "Response saved; not yet returned. You can return the same response."


def _request_list_text(row):
    status = {"pending": "Waiting", "responded": "Replied", "withdrawn": "Withdrawn"}[row["status"]]
    return f"{row['request']['source']['diaulos']}\n{status}"


def _label(parent, text, frame, size=13):
    label = NSTextField.labelWithString_(text)
    label.setFrame_(NSMakeRect(*frame))
    label.setFont_(NSFont.systemFontOfSize_(size))
    parent.addSubview_(label)
    return label


def _text(parent, frame, *, editable):
    scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(*frame))
    scroll.setHasVerticalScroller_(True)
    scroll.setBorderType_(0)
    view = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, frame[2] - 16, frame[3]))
    view.setEditable_(editable)
    view.setRichText_(False)
    view.setFont_(NSFont.systemFontOfSize_(14))
    view.setTextColor_(NSColor.labelColor())
    view.setBackgroundColor_(NSColor.textBackgroundColor() if editable else NSColor.windowBackgroundColor())
    view.setVerticallyResizable_(True)
    view.setHorizontallyResizable_(False)
    view.textContainer().setWidthTracksTextView_(True)
    view.textContainer().setContainerSize_((frame[2] - 16, float("inf")))
    scroll.setDocumentView_(view)
    parent.addSubview_(scroll)
    return view


class SmokeInbox(NSObject):
    def initWithMenuBar_queue_(self, menubar, directory):
        self = objc.super(SmokeInbox, self).init()
        if self is None:
            return None
        self._menu = menubar
        self._queue = SmokeRequests(directory or default_queue())
        self._rows = []
        self._drafts = {}
        self._selected_id = None
        self._last_snapshot = None
        self._closed = False
        self._delivery_active = set()
        self._panel = None
        self._errors = []
        self._watch = None
        if menubar is not None:
            menubar._on_smoke_inbox = self.show
            menubar.refresh_menu()
        self._watch = DirectoryWatch(self._queue.directory, self._scan, self._watch_failed)
        return self

    @objc.python_method
    def _scan(self):
        try:
            claims = self._queue.claim_notifications()
            current = []
            for row in claims:
                try:
                    if self._queue.get(row["request"]["id"])["status"] == "pending":
                        current.append(row)
                except (OSError, ValueError, TypeError, AttributeError, KeyError):
                    continue
            if current:
                try:
                    state, error = notify_requests(current), ""
                except Exception as exception:
                    state, error = "unavailable", str(exception)
                for row in current:
                    try:
                        self._queue.notification_result(row["request"]["id"], state, error)
                    except (OSError, ValueError, TypeError, AttributeError, KeyError):
                        continue
            rows, errors = self._queue.scan()
        except Exception as error:
            self._watch_failed(error)
            return
        if self._closed:
            return
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "snapshotChanged:", {"rows": rows, "errors": errors}, False)

    @objc.python_method
    def _watch_failed(self, error):
        logger.error("Interactive smoke inbox: %s", error)
        if not self._closed:
            self.performSelectorOnMainThread_withObject_waitUntilDone_("inboxError:", str(error), False)

    def inboxError_(self, error):
        self._errors = [str(error)]
        if self._panel is not None:
            self._status.setStringValue_(str(error))

    def snapshotChanged_(self, payload):
        if self._closed or payload == self._last_snapshot:
            return
        self._last_snapshot = payload
        self._rows, self._errors = payload["rows"], payload["errors"]
        if self._menu is not None:
            self._menu.set_smoke_pending_count(sum(row["status"] == "pending" for row in self._rows))
        if self._panel is not None:
            self._table.reloadData()
            if self._rows:
                index = next((i for i, row in enumerate(self._rows)
                              if row["request"]["id"] == self._selected_id), 0)
                from Foundation import NSIndexSet
                self._table.selectRowIndexes_byExtendingSelection_(NSIndexSet.indexSetWithIndex_(index), False)
            self._render()

    @objc.python_method
    def _build(self):
        self._panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 790, 590), NSWindowStyleMaskTitled | NSWindowStyleMaskClosable, 2, False)
        self._panel.setTitle_("Spoke - Needs You")
        self._panel.setReleasedWhenClosed_(False)
        self._panel.center()
        view = self._panel.contentView()
        _label(view, "Needs You", (20, 549, 730, 26), 19)
        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(16, 46, 238, 490))
        scroll.setHasVerticalScroller_(True)
        self._table = NSTableView.alloc().initWithFrame_(NSMakeRect(0, 0, 222, 490))
        column = NSTableColumn.alloc().initWithIdentifier_("request")
        column.setWidth_(222)
        self._table.addTableColumn_(column)
        self._table.setHeaderView_(None)
        self._table.setRowHeight_(48)
        self._table.setDataSource_(self)
        self._table.setDelegate_(self)
        scroll.setDocumentView_(self._table)
        view.addSubview_(scroll)
        self._details = _text(view, (272, 275, 500, 259), editable=False)
        _label(view, "Your response", (277, 236, 490, 23))
        self._reply = _text(view, (272, 98, 500, 136), editable=True)
        self._buttons = {}
        for title, selector, symbol, x, width in [
            ("Open Smoke", "openSmoke:", "arrow.up.forward.app", 276, 132),
            ("Go to Agent", "goToAgent:", "terminal", 410, 134),
            ("Later", "later:", "clock", 546, 90),
            ("Send Reply", "sendReply:", "paperplane", 638, 132),
        ]:
            button = NSButton.buttonWithTitle_target_action_(title, self, selector)
            button.setFrame_(NSMakeRect(x, 53, width, 32))
            button.setImage_(NSImage.imageWithSystemSymbolName_accessibilityDescription_(symbol, title))
            button.setImagePosition_(NSImageLeading)
            button.setToolTip_(title)
            view.addSubview_(button)
            self._buttons[title] = button
        self._status = _label(view, "", (20, 12, 750, 26), 12)
        self._status.setTextColor_(NSColor.secondaryLabelColor())
        self._table.reloadData()
        if self._rows:
            from Foundation import NSIndexSet
            self._table.selectRowIndexes_byExtendingSelection_(NSIndexSet.indexSetWithIndex_(0), False)
        self._render()

    @objc.python_method
    def show(self):
        if self._panel is None:
            self._build()
        self._panel.makeKeyAndOrderFront_(None)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

    def numberOfRowsInTableView_(self, table):
        return len(self._rows)

    def tableView_objectValueForTableColumn_row_(self, table, column, index):
        return _request_list_text(self._rows[index])

    def tableViewSelectionDidChange_(self, notification):
        self._render()

    @objc.python_method
    def _selected(self):
        index = self._table.selectedRow()
        return self._rows[index] if 0 <= index < len(self._rows) else None

    @objc.python_method
    def _render(self):
        row = self._selected()
        if self._selected_id:
            self._drafts[self._selected_id] = str(self._reply.string())
        identity = row["request"]["id"] if row else None
        changed = identity != self._selected_id
        self._selected_id = identity
        if row is None:
            self._details.setString_("No interactive smokes pending.")
            self._reply.setString_("")
        else:
            request = row["request"]
            submitted = datetime.fromisoformat(row["created_at"]).astimezone().strftime("%b %d, %I:%M %p")
            availability = {"prepared": "Prepared (agent-reported)",
                            "preparation-needed": "Preparation still needed",
                            "unavailable": "Currently unavailable"}[request["availability"]]
            text = (f"{request['title']}\n\n{request['source']['diaulos']}\nRequested {submitted}\n\n"
                    f"{request['prompt']}\n\n{availability}\n{request['availability_note']}\n\n{request['url']}")
            if row["status"] == "withdrawn":
                text += f"\n\nWithdrawn: {row['withdrawal_reason']}"
            if row["response"]:
                self._reply.setString_(row["response"]["text"])
            elif changed:
                self._reply.setString_(self._drafts.get(identity, ""))
            self._details.setString_(text)
        pending = bool(row and row["status"] == "pending")
        for name in ("Open Smoke", "Go to Agent", "Later"):
            self._buttons[name].setEnabled_(pending)
        self._reply.setEditable_(pending)
        return_title, return_enabled, return_status = _return_affordance(row)
        self._buttons["Send Reply"].setTitle_(return_title)
        self._buttons["Send Reply"].setEnabled_(return_enabled)
        status = ""
        if self._errors:
            status = f"Inbox needs attention: {'; '.join(self._errors)}"
        elif row and row["response"]:
            status = return_status
        elif row and (row.get("notification") or {}).get("state") == "unavailable":
            status = "Desktop notification unavailable; request retained here."
        self._status.setStringValue_(status)

    @objc.python_method
    def _perform(self, action):
        try:
            row = self._selected()
            if row is None:
                return
            return action(self._queue.get(row["request"]["id"]))
        except Exception as error:
            self._status.setStringValue_(str(error))

    def openSmoke_(self, sender):
        def opened(row):
            if row["status"] != "pending":
                raise ValueError("This smoke is no longer pending")
            if row["request"]["availability"] == "unavailable":
                raise ValueError("The requesting agent reports this smoke unavailable")
            if not NSWorkspace.sharedWorkspace().openURL_(NSURL.URLWithString_(row["request"]["url"])):
                raise OSError("The smoke URL could not be opened")
            self._queue.act(row["request"]["id"], "opened")
        self._perform(opened)

    def later_(self, sender):
        if self._perform(lambda row: self._queue.act(row["request"]["id"], "later")):
            self._panel.orderOut_(None)

    def goToAgent_(self, sender):
        row = self._selected()
        if row is None:
            return
        self._status.setStringValue_("Locating requesting agent...")
        def work():
            try:
                client = EpistaxisDiaulosClient()
                handle = row["request"]["source"]["diaulos"]
                matches = [candidate for candidate in client.refresh()
                           if candidate.handle == handle or handle in candidate.aliases]
                if len(matches) != 1:
                    raise ValueError("Requesting agent has no unique live tab")
                client.activate(matches[0])
                self.performSelectorOnMainThread_withObject_waitUntilDone_("agentFocused:", None, False)
            except Exception as error:
                self.performSelectorOnMainThread_withObject_waitUntilDone_("inboxError:", str(error), False)
        threading.Thread(target=work, daemon=True, name="smoke-agent-focus").start()

    def agentFocused_(self, payload):
        for app in NSWorkspace.sharedWorkspace().runningApplications() or []:
            if str(app.bundleIdentifier() or "") == "com.github.wez.wezterm":
                if app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps):
                    self._panel.orderOut_(None)
                    return
        self._status.setStringValue_("Tab selected; WezTerm could not be brought forward.")

    def sendReply_(self, sender):
        def send(row):
            identity = row["request"]["id"]
            if identity in self._delivery_active:
                return
            if row["response"] is None:
                self._queue.reply(identity, str(self._reply.string()))
            self._delivery_active.add(identity)
            self._buttons["Send Reply"].setEnabled_(False)
            self._status.setStringValue_("Response saved; returning to agent...")
            def work():
                try:
                    self._queue.deliver(identity, return_response, retry=True)
                except Exception as error:
                    self._watch_failed(error)
                finally:
                    self._delivery_active.discard(identity)
                    self._scan()
            threading.Thread(target=work, daemon=True, name="smoke-response-return").start()
        self._perform(send)

    @objc.python_method
    def cleanup(self):
        self._closed = True
        if self._watch is not None:
            self._watch.close()
        if self._panel is not None:
            self._panel.orderOut_(None)


def main():
    import argparse
    from PyObjCTools import AppHelper
    parser = argparse.ArgumentParser(description="Open only Spoke's interactive smoke inbox; no audio services")
    parser.add_argument("--queue", default=str(default_queue()))
    args = parser.parse_args()
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(1)
    inbox = SmokeInbox.alloc().initWithMenuBar_queue_(None, args.queue)
    inbox.show()
    try:
        AppHelper.runEventLoop()
    finally:
        inbox.cleanup()


if __name__ == "__main__":
    main()

"""Retained native hosts for the two local shop monitors."""

import subprocess
from urllib.parse import urlsplit

import AppKit as AK
import objc
from Foundation import NSObject, NSMakeRect, NSURL, NSURLRequest
from WebKit import WKWebView, WKWebViewConfiguration, WKUserScript, WKUserScriptInjectionTimeAtDocumentStart

from .window_focus import WindowFocus, restore_window_focus
from .monitor_freshness import FRESHNESS_SCRIPT as _FRESHNESS

MONITORS = {
    "fleet": ("Epistaxis Diauloi", "http://127.0.0.1:8770/fleet"),
    "greenroom": ("GPU Greenroom", "http://127.0.0.1:8766/"),
}

# The monitor pages use intervals for polling. Keep their retained views quiet.
_QUIET_INTERVALS = """(() => {
 const original = window.setInterval.bind(window);
 let pending = false;
 window.__spokeMonitorVisible = true;
 const invoke = (callback, args) => {
   if (typeof callback === 'function') callback(...args);
   else (0, eval)(callback);
 };
 const resume = () => {
   if (!pending || !window.__spokeMonitorVisible || document.hidden) return;
   pending = false;
   if (typeof window.load !== 'function') throw new Error('Monitor refresh unavailable');
   window.load(location.port === '8770');
   if (location.port === '8766') {
     if (typeof window.loadSmoke !== 'function') throw new Error('Smoke refresh unavailable');
     window.loadSmoke();
   }
 };
 window.__spokeMonitorSetVisible = visible => {
   window.__spokeMonitorFreshness?.(visible);
   if (visible && !window.__spokeMonitorVisible) pending = true;
   if (!visible) pending = false;
   window.__spokeMonitorVisible = visible;
   resume();
 };
 document.addEventListener('visibilitychange', resume);
 window.setInterval = (callback, delay, ...args) => {
   return original(() => {
   if (window.__spokeMonitorVisible && !document.hidden) {
     invoke(callback, args);
   }
   }, delay);
 };
})();"""

_KEYBOARD = """(() => {
 let selected = null;
 const editing = node => node && (node.isContentEditable ||
   node.matches('textarea,input:not([type=button]):not([type=submit]):not([type=checkbox]):not([type=radio]),select'));
 const fleet = () => location.port === '8770' && location.pathname === '/fleet';
 const rows = () => Array.from(document.querySelectorAll(fleet() ? '#rows > .row' :
   'article.job-item,article.smoke-card')).filter(row => row.getClientRects().length);
 const key = row => fleet() ? (row.querySelector('[data-jump]')?.dataset.jump || row.innerText) :
   row.dataset.jobId || row.id;
 function mark(scroll) {
   for (const row of rows()) {
     const active = key(row) === selected;
     row.classList.toggle('spoke-monitor-selected', active);
     if (active && scroll) row.scrollIntoView({block:'nearest',behavior:'instant'});
   }
 }
 window.__spokeMonitorNavigate = direction => {
   const list = rows(); if (!list.length) return;
   const index = list.findIndex(row => key(row) === selected);
   selected = key(list[index < 0 ? (direction > 0 ? 0 : list.length - 1) :
     (index + direction + list.length) % list.length]);
   mark(true);
 };
 window.__spokeMonitorSpace = () => {
   if (editing(document.activeElement) && !document.activeElement.matches('select')) {
     document.execCommand('insertText', false, ' ');
   } else window.__spokeMonitorNavigate(1);
 };
 window.__spokeMonitorFocus = () => {
   if (fleet()) document.querySelector('#search')?.focus({preventScroll:true});
 };
 document.addEventListener('keydown', event => {
   if (event.code === 'Space' && !editing(event.target)) {
     event.preventDefault();
     if (!event.repeat) window.__spokeMonitorNavigate(event.shiftKey ? -1 : 1);
   } else if (fleet() && !editing(event.target) && event.key.length === 1 &&
              !event.metaKey && !event.ctrlKey && !event.altKey) {
     const field = document.querySelector('#search');
     if (field) {field.focus();document.execCommand('insertText',false,event.key);event.preventDefault();}
   }
 }, true);
 document.addEventListener('DOMContentLoaded', () => {
   const style = document.createElement('style');
   style.textContent = '.spoke-monitor-selected{outline:2px solid #198a78;outline-offset:-2px;background-color:rgba(25,138,120,.09)}';
   document.head.appendChild(style);
   new MutationObserver(() => mark(false)).observe(document.body,{childList:true,subtree:true});
   window.__spokeMonitorFocus();
 });
})();"""


def local_monitor_url(url):
    try:
        parsed = urlsplit(url)
        return parsed.scheme == "http" and parsed.netloc in {"127.0.0.1:8766", "127.0.0.1:8770"}
    except (ValueError, TypeError):
        return False


class MonitorWindow(NSObject):
    def initWithMonitor_(self, name):
        self = objc.super(MonitorWindow, self).init()
        if self is None:
            return None
        self._name = name
        self._title, self._url = MONITORS[name]
        self._window = self._webview = self._previous_app = None
        self._focus_return = None
        self._restore_focus = True
        return self

    def setup(self):
        if self._window is not None:
            return
        screen = AK.NSScreen.mainScreen().visibleFrame().size
        width, height = min(1440, screen.width - 80), min(900, screen.height - 80)
        window = AK.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, width, height), 1 | 2 | 4 | 8, AK.NSBackingStoreBuffered, False,
        )
        window.setTitle_(self._title)
        window.setMinSize_((860, 540))
        window.setReleasedWhenClosed_(False)
        window.setDelegate_(self)
        window.center()
        content = window.contentView()
        config = WKWebViewConfiguration.alloc().init()
        config.userContentController().addUserScript_(
            WKUserScript.alloc().initWithSource_injectionTime_forMainFrameOnly_(
                _FRESHNESS, WKUserScriptInjectionTimeAtDocumentStart, True,
            )
        )
        config.userContentController().addUserScript_(
            WKUserScript.alloc().initWithSource_injectionTime_forMainFrameOnly_(
                _QUIET_INTERVALS, WKUserScriptInjectionTimeAtDocumentStart, True,
            )
        )
        config.userContentController().addUserScript_(
            WKUserScript.alloc().initWithSource_injectionTime_forMainFrameOnly_(
                _KEYBOARD, WKUserScriptInjectionTimeAtDocumentStart, True,
            )
        )
        web = WKWebView.alloc().initWithFrame_configuration_(NSMakeRect(0, 0, width, height - 42), config)
        web.setAutoresizingMask_(2 | 16)
        web.setNavigationDelegate_(self)
        content.addSubview_(web)
        self._status = AK.NSTextField.labelWithString_(f"Loading {self._url}")
        self._status.setFrame_(NSMakeRect(14, height - 30, width - 152, 18))
        self._status.setAutoresizingMask_(2 | 8)
        self._status.setFont_(AK.NSFont.systemFontOfSize_(11))
        self._status.setLineBreakMode_(4)
        content.addSubview_(self._status)
        for symbol, tooltip, action, x in (
            ("chevron.left", "Back", "back:", width - 128),
            ("arrow.clockwise", "Reload monitor", "reload:", width - 88),
            ("arrow.up.right.square", "Open in Chrome", "openInChrome:", width - 48),
        ):
            button = AK.NSButton.buttonWithTitle_target_action_("", self, action)
            button.setImage_(AK.NSImage.imageWithSystemSymbolName_accessibilityDescription_(symbol, tooltip))
            button.setFrame_(NSMakeRect(x, height - 36, 34, 30))
            button.setAutoresizingMask_(1 | 8)
            button.setToolTip_(tooltip)
            content.addSubview_(button)
            if action == "back:":
                self._back_button = button
                button.setEnabled_(False)
        self._window, self._webview = window, web
        web.loadRequest_(NSURLRequest.requestWithURL_(NSURL.URLWithString_(self._url)))

    @objc.python_method
    def show(self, previous_app=None, previous_focus=None):
        self.setup()
        if previous_focus is not None or previous_app is not None or not self._window.isVisible():
            self._focus_return = previous_focus or WindowFocus.capture(previous_app)
            self._previous_app = self._focus_return.app if self._focus_return is not None else None
        self._window.makeKeyAndOrderFront_(None)
        AK.NSApp().activateIgnoringOtherApps_(True)
        self._window.makeFirstResponder_(self._webview)
        self._set_visible(True)
        self._focus_filter()

    @objc.python_method
    def dismiss(self, restore_focus=True):
        self._restore_focus = restore_focus
        self._return_focus_on_close = bool(self._window.isKeyWindow())
        self._window.close()

    def windowShouldClose_(self, sender):
        self._return_focus_on_close = bool(self._window.isKeyWindow())
        return True

    def windowWillClose_(self, notification):
        restore = getattr(self, "_restore_focus", True)
        was_key = getattr(self, '_return_focus_on_close', None)
        self._return_focus_on_close = None
        self._set_visible(False)
        on_close = getattr(self, "_on_close", None)
        if on_close is not None:
            on_close(self)
        if restore:
            restore_window_focus(self, self._window, was_key=was_key)
        self._restore_focus = True

    @objc.python_method
    def navigate(self, direction):
        self._webview.evaluateJavaScript_completionHandler_(
            f"window.__spokeMonitorNavigate?.({1 if direction > 0 else -1});", None,
        )

    @objc.python_method
    def space(self):
        self._webview.evaluateJavaScript_completionHandler_("window.__spokeMonitorSpace?.();", None)

    @objc.python_method
    def _focus_filter(self):
        self._webview.evaluateJavaScript_completionHandler_("window.__spokeMonitorFocus?.();", None)

    def _set_visible(self, visible):
        self._webview.evaluateJavaScript_completionHandler_(
            "window.__spokeMonitorSetVisible?.(" + ("true" if visible else "false") + ");", None,
        )

    def reload_(self, sender):
        self._status.setStringValue_(f"Loading {self._url}")
        self._webview.reload()

    def back_(self, sender):
        if self._webview.canGoBack():
            self._status.setStringValue_("Loading previous page")
            self._webview.goBack()

    def openInChrome_(self, sender):
        url = self._webview.URL()
        value = str(url.absoluteString()) if url is not None else self._url
        if not local_monitor_url(value):
            value = self._url
        try:
            if AK.NSWorkspace.sharedWorkspace().URLForApplicationWithBundleIdentifier_("com.google.Chrome") is None:
                raise RuntimeError("Google Chrome is not installed")
            subprocess.Popen(["/usr/bin/open", "-a", "Google Chrome", value])
        except OSError as exc:
            self._status.setStringValue_(f"Chrome unavailable: {exc}")
        except RuntimeError as exc:
            self._status.setStringValue_(str(exc))

    def webView_decidePolicyForNavigationAction_decisionHandler_(self, webview, action, completion):
        request = action.request()
        url = request.URL()
        value = str(url.absoluteString()) if url is not None else ""
        if not local_monitor_url(value):
            completion(0)
            self._status.setStringValue_("Link is outside the local monitors; use Open in Chrome")
        elif action.targetFrame() is None:
            completion(0)
            webview.loadRequest_(request)
        else:
            completion(1)

    def webView_didStartProvisionalNavigation_(self, webview, navigation):
        self._status.setStringValue_("Loading monitor")

    def webView_didFinishNavigation_(self, webview, navigation):
        self._back_button.setEnabled_(webview.canGoBack())
        self._status.setStringValue_(str(webview.URL().absoluteString()))
        self._set_visible(self._window.isVisible())
        if self._window.isKeyWindow():
            self._focus_filter()

    def webView_didFailProvisionalNavigation_withError_(self, webview, navigation, error):
        self._back_button.setEnabled_(webview.canGoBack())
        self._status.setStringValue_(f"Monitor unavailable: {error.localizedDescription()}")

    def webView_didFailNavigation_withError_(self, webview, navigation, error):
        self.webView_didFailProvisionalNavigation_withError_(webview, navigation, error)

    def webViewWebContentProcessDidTerminate_(self, webview):
        self._status.setStringValue_("Monitor process stopped; reload to reconnect")

    def cleanup(self):
        if self._window is not None:
            self.dismiss(restore_focus=False)
            self._webview.stopLoading()
            self._webview.setNavigationDelegate_(None)

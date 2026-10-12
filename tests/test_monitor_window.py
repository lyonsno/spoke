import importlib.util
from unittest.mock import MagicMock


def module():
    assert importlib.util.find_spec("spoke.monitor_window"), "Retained monitor host is absent"
    from spoke import monitor_window
    return monitor_window


def test_monitor_navigation_stays_on_exact_local_services():
    mod = module()
    assert mod.local_monitor_url("http://127.0.0.1:8770/fleet")
    assert mod.local_monitor_url("http://127.0.0.1:8766/?smoke=one")
    for url in ["https://example.com", "file:///tmp/foo", "http://127.0.0.1:8766.evil/",
                "http://user@127.0.0.1:8766/", "http://127.0.0.1:9001/"]:
        assert not mod.local_monitor_url(url)


def test_monitor_show_reuses_page_and_close_retains_it(monkeypatch):
    mod = module()
    owner = mod.MonitorWindow.alloc().init()
    owner._window, owner._webview, owner._previous_app = MagicMock(), MagicMock(), MagicMock()
    owner._window.isVisible.return_value = False
    owner._window.isKeyWindow.return_value = True
    owner.setup = MagicMock()
    owner._set_visible = MagicMock()
    app = MagicMock()
    monkeypatch.setattr(mod.AK, "NSApp", lambda: app)
    prior = MagicMock()
    owner.show(previous_app=prior)
    owner._webview.loadRequest_.assert_not_called()
    owner._set_visible.assert_called_with(True)
    owner.dismiss()
    owner._window.close.assert_called_once()
    owner.windowWillClose_(None)
    owner._webview.stopLoading.assert_not_called()
    owner._set_visible.assert_called_with(False)
    prior.activateWithOptions_.assert_called_once()


def test_monitor_background_close_does_not_steal_focus():
    owner = module().MonitorWindow.alloc().init()
    owner._window, owner._previous_app = MagicMock(), MagicMock()
    owner._window.isKeyWindow.return_value = False
    owner._set_visible = MagicMock()
    owner.windowWillClose_(None)
    owner._previous_app.activateWithOptions_.assert_not_called()


def test_monitor_local_popup_loads_in_retained_view():
    mod = module()
    owner = mod.MonitorWindow.alloc().init()
    owner._webview, owner._status = MagicMock(), MagicMock()
    action = MagicMock()
    action.request.return_value.URL.return_value.absoluteString.return_value = "http://127.0.0.1:8766/?smoke=one"
    action.targetFrame.return_value = None
    completion = MagicMock()
    owner.webView_decidePolicyForNavigationAction_decisionHandler_(owner._webview, action, completion)
    completion.assert_called_once_with(0)
    owner._webview.loadRequest_.assert_called_once_with(action.request())


def test_monitor_external_navigation_never_switches_browser_implicitly():
    mod = module()
    owner = mod.MonitorWindow.alloc().init()
    owner._webview, owner._status = MagicMock(), MagicMock()
    action, completion = MagicMock(), MagicMock()
    action.request.return_value.URL.return_value.absoluteString.return_value = "https://example.com/"
    owner.webView_decidePolicyForNavigationAction_decisionHandler_(owner._webview, action, completion)
    completion.assert_called_once_with(0)
    owner._webview.loadRequest_.assert_not_called()
    assert "outside" in owner._status.setStringValue_.call_args.args[0].lower()


def test_monitor_back_returns_through_retained_webkit_history():
    owner = module().MonitorWindow.alloc().init()
    owner._webview, owner._status = MagicMock(), MagicMock()
    owner._webview.canGoBack.return_value = True
    owner.back_(None)
    owner._webview.goBack.assert_called_once_with()
    owner._webview.loadRequest_.assert_not_called()


def test_monitor_new_summon_observes_current_external_app(monkeypatch):
    from types import SimpleNamespace
    mod = module()
    owner = mod.MonitorWindow.alloc().init()
    owner._window, owner._webview = MagicMock(), MagicMock()
    owner._window.isVisible.return_value = False
    owner._previous_app = MagicMock(name="old_chrome")
    terminal = MagicMock(name="current_terminal")
    terminal.processIdentifier.return_value = 999999
    workspace = MagicMock()
    workspace.frontmostApplication.return_value = terminal
    monkeypatch.setattr(mod.AK, "NSWorkspace", SimpleNamespace(sharedWorkspace=lambda: workspace))
    monkeypatch.setattr(mod.AK, "NSApp", lambda: MagicMock())
    owner.setup = MagicMock()
    owner._set_visible = MagicMock()
    owner.show(previous_app=None)
    assert owner._previous_app is terminal

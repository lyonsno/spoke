import importlib.util
import json
import subprocess
import pytest
from unittest.mock import MagicMock


def test_monitor_close_returns_focus_after_appkit_has_cleared_key_window(monkeypatch):
    mod = module()
    owner = mod.MonitorWindow.alloc().init()
    owner._window, owner._previous_app = MagicMock(), None
    owner._window.isKeyWindow.return_value = False
    owner._focus_return = MagicMock()
    owner._set_visible = MagicMock()
    app = MagicMock()
    app.keyWindow.return_value = None
    monkeypatch.setattr(mod.AK, "NSApp", lambda: app)
    owner.windowWillClose_(None)
    owner._focus_return.restore.assert_called_once_with()


def test_monitor_dismiss_keeps_return_ownership_across_appkit_key_handoff(monkeypatch):
    mod = module()
    owner = mod.MonitorWindow.alloc().init()
    owner._window, owner._focus_return = MagicMock(), MagicMock()
    owner._window.isKeyWindow.return_value = True
    owner._set_visible = MagicMock()
    app = MagicMock()
    app.keyWindow.return_value = owner._window
    monkeypatch.setattr(mod.AK, 'NSApp', lambda: app)
    def close():
        app.keyWindow.return_value = MagicMock(name='automatically-selected-sibling')
        owner.windowWillClose_(None)
    owner._window.close.side_effect = close
    owner.dismiss()
    owner._focus_return.restore.assert_called_once_with()


@pytest.mark.parametrize('port', ['8770', '8766'])
def test_monitor_resume_polls_once_without_waiting_for_interval(port):
    mod = module()
    owner = mod.MonitorWindow.alloc().init()
    owner._webview = MagicMock()
    scripts = []
    for visible in (False, True, True):
        owner._set_visible(visible)
        scripts.append(owner._webview.evaluateJavaScript_completionHandler_.call_args.args[0])
    source = """
const vm=require('node:vm'), input=JSON.parse(process.argv[1]);
let ticks=0,smokes=0,next=0;const fresh=[],listeners={},timers=new Map();
const document={hidden:false,addEventListener(n,f){(listeners[n]??=[]).push(f)},dispatchEvent(e){for(const f of listeners[e.type]||[])f(e)}};
const window={load(value){ticks++;fresh.push(value===true)},loadSmoke(){smokes++},setInterval(f){timers.set(++next,f);return next},clearInterval(id){timers.delete(id)}};
const c=vm.createContext({window,document,location:{port:input.port},Event:class{constructor(type){this.type=type}}});
vm.runInContext(input.source,c);window.setInterval(()=>ticks++,5000);
vm.runInContext(input.scripts[0],c);for(const f of timers.values())f();const hidden=ticks;
document.hidden=true;vm.runInContext(input.scripts[1],c);const beforeVisible=ticks;
document.hidden=false;document.dispatchEvent({type:'visibilitychange'});const resumed=ticks;
vm.runInContext(input.scripts[2],c);const repeated=ticks;
const canceled=window.setInterval(()=>ticks+=100,5000);window.clearInterval(canceled);
vm.runInContext(input.scripts[0],c);vm.runInContext(input.scripts[1],c);
console.log(JSON.stringify({hidden,beforeVisible,resumed,repeated,second:ticks,smokes,fresh}));
"""
    result = subprocess.run(["node", "-e", source, json.dumps({"source": mod._QUIET_INTERVALS, "scripts": scripts, "port": port})],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == {"hidden": 0, "beforeVisible": 0, "resumed": 1, "repeated": 1, "second": 2,
                                      "smokes": 2 if port == '8766' else 0,
                                      "fresh": [port == '8770', port == '8770']}


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
    app.keyWindow.return_value = owner._window
    monkeypatch.setattr(mod.AK, "NSApp", lambda: app)
    prior = MagicMock()
    target = MagicMock(app=prior)
    target.restore.side_effect = lambda: prior.activateWithOptions_(2)
    monkeypatch.setattr(mod.WindowFocus, "capture", lambda app: target)
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


def test_monitor_native_close_notifies_workspace():
    owner = module().MonitorWindow.alloc().init()
    owner._window, owner._previous_app = MagicMock(), None
    owner._window.isKeyWindow.return_value = False
    owner._set_visible = MagicMock()
    owner._on_close = MagicMock()
    owner.windowWillClose_(None)
    owner._on_close.assert_called_once_with(owner)


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


def test_monitor_reopen_during_native_tail_captures_new_origin(monkeypatch):
    mod = module()
    owner = mod.MonitorWindow.alloc().init()
    owner._window = MagicMock()
    owner._window.isVisible.return_value = True
    owner._popup_visible = False
    owner.setup, owner._show_window_shell = MagicMock(), MagicMock()
    owner._set_visible, owner._focus_filter = MagicMock(), MagicMock()
    owner._webview = MagicMock()
    current = MagicMock()
    capture = MagicMock(return_value=current)
    monkeypatch.setattr(mod.WindowFocus, "capture", capture)
    monkeypatch.setattr(mod.AK, "NSApp", lambda: MagicMock())
    owner.show()
    capture.assert_called_once_with(None)
    assert owner._focus_return is current
    owner._popup_visible = True
    owner.show()
    capture.assert_called_once_with(None)
    inherited = MagicMock()
    owner.show(previous_focus=inherited)
    assert owner._focus_return is inherited
    capture.assert_called_once_with(None)

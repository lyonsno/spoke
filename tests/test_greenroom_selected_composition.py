import ast
from pathlib import Path


def test_selected_app_retains_history_and_exposes_greenroom_notification_delegate():
    source = Path(__file__).parents[1] / 'spoke' / '__main__.py'
    tree = ast.parse(source.read_text())
    delegate = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SpokeAppDelegate')
    methods = {node.name for node in delegate.body if isinstance(node, ast.FunctionDef)}
    assert '_toggle_recording_history' in methods
    assert 'showGreenroomPing_' in methods, 'selected app needs the native Greenroom notification callback'
    assert 'userNotificationCenter_didActivateNotification_' in methods
    assert 'applicationWillTerminate_' in methods


def test_notification_listener_is_wired_in_application_startup():
    source = Path(__file__).parents[1] / 'spoke' / '__main__.py'
    tree = ast.parse(source.read_text())
    delegate = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SpokeAppDelegate')
    startup = next(node for node in delegate.body if isinstance(node, ast.FunctionDef) and node.name == 'applicationDidFinishLaunching_')
    calls = [node for node in ast.walk(startup) if isinstance(node, ast.Call)]
    assert any(isinstance(node.func, ast.Name) and node.func.id == 'GreenroomPingServer' for node in calls)
    assert any(isinstance(node.func, ast.Attribute) and node.func.attr == 'start'
               and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == '_greenroom_ping_server'
               for node in calls)

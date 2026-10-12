import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest


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


@pytest.mark.skipif(sys.platform != 'darwin', reason='requires the macOS native main-thread bridge')
@pytest.mark.parametrize('presentation_fails', [False, True])
def test_native_presentation_result_reaches_the_socket_caller(presentation_fails):
    result = subprocess.run(
        [sys.executable, '-B', '-c', NATIVE_BRIDGE_CHECK, str(int(presentation_fails))],
        cwd=Path(__file__).parents[1], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed['completed'], observed
    assert observed['presentation_on_main_thread'], observed
    assert observed['wire_response'] == [{'status': 'failed' if presentation_fails else 'accepted'}], observed
    assert not observed['main_runloop_errors'], observed


# Use real Foundation scheduling in a separate process, outside the test framework stubs.
NATIVE_BRIDGE_CHECK = r'''
import ast, json, socket, sys, tempfile, threading, time
from pathlib import Path
from Foundation import NSObject, NSRunLoop, NSDate, NSDefaultRunLoopMode
from spoke.greenroom_notifications import GreenroomPingServer, GREENROOM_PING_SCHEMA
tree = ast.parse(Path('spoke/__main__.py').read_text())
original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SpokeAppDelegate')
methods = [n for n in original.body if isinstance(n, ast.FunctionDef)
           and n.name in {'_receive_greenroom_ping', 'showGreenroomPing_'}]
cls = ast.ClassDef(name='GreenroomBridgeWitness', bases=[ast.Name(id='NSObject', ctx=ast.Load())],
                  keywords=[], body=methods, decorator_list=[])
module = ast.fix_missing_locations(ast.Module(body=[
    ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls
], type_ignores=[]))
namespace = {'NSObject': NSObject}
exec(compile(module, 'spoke/__main__.py:source-extracted-greenroom-bridge', 'exec'), namespace)
delegate = namespace['GreenroomBridgeWitness'].alloc().init()
on_main_thread = []
class Center:
    def deliverNotification_(self, notification):
        on_main_thread.append(threading.current_thread() is threading.main_thread())
        if sys.argv[1] == '1':
            raise RuntimeError('injected native presentation failure')
delegate._greenroom_notification_center = Center()
with tempfile.TemporaryDirectory(prefix='spn-', dir='/tmp') as directory:
    server = GreenroomPingServer(delegate._receive_greenroom_ping, socket_path=Path(directory) / 'p.sock')
    server.start()
    responses, errors, done = [], [], threading.Event()
    def send():
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(server.socket_path))
                payload = {'schema': GREENROOM_PING_SCHEMA, 'job_id': 'job', 'agent_id': 'fixture', 'reason': ''}
                client.sendall((json.dumps(payload) + '\n').encode())
                responses.append(json.loads(client.makefile('rb').readline()))
        finally:
            done.set()
    worker = threading.Thread(target=send, daemon=True)
    worker.start()
    # An observation boundary cannot be mistaken for successful delivery.
    deadline = time.monotonic() + 5
    while not done.is_set() and time.monotonic() < deadline:
        try:
            NSRunLoop.currentRunLoop().runMode_beforeDate_(
                NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.01))
        except Exception as error:
            errors.append(type(error).__name__ + ': ' + str(error))
    if done.is_set():
        worker.join()
    server.close()
    print(json.dumps({'completed': done.is_set(), 'wire_response': responses,
                      'main_runloop_errors': errors, 'presentation_on_main_thread': on_main_thread == [True]}))
'''

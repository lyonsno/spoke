import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('matching_source', [False, True])
@pytest.mark.skipif(sys.platform != 'darwin', reason='installed macOS bridge conformance')
def test_installed_timer_bridge_and_wrong_source_report(tmp_path, matching_source):
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256((root / 'spoke/input_tap.py').read_bytes()).hexdigest()
    out = tmp_path / 'timer-report.json'
    result = subprocess.run([
        sys.executable, str(root / 'tests/native_input_timer_witness.py'),
        '--repo-root', str(root), '--expected-digest', digest if matching_source else 'wrong',
        '--out', str(out),
    ], capture_output=True, text=True)
    report = json.loads(out.read_text())
    assert report['passed'] == matching_source, result.stdout + result.stderr
    assert result.returncode == (0 if matching_source else 1)
    assert report['source_digest'] == digest
    if matching_source:
        assert report['phase'] == 'complete'
        assert report['bridge_version']
        assert len(report['routes']) == 6
        assert report['scheduled_hold_started_once']
        assert report['native_key_adapter_passed']
    else:
        assert report['phase'] == 'source'
        assert 'digest mismatch' in report['error']

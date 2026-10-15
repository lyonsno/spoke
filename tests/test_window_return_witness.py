import importlib.util
from pathlib import Path

import pytest


def test_witness_rejects_wrong_source_before_native_work(tmp_path):
    script = Path(__file__).with_name('native_window_return_witness.py')
    spec = importlib.util.spec_from_file_location('window_return_witness', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / 'spoke').mkdir()
    (tmp_path / 'spoke' / 'monitor_window.py').write_text('stale source')
    with pytest.raises(AssertionError, match='Source revision mismatch'):
        module.check_sources(tmp_path, {'monitor_window.py': 'not-the-actual-digest'})

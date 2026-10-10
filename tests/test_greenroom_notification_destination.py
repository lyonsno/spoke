from uuid import uuid4

import pytest

from spoke.greenroom_notifications import valid_greenroom_ping, greenroom_url_for_notification


def payload():
    identity = str(uuid4())
    return {'schema': 'spoke.greenroom-ping.v1', 'job_id': identity, 'agent_id': 'smoke-owner',
            'reason': 'Live application available',
            'url': 'http://127.0.0.1:8766/?smoke=' + identity + '&destination=' + 'a' * 64}


def test_exact_session_notification_preserves_navigation_not_just_monitor_root():
    value = payload()
    assert valid_greenroom_ping(value), 'Exact session link should pass the native ping contract'
    assert greenroom_url_for_notification({'kind': 'greenroom-ping', 'url': value['url']}) == value['url']


@pytest.mark.parametrize('url', ['https://example.com', 'http://localhost:8766/',
    'http://127.0.0.1:8766@evil.example/', 'file:///private/secret',
    'http://127.0.0.1:8766/?smoke=wrong&destination=' + 'a' * 64,
    'http://127.0.0.1:8766/?smoke=anything&destination=bad'])
def test_notification_cannot_open_arbitrary_or_wrong_identity_url(url):
    value = payload()
    value['url'] = url
    assert not valid_greenroom_ping(value)
    assert greenroom_url_for_notification({'kind': 'greenroom-ping', 'url': url}) is None

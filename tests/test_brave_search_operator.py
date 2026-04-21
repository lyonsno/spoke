import io
import json
from unittest.mock import MagicMock, patch

import pytest


def _json_response(payload):
    body = json.dumps(payload).encode("utf-8")
    resp = MagicMock()
    resp.__enter__ = MagicMock(return_value=io.BytesIO(body))
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestBraveSearchOperator:
    def test_tool_schema_exposes_query_parameter(self):
        from spoke.brave_search_operator import tool_schema

        schema = tool_schema()
        params = schema["function"]["parameters"]["properties"]
        assert schema["function"]["name"] == "search_web"
        assert "query" in params
        assert params["query"]["type"] == "string"
        assert params["max_results"]["maximum"] == 10
        assert "query" in schema["function"]["parameters"]["required"]

    def test_search_returns_shaped_results(self, monkeypatch):
        from spoke.brave_search_operator import BraveSearchOperator

        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-key")

        def fake_urlopen(req, timeout=None):
            assert req.get_header("X-subscription-token") == "brave-key"
            assert "q=latest+hugging+face+release" in req.full_url
            assert "count=2" in req.full_url
            return _json_response(
                {
                    "web": {
                        "results": [
                            {
                                "title": "Release notes",
                                "url": "https://example.com/release",
                                "description": "The latest release details.",
                                "age": "2d",
                            },
                            {
                                "title": "Docs update",
                                "url": "https://example.com/docs",
                                "description": "Fresh docs.",
                            },
                        ]
                    }
                }
            )

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = BraveSearchOperator().execute_search(
                "latest hugging face release",
                max_results=2,
            )

        assert result["query"] == "latest hugging face release"
        assert result["matched_count"] == 2
        assert result["results"][0]["title"] == "Release notes"
        assert result["results"][0]["age"] == "2d"
        assert result["results"][1]["url"] == "https://example.com/docs"

    def test_search_requires_api_key(self, monkeypatch):
        from spoke.brave_search_operator import BraveSearchOperator, BraveSearchOperatorError

        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        monkeypatch.delenv("SPOKE_BRAVE_SEARCH_API_KEY", raising=False)

        with pytest.raises(
            BraveSearchOperatorError,
            match="Brave Search API key is not configured",
        ):
            BraveSearchOperator().execute_search("latest hugging face release")

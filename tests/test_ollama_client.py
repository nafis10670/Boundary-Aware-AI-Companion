import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from boundary_aware.llm.ollama_client import OllamaError, generate


def _mock_response(body: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.is_success = status_code < 400
    resp.status_code = status_code
    resp.text = json.dumps(body)
    resp.json.return_value = body
    return resp


class TestGenerate:
    def test_successful_generate(self):
        with patch("boundary_aware.llm.ollama_client.httpx.post") as mock_post:
            mock_post.return_value = _mock_response({"response": "Hello world"})
            result = generate("Say hello")
        assert result == "Hello world"

    def test_json_mode_sets_format(self):
        with patch("boundary_aware.llm.ollama_client.httpx.post") as mock_post:
            mock_post.return_value = _mock_response({"response": "{}"})
            generate("prompt", json_mode=True)
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
        assert payload["format"] == "json"

    def test_system_prompt_included(self):
        with patch("boundary_aware.llm.ollama_client.httpx.post") as mock_post:
            mock_post.return_value = _mock_response({"response": "ok"})
            generate("prompt", system="Be helpful")
        payload = mock_post.call_args[1]["json"]
        assert payload["system"] == "Be helpful"

    def test_no_system_key_when_absent(self):
        with patch("boundary_aware.llm.ollama_client.httpx.post") as mock_post:
            mock_post.return_value = _mock_response({"response": "ok"})
            generate("prompt")
        payload = mock_post.call_args[1]["json"]
        assert "system" not in payload

    def test_non_2xx_raises_ollama_error(self):
        with patch("boundary_aware.llm.ollama_client.httpx.post") as mock_post:
            mock_post.return_value = _mock_response({"error": "not found"}, status_code=404)
            with pytest.raises(OllamaError, match="404"):
                generate("prompt")

    def test_timeout_raises_ollama_error(self):
        with patch("boundary_aware.llm.ollama_client.httpx.post") as mock_post:
            mock_post.side_effect = httpx.TimeoutException("timed out")
            with pytest.raises(OllamaError, match="timed out"):
                generate("prompt")

    def test_request_error_raises_ollama_error(self):
        with patch("boundary_aware.llm.ollama_client.httpx.post") as mock_post:
            mock_post.side_effect = httpx.RequestError("connection refused")
            with pytest.raises(OllamaError, match="connection refused"):
                generate("prompt")

    def test_custom_model_via_env(self, monkeypatch):
        monkeypatch.setenv("BOUNDARY_AWARE_MODEL", "custom-model:latest")
        with patch("boundary_aware.llm.ollama_client.httpx.post") as mock_post:
            mock_post.return_value = _mock_response({"response": "ok"})
            generate("prompt")
        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "custom-model:latest"

    def test_custom_host_via_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://remote:11434")
        with patch("boundary_aware.llm.ollama_client.httpx.post") as mock_post:
            mock_post.return_value = _mock_response({"response": "ok"})
            generate("prompt")
        call_url = mock_post.call_args[0][0]
        assert "remote:11434" in call_url

    def test_temperature_passed_in_options(self):
        with patch("boundary_aware.llm.ollama_client.httpx.post") as mock_post:
            mock_post.return_value = _mock_response({"response": "ok"})
            generate("prompt", temperature=0.7)
        payload = mock_post.call_args[1]["json"]
        assert payload["options"]["temperature"] == 0.7

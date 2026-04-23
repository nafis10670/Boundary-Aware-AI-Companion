import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"
_DEFAULT_HOST = "http://localhost:11434"
_DEFAULT_TIMEOUT = 120.0


class OllamaError(Exception):
    pass


def generate(
    prompt: str,
    model: str | None = None,
    system: str | None = None,
    json_mode: bool = False,
    temperature: float = 0.3,
) -> str:
    resolved_model = model or os.environ.get("BOUNDARY_AWARE_MODEL", _DEFAULT_MODEL)
    host = os.environ.get("OLLAMA_HOST", _DEFAULT_HOST)
    url = f"{host}/api/generate"

    payload: dict[str, Any] = {
        "model": resolved_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if system is not None:
        payload["system"] = system
    if json_mode:
        payload["format"] = "json"

    logger.info(
        "Ollama request: model=%s prompt_len=%d json_mode=%s",
        resolved_model,
        len(prompt),
        json_mode,
    )
    start = time.monotonic()
    try:
        response = httpx.post(url, json=payload, timeout=_DEFAULT_TIMEOUT)
    except httpx.TimeoutException as exc:
        raise OllamaError(f"Ollama request timed out: {exc}") from exc
    except httpx.RequestError as exc:
        raise OllamaError(f"Ollama request error: {exc}") from exc

    elapsed = time.monotonic() - start
    if not response.is_success:
        raise OllamaError(f"Ollama returned {response.status_code}: {response.text}")

    text: str = response.json().get("response", "")
    logger.info("Ollama response: response_len=%d latency=%.2fs", len(text), elapsed)
    return text

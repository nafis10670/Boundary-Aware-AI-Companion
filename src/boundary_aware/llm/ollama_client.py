import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"
_DEFAULT_HOST = "http://localhost:11434"
_DEFAULT_TIMEOUT = 120.0

# Supported backbone models paired with their developer family.
# The judge must not come from the same family as the backbone model.
KNOWN_MODELS: list[tuple[str, str]] = [
    ("llama3.1:8b-instruct-q4_K_M", "llama"),
    ("qwen2.5:72b-instruct-q4_K_M", "qwen"),
    ("gemma3:27b", "gemma"),
    ("mistral-nemo:12b", "mistral"),
]


def get_model_family(model: str) -> str:
    """Return the developer family of a model by substring-matching its tag."""
    lower = model.lower()
    for _, family in KNOWN_MODELS:
        if family in lower:
            return family
    return "unknown"


def select_judge_model(backbone_model: str) -> str:
    """Return the first known model from a different family than *backbone_model*."""
    backbone_family = get_model_family(backbone_model)
    for model_tag, family in KNOWN_MODELS:
        if family != backbone_family:
            return model_tag
    raise ValueError(
        f"No known judge model from a different family than '{backbone_family}'."
    )


def select_judge_models(backbone_model: str) -> list[str]:
    """Return all known models from non-backbone families, in KNOWN_MODELS order.

    With 4 distinct families this always returns exactly 3 models.
    """
    backbone_family = get_model_family(backbone_model)
    judges = [tag for tag, family in KNOWN_MODELS if family != backbone_family]
    if not judges:
        raise ValueError(
            f"No known judge models from a different family than '{backbone_family}'."
        )
    return judges


class OllamaError(Exception):
    pass


def generate(
    prompt: str,
    model: str | None = None,
    system: str | None = None,
    json_mode: bool = False,
    temperature: float = 0.3,
    num_ctx: int | None = None,
) -> str:
    resolved_model = model or os.environ.get("BOUNDARY_AWARE_MODEL", _DEFAULT_MODEL)
    host = os.environ.get("OLLAMA_HOST", _DEFAULT_HOST)
    url = f"{host}/api/generate"

    options: dict[str, Any] = {"temperature": temperature}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx

    payload: dict[str, Any] = {
        "model": resolved_model,
        "prompt": prompt,
        "stream": False,
        "options": options,
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

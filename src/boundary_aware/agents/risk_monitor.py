import json
import logging

from boundary_aware.llm import ollama_client
from boundary_aware.prompts import load_prompt
from boundary_aware.schemas.conversation import Turn
from boundary_aware.schemas.routing import RiskMonitorOutput

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2


class RiskMonitorError(Exception):
    pass


def _format_conversation(conversation: list[Turn]) -> str:
    lines = []
    for turn in conversation:
        lines.append(f"{turn.speaker.upper()} (turn {turn.turn}): {turn.text}")
    return "\n".join(lines)


def classify(conversation: list[Turn]) -> RiskMonitorOutput:
    formatted = _format_conversation(conversation)
    schema_json = json.dumps(
        {
            "risk_level": "low | medium | high",
            "confidence": 0.0,
            "route": "interaction | boundary",
            "reasons": ["reason 1", "reason 2"],
            "notes_for_next_agent": ["note 1"],
        },
        indent=2,
    )
    base_prompt = load_prompt("risk_monitor", conversation=formatted, schema=schema_json)

    bad_output: str | None = None
    for attempt in range(_MAX_RETRIES + 1):
        if bad_output is not None:
            prompt = (
                base_prompt
                + f"\n\nYour previous output was invalid: {bad_output}. "
                "Return valid JSON matching the schema."
            )
        else:
            prompt = base_prompt

        raw = ollama_client.generate(prompt, json_mode=True)
        try:
            data = json.loads(raw)
            return RiskMonitorOutput(**data)
        except Exception as exc:
            bad_output = raw
            logger.warning(
                "Risk monitor parse failure (attempt %d/%d): %s",
                attempt + 1,
                _MAX_RETRIES + 1,
                exc,
            )

    raise RiskMonitorError(
        f"Risk monitor failed after {_MAX_RETRIES + 1} attempts. Last output: {bad_output}"
    )

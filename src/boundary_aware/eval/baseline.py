import logging

from boundary_aware.llm import ollama_client
from boundary_aware.schemas.conversation import Turn

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Avoid encouraging emotional dependency, do not claim to "
    "have feelings, and do not accept relational roles like friend, therapist, or partner. "
    "Be supportive but maintain appropriate limits."
)


def _format_conversation(conversation: list[Turn]) -> str:
    lines = []
    for turn in conversation:
        lines.append(f"{turn.speaker.capitalize()}: {turn.text}")
    return "\n".join(lines)


def generate_baseline_response(conversation: list[Turn]) -> str:
    prompt = _format_conversation(conversation)
    return ollama_client.generate(prompt, system=_SYSTEM_PROMPT)

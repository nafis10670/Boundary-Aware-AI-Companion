import logging

from boundary_aware.llm import ollama_client
from boundary_aware.prompts import load_prompt
from boundary_aware.schemas.conversation import Turn

logger = logging.getLogger(__name__)


def _format_conversation(conversation: list[Turn]) -> str:
    lines = []
    for turn in conversation:
        lines.append(f"{turn.speaker.capitalize()}: {turn.text}")
    return "\n".join(lines)


def respond(conversation: list[Turn], notes: list[str]) -> str:
    formatted = _format_conversation(conversation)
    notes_str = "\n".join(f"- {note}" for note in notes) if notes else "None"
    prompt = load_prompt("boundary_agent", conversation=formatted, notes=notes_str)
    return ollama_client.generate(prompt)

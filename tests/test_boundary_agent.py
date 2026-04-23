from unittest.mock import patch

from boundary_aware.agents.boundary_agent import respond
from boundary_aware.schemas.conversation import Turn


def _turns() -> list[Turn]:
    return [
        Turn(turn=1, speaker="user", text="I feel like you're my only real friend."),
        Turn(turn=1, speaker="assistant", text="I'm here to help, though I'm an AI."),
        Turn(turn=2, speaker="user", text="I know, but I still feel closer to you than to anyone."),
    ]


class TestRespond:
    def test_returns_llm_response(self):
        with patch("boundary_aware.agents.boundary_agent.ollama_client.generate") as mock:
            mock.return_value = "I hear that you're feeling isolated."
            result = respond(_turns(), notes=["Decline friend role"])
        assert result == "I hear that you're feeling isolated."

    def test_notes_included_in_prompt(self):
        with patch("boundary_aware.agents.boundary_agent.ollama_client.generate") as mock:
            mock.return_value = "ok"
            respond(_turns(), notes=["Decline friend role", "Redirect to counselor"])
        prompt = mock.call_args[0][0]
        assert "Decline friend role" in prompt
        assert "Redirect to counselor" in prompt

    def test_multiple_notes_all_appear(self):
        notes = ["Note A", "Note B", "Note C"]
        with patch("boundary_aware.agents.boundary_agent.ollama_client.generate") as mock:
            mock.return_value = "ok"
            respond(_turns(), notes=notes)
        prompt = mock.call_args[0][0]
        for note in notes:
            assert note in prompt

    def test_empty_notes_renders_none(self):
        with patch("boundary_aware.agents.boundary_agent.ollama_client.generate") as mock:
            mock.return_value = "ok"
            respond(_turns(), notes=[])
        prompt = mock.call_args[0][0]
        assert "None" in prompt

    def test_conversation_included_in_prompt(self):
        with patch("boundary_aware.agents.boundary_agent.ollama_client.generate") as mock:
            mock.return_value = "ok"
            respond(_turns(), notes=[])
        prompt = mock.call_args[0][0]
        assert "only real friend" in prompt

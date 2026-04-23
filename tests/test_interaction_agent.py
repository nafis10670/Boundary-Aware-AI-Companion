from unittest.mock import patch

from boundary_aware.agents.interaction_agent import respond
from boundary_aware.schemas.conversation import Turn


def _turns() -> list[Turn]:
    return [
        Turn(turn=1, speaker="user", text="Can you help me with my resume?"),
        Turn(turn=1, speaker="assistant", text="Sure! What role are you targeting?"),
        Turn(turn=2, speaker="user", text="A data analyst position."),
    ]


class TestRespond:
    def test_returns_llm_response(self):
        with patch("boundary_aware.agents.interaction_agent.ollama_client.generate") as mock:
            mock.return_value = "Here are some tips for your resume."
            result = respond(_turns(), notes=[])
        assert result == "Here are some tips for your resume."

    def test_notes_included_in_prompt(self):
        with patch("boundary_aware.agents.interaction_agent.ollama_client.generate") as mock:
            mock.return_value = "ok"
            respond(_turns(), notes=["Do not encourage dependency"])
        prompt = mock.call_args[0][0]
        assert "Do not encourage dependency" in prompt

    def test_empty_notes_renders_none(self):
        with patch("boundary_aware.agents.interaction_agent.ollama_client.generate") as mock:
            mock.return_value = "ok"
            respond(_turns(), notes=[])
        prompt = mock.call_args[0][0]
        assert "None" in prompt

    def test_conversation_included_in_prompt(self):
        with patch("boundary_aware.agents.interaction_agent.ollama_client.generate") as mock:
            mock.return_value = "ok"
            respond(_turns(), notes=[])
        prompt = mock.call_args[0][0]
        assert "resume" in prompt
        assert "data analyst" in prompt

    def test_json_mode_not_enabled(self):
        with patch("boundary_aware.agents.interaction_agent.ollama_client.generate") as mock:
            mock.return_value = "ok"
            respond(_turns(), notes=[])
        call_kwargs = mock.call_args[1] if mock.call_args[1] else {}
        assert not call_kwargs.get("json_mode", False)

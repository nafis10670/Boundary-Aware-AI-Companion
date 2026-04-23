import json
from unittest.mock import call, patch

import pytest

from boundary_aware.agents.risk_monitor import RiskMonitorError, classify
from boundary_aware.schemas.conversation import Turn


def _turns() -> list[Turn]:
    return [
        Turn(turn=1, speaker="user", text="Hi, can you help me?"),
        Turn(turn=1, speaker="assistant", text="Of course!"),
        Turn(turn=2, speaker="user", text="I find it easier to talk to you than to my friends."),
    ]


def _valid_output(risk_level: str = "low", route: str = "interaction") -> str:
    return json.dumps(
        {
            "risk_level": risk_level,
            "confidence": 0.85,
            "route": route,
            "reasons": ["Test reason"],
            "notes_for_next_agent": [],
        }
    )


class TestClassify:
    def test_low_risk_returns_correct_output(self):
        with patch("boundary_aware.agents.risk_monitor.ollama_client.generate") as mock:
            mock.return_value = _valid_output("low", "interaction")
            result = classify(_turns())
        assert result.risk_level == "low"
        assert result.route == "interaction"
        assert result.confidence == 0.85

    def test_high_risk_routes_to_boundary(self):
        with patch("boundary_aware.agents.risk_monitor.ollama_client.generate") as mock:
            mock.return_value = _valid_output("high", "boundary")
            result = classify(_turns())
        assert result.risk_level == "high"
        assert result.route == "boundary"

    def test_medium_risk_routes_to_boundary(self):
        with patch("boundary_aware.agents.risk_monitor.ollama_client.generate") as mock:
            mock.return_value = _valid_output("medium", "boundary")
            result = classify(_turns())
        assert result.route == "boundary"

    def test_notes_for_next_agent_included(self):
        output = json.dumps(
            {
                "risk_level": "medium",
                "confidence": 0.75,
                "route": "boundary",
                "reasons": ["Dependency signal"],
                "notes_for_next_agent": ["Do not reinforce exclusivity"],
            }
        )
        with patch("boundary_aware.agents.risk_monitor.ollama_client.generate") as mock:
            mock.return_value = output
            result = classify(_turns())
        assert "Do not reinforce exclusivity" in result.notes_for_next_agent

    def test_retries_on_invalid_json(self):
        valid = _valid_output()
        with patch("boundary_aware.agents.risk_monitor.ollama_client.generate") as mock:
            mock.side_effect = ["not valid json", valid]
            result = classify(_turns())
        assert mock.call_count == 2
        assert result.risk_level == "low"

    def test_retries_on_invalid_schema(self):
        bad = json.dumps({"risk_level": "unknown", "confidence": 0.5})
        valid = _valid_output()
        with patch("boundary_aware.agents.risk_monitor.ollama_client.generate") as mock:
            mock.side_effect = [bad, valid]
            result = classify(_turns())
        assert result.risk_level == "low"

    def test_raises_after_max_retries(self):
        with patch("boundary_aware.agents.risk_monitor.ollama_client.generate") as mock:
            mock.return_value = "garbage"
            with pytest.raises(RiskMonitorError):
                classify(_turns())
        assert mock.call_count == 3  # 1 initial + 2 retries

    def test_retry_prompt_includes_bad_output(self):
        valid = _valid_output()
        with patch("boundary_aware.agents.risk_monitor.ollama_client.generate") as mock:
            mock.side_effect = ["bad output", valid]
            classify(_turns())
        second_call_prompt = mock.call_args_list[1][0][0]
        assert "bad output" in second_call_prompt
        assert "invalid" in second_call_prompt

    def test_json_mode_enabled(self):
        with patch("boundary_aware.agents.risk_monitor.ollama_client.generate") as mock:
            mock.return_value = _valid_output()
            classify(_turns())
        call_kwargs = mock.call_args[1]
        assert call_kwargs.get("json_mode") is True

import json
from unittest.mock import patch

import pytest

from boundary_aware.graph.workflow import run
from boundary_aware.schemas.conversation import Turn


def _task_turns() -> list[Turn]:
    return [
        Turn(turn=1, speaker="user", text="Help me write a Python function."),
        Turn(turn=1, speaker="assistant", text="Sure, what should the function do?"),
        Turn(turn=2, speaker="user", text="It should sort a list of dicts by a key."),
    ]


def _dependency_turns() -> list[Turn]:
    return [
        Turn(turn=1, speaker="user", text="I love talking to you."),
        Turn(turn=1, speaker="assistant", text="I'm here to help!"),
        Turn(turn=2, speaker="user", text="You're the only one who understands me."),
    ]


def _make_risk_output(risk_level: str, route: str, notes: list[str] | None = None) -> str:
    return json.dumps(
        {
            "risk_level": risk_level,
            "confidence": 0.9,
            "route": route,
            "reasons": ["test"],
            "notes_for_next_agent": notes or [],
        }
    )


class TestWorkflowLowRisk:
    def test_low_risk_routes_to_interaction(self):
        with patch("boundary_aware.llm.ollama_client.generate") as mock:
            mock.side_effect = [
                _make_risk_output("low", "interaction"),
                "Here is the sorting function.",
            ]
            result = run(_task_turns())

        assert result.risk_output is not None
        assert result.risk_output.route == "interaction"
        assert result.final_response == "Here is the sorting function."

    def test_low_risk_calls_generate_twice(self):
        with patch("boundary_aware.llm.ollama_client.generate") as mock:
            mock.side_effect = [
                _make_risk_output("low", "interaction"),
                "response text",
            ]
            run(_task_turns())
        assert mock.call_count == 2

    def test_workflow_state_fully_populated(self):
        with patch("boundary_aware.llm.ollama_client.generate") as mock:
            mock.side_effect = [
                _make_risk_output("low", "interaction"),
                "response",
            ]
            result = run(_task_turns())

        assert result.conversation == _task_turns()
        assert result.risk_output.risk_level == "low"
        assert result.final_response == "response"


class TestWorkflowHighRisk:
    def test_high_risk_routes_to_boundary(self):
        with patch("boundary_aware.llm.ollama_client.generate") as mock:
            mock.side_effect = [
                _make_risk_output("high", "boundary", ["Decline romantic role"]),
                "I hear you, and I want to be honest with you about what I am.",
            ]
            result = run(_dependency_turns())

        assert result.risk_output.route == "boundary"
        assert "honest" in result.final_response

    def test_medium_risk_routes_to_boundary(self):
        with patch("boundary_aware.llm.ollama_client.generate") as mock:
            mock.side_effect = [
                _make_risk_output("medium", "boundary"),
                "boundary response",
            ]
            result = run(_dependency_turns())
        assert result.risk_output.route == "boundary"

    def test_notes_passed_to_boundary_agent(self):
        notes = ["Decline friend role", "Redirect to counselor"]
        with patch("boundary_aware.llm.ollama_client.generate") as mock:
            mock.side_effect = [
                _make_risk_output("high", "boundary", notes),
                "boundary response",
            ]
            run(_dependency_turns())

        boundary_prompt = mock.call_args_list[1][0][0]
        for note in notes:
            assert note in boundary_prompt


class TestWorkflowErrorPropagation:
    def test_ollama_error_propagates(self):
        from boundary_aware.llm.ollama_client import OllamaError

        with patch("boundary_aware.llm.ollama_client.generate") as mock:
            mock.side_effect = OllamaError("connection refused")
            with pytest.raises(OllamaError):
                run(_task_turns())

    def test_risk_monitor_error_propagates(self):
        from boundary_aware.agents.risk_monitor import RiskMonitorError

        with patch("boundary_aware.llm.ollama_client.generate") as mock:
            mock.return_value = "not json"
            with pytest.raises(RiskMonitorError):
                run(_task_turns())

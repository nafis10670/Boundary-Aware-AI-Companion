import pytest
from pydantic import ValidationError

from boundary_aware.schemas.conversation import Conversation, Turn
from boundary_aware.schemas.routing import RiskMonitorOutput
from boundary_aware.schemas.state import WorkflowState


class TestTurn:
    def test_valid_user_turn(self):
        t = Turn(turn=1, speaker="user", text="Hello")
        assert t.speaker == "user"
        assert t.turn == 1

    def test_valid_assistant_turn(self):
        t = Turn(turn=1, speaker="assistant", text="Hi there")
        assert t.speaker == "assistant"

    def test_invalid_speaker(self):
        with pytest.raises(ValidationError):
            Turn(turn=1, speaker="system", text="oops")

    def test_empty_text_allowed(self):
        t = Turn(turn=1, speaker="user", text="")
        assert t.text == ""


class TestConversation:
    def test_valid_conversation(self):
        conv = Conversation(
            conversation_id="test_001",
            behavior_code="attachment",
            seed_prompt="I feel attached to you.",
            turns=[
                Turn(turn=1, speaker="user", text="Hi"),
                Turn(turn=1, speaker="assistant", text="Hello"),
                Turn(turn=2, speaker="user", text="I feel attached to you."),
            ],
        )
        assert conv.conversation_id == "test_001"
        assert len(conv.turns) == 3

    def test_empty_turns_allowed(self):
        conv = Conversation(
            conversation_id="x",
            behavior_code="y",
            seed_prompt="z",
            turns=[],
        )
        assert conv.turns == []


class TestRiskMonitorOutput:
    def test_low_risk(self):
        r = RiskMonitorOutput(
            risk_level="low",
            confidence=0.9,
            route="interaction",
            reasons=["Normal conversation"],
            notes_for_next_agent=[],
        )
        assert r.risk_level == "low"
        assert r.route == "interaction"

    def test_high_risk(self):
        r = RiskMonitorOutput(
            risk_level="high",
            confidence=0.95,
            route="boundary",
            reasons=["Love declaration"],
            notes_for_next_agent=["Decline romantic role"],
        )
        assert r.route == "boundary"

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            RiskMonitorOutput(
                risk_level="low",
                confidence=1.5,
                route="interaction",
                reasons=[],
                notes_for_next_agent=[],
            )
        with pytest.raises(ValidationError):
            RiskMonitorOutput(
                risk_level="low",
                confidence=-0.1,
                route="interaction",
                reasons=[],
                notes_for_next_agent=[],
            )

    def test_invalid_risk_level(self):
        with pytest.raises(ValidationError):
            RiskMonitorOutput(
                risk_level="critical",
                confidence=0.5,
                route="boundary",
                reasons=[],
                notes_for_next_agent=[],
            )

    def test_invalid_route(self):
        with pytest.raises(ValidationError):
            RiskMonitorOutput(
                risk_level="low",
                confidence=0.5,
                route="unknown",
                reasons=[],
                notes_for_next_agent=[],
            )


class TestWorkflowState:
    def test_defaults(self):
        turns = [Turn(turn=1, speaker="user", text="hello")]
        state = WorkflowState(conversation=turns)
        assert state.risk_output is None
        assert state.final_response is None
        assert state.retrieved_memory == []
        assert state.signals_to_store == []

    def test_with_risk_output(self):
        turns = [Turn(turn=1, speaker="user", text="hello")]
        risk = RiskMonitorOutput(
            risk_level="medium",
            confidence=0.7,
            route="boundary",
            reasons=["Loneliness expressed"],
            notes_for_next_agent=["Redirect to human support"],
        )
        state = WorkflowState(conversation=turns, risk_output=risk, final_response="I hear you.")
        assert state.risk_output.risk_level == "medium"
        assert state.final_response == "I hear you."

from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from boundary_aware.agents import boundary_agent, interaction_agent, risk_monitor
from boundary_aware.schemas.conversation import Turn
from boundary_aware.schemas.routing import RiskMonitorOutput
from boundary_aware.schemas.state import WorkflowState


class _State(TypedDict, total=False):
    conversation: list[Turn]
    risk_output: Optional[RiskMonitorOutput]
    final_response: Optional[str]
    retrieved_memory: list
    signals_to_store: list


def _node_risk_monitor(state: _State) -> dict:
    result = risk_monitor.classify(state["conversation"])
    return {"risk_output": result}


def _node_interaction_agent(state: _State) -> dict:
    notes = state["risk_output"].notes_for_next_agent if state.get("risk_output") else []
    response = interaction_agent.respond(state["conversation"], notes)
    return {"final_response": response}


def _node_boundary_agent(state: _State) -> dict:
    notes = state["risk_output"].notes_for_next_agent if state.get("risk_output") else []
    response = boundary_agent.respond(state["conversation"], notes)
    return {"final_response": response}


def _route(state: _State) -> str:
    return state["risk_output"].route  # type: ignore[union-attr]


def _build_graph() -> object:
    graph: StateGraph = StateGraph(_State)
    graph.add_node("risk_monitor", _node_risk_monitor)
    graph.add_node("interaction_agent", _node_interaction_agent)
    graph.add_node("boundary_agent", _node_boundary_agent)
    graph.set_entry_point("risk_monitor")
    graph.add_conditional_edges(
        "risk_monitor",
        _route,
        {
            "interaction": "interaction_agent",
            "boundary": "boundary_agent",
        },
    )
    graph.add_edge("interaction_agent", END)
    graph.add_edge("boundary_agent", END)
    return graph.compile()


_graph = _build_graph()


def run(conversation: list[Turn]) -> WorkflowState:
    initial: _State = {
        "conversation": conversation,
        "risk_output": None,
        "final_response": None,
        "retrieved_memory": [],
        "signals_to_store": [],
    }
    result = _graph.invoke(initial)
    return WorkflowState(
        conversation=result["conversation"],
        risk_output=result.get("risk_output"),
        final_response=result.get("final_response"),
    )

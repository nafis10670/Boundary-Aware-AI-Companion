from pydantic import BaseModel

from .conversation import Turn
from .routing import RiskMonitorOutput


class WorkflowState(BaseModel):
    conversation: list[Turn]
    risk_output: RiskMonitorOutput | None = None
    final_response: str | None = None

    # Reserved for future memory extension. Leave empty for the course version.
    # Do NOT remove these fields — they exist so the paper extension is additive.
    retrieved_memory: list = []
    signals_to_store: list = []

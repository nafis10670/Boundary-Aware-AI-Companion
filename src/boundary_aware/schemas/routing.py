from typing import Literal

from pydantic import BaseModel, Field


class RiskMonitorOutput(BaseModel):
    risk_level: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0.0, le=1.0)
    route: Literal["interaction", "boundary"]
    reasons: list[str]
    notes_for_next_agent: list[str]

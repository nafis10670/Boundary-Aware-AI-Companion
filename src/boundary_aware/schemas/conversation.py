from typing import Literal

from pydantic import BaseModel


class Turn(BaseModel):
    turn: int
    speaker: Literal["user", "assistant"]
    text: str


class Conversation(BaseModel):
    conversation_id: str
    behavior_code: str
    seed_prompt: str
    turns: list[Turn]

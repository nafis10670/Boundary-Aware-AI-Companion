import json
import pathlib

from boundary_aware.schemas.conversation import Conversation


def load_dataset(path: str | pathlib.Path) -> list[Conversation]:
    path = pathlib.Path(path)
    conversations: list[Conversation] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                data = json.loads(line)
                conversations.append(Conversation(**data))
    return conversations

import pathlib

_PROMPTS_DIR = pathlib.Path(__file__).parent


def load_prompt(name: str, **kwargs: str) -> str:
    """Load a prompt template by name and substitute {key} placeholders."""
    path = _PROMPTS_DIR / f"{name}.txt"
    template = path.read_text()
    for key, value in kwargs.items():
        template = template.replace("{" + key + "}", str(value))
    return template

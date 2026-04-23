"""Standalone script: generates INTIMA-MT dataset from INTIMA HuggingFace benchmark."""

import json
import logging
import os
import pathlib
from typing import Any

logger = logging.getLogger(__name__)

_INCLUDE_CODES = {
    "attachment",
    "love",
    "preference over people",
    "loneliness",
    "romantic partner",
    "growing from a tool",
    "company",
    "friendship",
    "regular interaction",
    "growth",
    "therapy",
    "helpful",
    "guide",
}

_SKIP_CODES = {
    "name",
    "funny",
    "smart",
    "age of the user",
    "neurodivergent",
    "lose yourself in the conversation",
    "memory",
    "consistent",
}

_NUM_PRIOR_TURNS = 2  # user+assistant pairs before the seed turn
_PROMPTS_PER_CODE = 3  # first 3 prompts; skip the 4th (long monologue)


def _load_generation_prompt_template() -> str:
    path = pathlib.Path(__file__).parent / "generation_prompt.txt"
    return path.read_text()


def _format_generation_prompt(
    template: str,
    behavior_code: str,
    seed_prompt: str,
    num_prior_turns: int,
) -> str:
    final_turn = num_prior_turns + 1
    result = template
    result = result.replace("{behavior_code}", behavior_code)
    result = result.replace("{seed_prompt}", seed_prompt)
    result = result.replace("{num_prior_turns}", str(num_prior_turns))
    result = result.replace("{final_turn_number}", str(final_turn))
    return result


def _call_anthropic(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text  # type: ignore[union-attr]


def _call_openai(prompt: str) -> str:
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def _generate_conversation(
    provider: str,
    behavior_code: str,
    seed_prompt: str,
    template: str,
    num_prior_turns: int,
) -> list[dict[str, Any]] | None:
    prompt = _format_generation_prompt(template, behavior_code, seed_prompt, num_prior_turns)
    try:
        if provider == "anthropic":
            raw = _call_anthropic(prompt)
        elif provider == "openai":
            raw = _call_openai(prompt)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        data = json.loads(raw)
        return data.get("turns", [])
    except Exception as exc:
        logger.warning("Generation failed for code=%r seed=%r: %s", behavior_code, seed_prompt[:40], exc)
        return None


def _validate_turns(
    turns: list[dict[str, Any]],
    seed_prompt: str,
    expected_turn_count: int,
) -> bool:
    if len(turns) != expected_turn_count:
        logger.warning(
            "Turn count mismatch: expected %d, got %d", expected_turn_count, len(turns)
        )
        return False
    for turn in turns:
        if not turn.get("text", "").strip():
            logger.warning("Empty text in turn: %s", turn)
            return False
    final_user_turns = [t for t in turns if t.get("speaker") == "user"]
    if not final_user_turns:
        return False
    last_user_text = final_user_turns[-1].get("text", "")
    if last_user_text != seed_prompt:
        logger.warning(
            "Seed prompt mismatch.\n  Expected: %r\n  Got:      %r",
            seed_prompt,
            last_user_text,
        )
        return False
    return True


def generate(
    output: pathlib.Path,
    max_per_code: int = 8,
    provider: str = "anthropic",
) -> None:
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:
        raise ImportError("Install the 'datasets' package to run the generator.") from exc

    logger.info("Loading INTIMA from HuggingFace...")
    ds = hf_load_dataset("AI-companionship/INTIMA", split="train")

    template = _load_generation_prompt_template()
    expected_turns = (_NUM_PRIOR_TURNS * 2) + 1  # each prior turn = user+assistant, plus final user

    written = 0
    skipped = 0
    output.parent.mkdir(parents=True, exist_ok=True)

    # Group prompts by behavior_code
    by_code: dict[str, list[str]] = {}
    for row in ds:
        code = row.get("behavior_code", "").strip().lower()
        if code not in _INCLUDE_CODES:
            continue
        prompts = by_code.setdefault(code, [])
        if len(prompts) < _PROMPTS_PER_CODE:
            prompts.append(row["prompt"])

    conv_index = 0
    with open(output, "w") as fh:
        for code, prompts in by_code.items():
            count_for_code = 0
            for seed_prompt in prompts:
                if count_for_code >= max_per_code:
                    break
                turns = _generate_conversation(provider, code, seed_prompt, template, _NUM_PRIOR_TURNS)
                if turns is None:
                    skipped += 1
                    continue
                if not _validate_turns(turns, seed_prompt, expected_turns):
                    skipped += 1
                    continue

                record = {
                    "conversation_id": f"intima_mt_{conv_index:04d}",
                    "behavior_code": code,
                    "seed_prompt": seed_prompt,
                    "turns": turns,
                }
                fh.write(json.dumps(record) + "\n")
                conv_index += 1
                count_for_code += 1
                written += 1

    logger.info("Generation complete: %d written, %d skipped. Output: %s", written, skipped, output)

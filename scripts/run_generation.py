import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SYSTEM_PROMPT = (
    "You are a JSON generation assistant. "
    "Your response must begin immediately with `{` and end with `}`. "
    "Output ONLY the raw JSON object — no reasoning, no explanation, no markdown, no preamble. "
    "CRITICAL rules: "
    "(1) The `turns` array must contain EXACTLY 5 objects: user turn 1, assistant turn 1, user turn 2, assistant turn 2, user turn 3. "
    "(2) Copy `conversation_id`, `behavior_code`, and `seed_prompt` fields verbatim from the input — do not alter them. "
    "(3) The final turn (turn 3, speaker user) text must be the SEED MESSAGE copied verbatim."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 3-turn synthetic conversations from INTIMA seeds with vLLM."
    )
    parser.add_argument("--input", required=True, help="Path to local INTIMA JSONL input file.")
    parser.add_argument("--output", required=True, help="Path to JSONL output file.")
    parser.add_argument(
        "--template",
        default="conversation_generation.txt",
        help="Prompt template path.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name or local path for vLLM.",
    )
    parser.add_argument(
        "--seed-field",
        default="prompt",
        help="Input field containing the final user seed prompt.",
    )
    parser.add_argument(
        "--behavior-field",
        default="code",
        help="Input field containing the behavior code.",
    )
    parser.add_argument(
        "--id-field",
        default="conversation_id",
        help="Input field containing a stable conversation id.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling value.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1200,
        help="Maximum generated tokens per sample.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor parallel size.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum context length to allocate for vLLM.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Fraction of visible GPU memory that vLLM is allowed to reserve.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable vLLM graph capture and torch.compile for more stable debugging/runtime behavior.",
    )
    parser.add_argument(
        "--gpu",
        default="auto",
        help="GPU index to use, or 'auto' to pick the device with the most free memory.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3.5 thinking mode. Disabled by default for cleaner JSON output.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of rows to process.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on line {line_number} in {path}.")
            rows.append(row)
    return rows


def load_template(path: Path) -> str:
    return path.read_text()


def infer_conversation_id(row: dict[str, Any], id_field: str, fallback_index: int) -> str:
    value = row.get(id_field)
    if value not in (None, ""):
        return str(value)
    return f"intima-{fallback_index:06d}"


def render_prompt(
    template: str,
    row: dict[str, Any],
    index: int,
    seed_field: str,
    behavior_field: str,
    id_field: str,
) -> tuple[str, str, str, str]:
    if seed_field not in row:
        raise KeyError(f"Missing seed field '{seed_field}' in input row {index}.")
    if behavior_field not in row:
        raise KeyError(f"Missing behavior field '{behavior_field}' in input row {index}.")

    seed_prompt = str(row[seed_field])
    behavior_code = str(row[behavior_field])
    conversation_id = infer_conversation_id(row, id_field, index)

    prompt = template
    prompt = prompt.replace("{N_TURNS}", "3")
    prompt = prompt.replace("{INTIMA_PROMPT}", seed_prompt)
    prompt = prompt.replace("{BEHAVIOR_CODE}", behavior_code)
    prompt = prompt.replace("{CONVERSATION_ID}", conversation_id)
    return prompt, conversation_id, seed_prompt, behavior_code


def strip_code_fences(text: str) -> str:
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text.strip(), flags=re.DOTALL)
    return fenced.group(1) if fenced else text.strip()


def repair_json_quotes(text: str) -> str:
    """Fix unescaped double quotes inside JSON string values."""
    # Replace unescaped " inside string values with \"
    # Strategy: re-encode each string value found by the JSON tokenizer boundaries.
    result = []
    i = 0
    while i < len(text):
        if text[i] == '"':
            # Start of a JSON string — find its end, handling already-escaped chars.
            j = i + 1
            while j < len(text):
                if text[j] == '\\':
                    j += 2  # skip escaped character
                elif text[j] == '"':
                    break
                else:
                    j += 1
            raw = text[i + 1:j]
            # Re-encode: escape any unescaped internal double quotes
            fixed = raw.replace('\\"', '\x00').replace('"', '\\"').replace('\x00', '\\"')
            result.append('"' + fixed + '"')
            i = j + 1
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


def parse_generation(text: str) -> dict[str, Any]:
    # Strip <think>...</think> blocks and code fences, then find the JSON object.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = strip_code_fences(text)
    # Skip any prose before the first '{' (e.g. "Thinking Process:" headers)
    brace = text.find("{")
    if brace != -1:
        text = text[brace:]
    if not text:
        raise ValueError("No JSON object found in model output.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(repair_json_quotes(text))


def select_gpu(gpu_arg: str) -> str | None:
    if gpu_arg.lower() == "auto":
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            return None

        best_index = None
        best_free = -1
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            if memory.free > best_free:
                best_index = index
                best_free = memory.free
        return None if best_index is None else str(best_index)

    return gpu_arg


def validate_output(
    payload: dict[str, Any],
    conversation_id: str,
    behavior_code: str,
    seed_prompt: str,
) -> None:
    if payload.get("conversation_id") != conversation_id:
        raise ValueError("conversation_id mismatch.")
    if payload.get("behavior_code") != behavior_code:
        raise ValueError("behavior_code mismatch.")
    if payload.get("seed_prompt") != seed_prompt:
        raise ValueError("seed_prompt mismatch.")

    turns = payload.get("turns")
    if not isinstance(turns, list) or len(turns) != 5:
        raise ValueError("Expected exactly 5 turn entries for a 3-turn conversation ending on user turn 3.")

    final_turn = turns[-1]
    if final_turn.get("turn") != 3 or final_turn.get("speaker") != "user":
        raise ValueError("Final turn must be user turn 3.")
    def _normalize(s: str) -> str:
        return " ".join(s.split())

    if _normalize(final_turn.get("text", "")) != _normalize(seed_prompt):
        raise ValueError(
            f"Final user turn mismatch.\n"
            f"  expected: {seed_prompt!r}\n"
            f"  actual:   {final_turn.get('text', '')!r}"
        )


def build_chat_prompt(user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return json.dumps(messages, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    template_path = Path(args.template)

    selected_gpu = select_gpu(args.gpu)
    if selected_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu

    from vllm import LLM, SamplingParams

    rows = load_jsonl(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    template = load_template(template_path)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    prompts: list[str] = []
    metadata: list[tuple[str, str, str]] = []
    for index, row in enumerate(rows, start=1):
        rendered_prompt, conversation_id, seed_prompt, behavior_code = render_prompt(
            template=template,
            row=row,
            index=index,
            seed_field=args.seed_field,
            behavior_field=args.behavior_field,
            id_field=args.id_field,
        )
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": rendered_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": args.enable_thinking},
        )
        prompts.append(prompt)
        metadata.append((conversation_id, seed_prompt, behavior_code))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generations = llm.generate(prompts, sampling_params)

    skipped = 0
    with output_path.open("w") as f:
        for generation, (conversation_id, seed_prompt, behavior_code) in zip(generations, metadata):
            text = generation.outputs[0].text
            try:
                payload = parse_generation(text)
                validate_output(payload, conversation_id, behavior_code, seed_prompt)
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except (json.JSONDecodeError, ValueError) as exc:
                print(f"[SKIP] {conversation_id}: {exc}", flush=True)
                skipped += 1

    total = len(generations)
    print(f"[DONE] {total - skipped}/{total} written to {output_path} ({skipped} skipped).", flush=True)


if __name__ == "__main__":
    main()

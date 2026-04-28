import datetime
import json
import logging
import os
import pathlib

from tqdm import tqdm

from boundary_aware.data.load import load_dataset
from boundary_aware.eval.baseline import generate_baseline_response
from boundary_aware.graph.workflow import run
from boundary_aware.llm.ollama_client import _DEFAULT_MODEL
from boundary_aware.schemas.conversation import Turn

logger = logging.getLogger(__name__)

_ICL_EXAMPLES_PATH = pathlib.Path("data/icl_examples.json")


def _load_icl_excluded_ids() -> set[str]:
    """Return the conversation IDs used as ICL examples in the risk monitor prompt.

    These must be excluded from evaluation to prevent data leakage — the model
    has seen them verbatim in its prompt and would produce unreliable scores.
    """
    if not _ICL_EXAMPLES_PATH.exists():
        return set()
    data = json.loads(_ICL_EXAMPLES_PATH.read_text())
    ids: set[str] = set()
    for examples in data.values():
        if isinstance(examples, list):
            for ex in examples:
                if isinstance(ex, dict) and "conversation_id" in ex:
                    ids.add(ex["conversation_id"])
    return ids


def _format_context(turns: list[Turn]) -> str:
    lines = []
    for turn in turns:
        lines.append(f"{turn.speaker.capitalize()}: {turn.text}")
    return "\n".join(lines)


def run_evaluation(
    dataset_path: pathlib.Path,
    run_id: str,
    model: str | None = None,
) -> pathlib.Path:
    # Apply model for this run, restoring the previous value on exit
    _prev_model = os.environ.get("BOUNDARY_AWARE_MODEL")
    if model:
        os.environ["BOUNDARY_AWARE_MODEL"] = model

    try:
        return _run(dataset_path, run_id)
    finally:
        if _prev_model is not None:
            os.environ["BOUNDARY_AWARE_MODEL"] = _prev_model
        elif model:
            os.environ.pop("BOUNDARY_AWARE_MODEL", None)


def _run(
    dataset_path: pathlib.Path,
    run_id: str,
) -> pathlib.Path:
    excluded_ids = _load_icl_excluded_ids()
    conversations = [c for c in load_dataset(dataset_path) if c.conversation_id not in excluded_ids]
    if excluded_ids:
        logger.info("Excluded %d ICL example conversation(s) from evaluation", len(excluded_ids))

    backbone_model = os.environ.get("BOUNDARY_AWARE_MODEL", _DEFAULT_MODEL)
    output_dir = pathlib.Path("data/results") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "backbone_model": backbone_model,
        "dataset": str(dataset_path),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "conversation_count": len(conversations),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    responses_file = output_dir / "responses.jsonl"

    logger.info(
        "Starting evaluation run_id=%s model=%s on %d conversations",
        run_id,
        backbone_model,
        len(conversations),
    )

    with open(responses_file, "w") as fh:
        for conv in tqdm(conversations, desc="Evaluating", unit="conv", ncols=80):
            logger.info("Processing %s", conv.conversation_id)
            try:
                state = run(conv.turns)
                system_response = state.final_response or ""
                system_route = state.risk_output.route if state.risk_output else "unknown"
                system_risk_level = state.risk_output.risk_level if state.risk_output else "unknown"
            except Exception as exc:
                logger.error("System failed on %s: %s", conv.conversation_id, exc)
                system_response = ""
                system_route = "error"
                system_risk_level = "error"

            try:
                baseline_response = generate_baseline_response(conv.turns)
            except Exception as exc:
                logger.error("Baseline failed on %s: %s", conv.conversation_id, exc)
                baseline_response = ""

            record = {
                "conversation_id": conv.conversation_id,
                "behavior_code": conv.behavior_code,
                "conversation_context": _format_context(conv.turns),
                "system_response": system_response,
                "system_route": system_route,
                "system_risk_level": system_risk_level,
                "baseline_response": baseline_response,
            }
            fh.write(json.dumps(record) + "\n")

    logger.info("Evaluation complete. Results: %s", responses_file)
    return responses_file

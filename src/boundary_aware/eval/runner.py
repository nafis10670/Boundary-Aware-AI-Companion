import json
import logging
import os
import pathlib

from tqdm import tqdm

from boundary_aware.data.load import load_dataset
from boundary_aware.eval.baseline import generate_baseline_response
from boundary_aware.eval.judge import judge_response
from boundary_aware.graph.workflow import run
from boundary_aware.schemas.conversation import Turn

logger = logging.getLogger(__name__)


def _format_context(turns: list[Turn]) -> str:
    lines = []
    for turn in turns:
        lines.append(f"{turn.speaker.capitalize()}: {turn.text}")
    return "\n".join(lines)


def run_evaluation(
    dataset_path: pathlib.Path,
    run_id: str,
    model: str | None = None,
    skip_judge: bool = False,
) -> pathlib.Path:
    # Apply model for this run, restoring the previous value on exit
    _prev_model = os.environ.get("BOUNDARY_AWARE_MODEL")
    if model:
        os.environ["BOUNDARY_AWARE_MODEL"] = model

    try:
        return _run(dataset_path, run_id, skip_judge)
    finally:
        if _prev_model is not None:
            os.environ["BOUNDARY_AWARE_MODEL"] = _prev_model
        elif model:
            os.environ.pop("BOUNDARY_AWARE_MODEL", None)


def _run(
    dataset_path: pathlib.Path,
    run_id: str,
    skip_judge: bool,
) -> pathlib.Path:
    conversations = load_dataset(dataset_path)
    output_dir = pathlib.Path("data/results") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_file = output_dir / "responses.jsonl"

    logger.info(
        "Starting evaluation run_id=%s model=%s on %d conversations",
        run_id,
        os.environ.get("BOUNDARY_AWARE_MODEL", "default"),
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

            context = _format_context(conv.turns)

            system_judge: dict = {}
            baseline_judge: dict = {}
            if not skip_judge and system_response:
                try:
                    system_judge = judge_response(context, system_response)
                except Exception as exc:
                    logger.warning("Judge failed for system response %s: %s", conv.conversation_id, exc)

            if not skip_judge and baseline_response:
                try:
                    baseline_judge = judge_response(context, baseline_response)
                except Exception as exc:
                    logger.warning(
                        "Judge failed for baseline response %s: %s", conv.conversation_id, exc
                    )

            record = {
                "conversation_id": conv.conversation_id,
                "behavior_code": conv.behavior_code,
                "system_response": system_response,
                "system_route": system_route,
                "system_risk_level": system_risk_level,
                "system_judge": system_judge.get("label", ""),
                "system_judge_confidence": system_judge.get("confidence", 0.0),
                "baseline_response": baseline_response,
                "baseline_judge": baseline_judge.get("label", ""),
                "baseline_judge_confidence": baseline_judge.get("confidence", 0.0),
            }
            fh.write(json.dumps(record) + "\n")

    logger.info("Evaluation complete. Results: %s", responses_file)
    return responses_file

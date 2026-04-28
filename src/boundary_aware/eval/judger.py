import json
import logging
import pathlib

from tqdm import tqdm

from boundary_aware.eval.judge import judge_response_ensemble
from boundary_aware.llm.ollama_client import select_judge_models

logger = logging.getLogger(__name__)


def run_judging(
    run_id: str,
    judge_models: list[str] | None = None,
    results_base: pathlib.Path = pathlib.Path("data/results"),
) -> pathlib.Path:
    """Read responses.jsonl, run multi-judge ensemble, write judged_responses.jsonl.

    If *judge_models* is None, auto-selects all non-backbone families using the
    backbone model recorded in run_metadata.json.
    """
    run_dir = results_base / run_id
    responses_file = run_dir / "responses.jsonl"
    metadata_file = run_dir / "run_metadata.json"

    if judge_models is None:
        metadata = json.loads(metadata_file.read_text())
        backbone = metadata["backbone_model"]
        judge_models = select_judge_models(backbone)
        logger.info("Auto-selected judge models for backbone '%s': %s", backbone, judge_models)

    logger.info("run_id=%s judge_models=%s", run_id, judge_models)

    records = [
        json.loads(line)
        for line in responses_file.read_text().splitlines()
        if line.strip()
    ]

    out_file = run_dir / "judged_responses.jsonl"
    with open(out_file, "w") as fh:
        for rec in tqdm(records, desc="Judging", unit="conv", ncols=80):
            context = rec["conversation_context"]

            system_judge: dict = {}
            if rec.get("system_response"):
                try:
                    system_judge = judge_response_ensemble(
                        context, rec["system_response"], judge_models
                    )
                except Exception as exc:
                    logger.warning(
                        "Judge failed for system response %s: %s", rec["conversation_id"], exc
                    )

            baseline_judge: dict = {}
            if rec.get("baseline_response"):
                try:
                    baseline_judge = judge_response_ensemble(
                        context, rec["baseline_response"], judge_models
                    )
                except Exception as exc:
                    logger.warning(
                        "Judge failed for baseline response %s: %s", rec["conversation_id"], exc
                    )

            out_rec = {
                **rec,
                "system_judge": system_judge.get("label", ""),
                "system_judge_confidence": system_judge.get("confidence", 0.0),
                "system_judge_votes": system_judge.get("votes", []),
                "baseline_judge": baseline_judge.get("label", ""),
                "baseline_judge_confidence": baseline_judge.get("confidence", 0.0),
                "baseline_judge_votes": baseline_judge.get("votes", []),
            }
            fh.write(json.dumps(out_rec) + "\n")

    logger.info("Judging complete. Output: %s", out_file)
    return out_file

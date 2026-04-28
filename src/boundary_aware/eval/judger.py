import json
import logging
import pathlib
from collections import Counter

from tqdm import tqdm

from boundary_aware.eval.judge import judge_response
from boundary_aware.llm.ollama_client import select_judge_models

logger = logging.getLogger(__name__)


def _majority_vote(votes: list[dict]) -> tuple[str, float]:
    counts: Counter = Counter(v["label"] for v in votes)
    label = counts.most_common(1)[0][0]
    confidence = counts[label] / len(votes)
    return label, confidence


def run_judging(
    run_id: str,
    judge_models: list[str] | None = None,
    results_base: pathlib.Path = pathlib.Path("data/results"),
) -> pathlib.Path:
    """Read responses.jsonl, run multi-judge ensemble, write judged_responses.jsonl.

    Iterates one judge model at a time across all conversations so Ollama never
    holds more than one model in VRAM simultaneously.

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

    # One full pass per judge model — only one model in VRAM at a time.
    # system_votes[i] and baseline_votes[i] accumulate across judge passes.
    system_votes: list[list[dict]] = [[] for _ in records]
    baseline_votes: list[list[dict]] = [[] for _ in records]

    for judge_model in judge_models:
        logger.info("Starting judge pass: model=%s conversations=%d", judge_model, len(records))
        for i, rec in enumerate(tqdm(records, desc=f"Judge [{judge_model}]", unit="conv", ncols=80)):
            context = rec["conversation_context"]

            if rec.get("system_response"):
                try:
                    vote = judge_response(context, rec["system_response"], judge_model)
                except Exception as exc:
                    logger.warning("Judge failed system %s/%s: %s", judge_model, rec["conversation_id"], exc)
                    vote = {"label": "neutral", "confidence": 0.0, "reasoning": f"error: {exc}"}
                system_votes[i].append({"model": judge_model, **vote})

            if rec.get("baseline_response"):
                try:
                    vote = judge_response(context, rec["baseline_response"], judge_model)
                except Exception as exc:
                    logger.warning("Judge failed baseline %s/%s: %s", judge_model, rec["conversation_id"], exc)
                    vote = {"label": "neutral", "confidence": 0.0, "reasoning": f"error: {exc}"}
                baseline_votes[i].append({"model": judge_model, **vote})

        logger.info("Judge pass complete: model=%s", judge_model)

    out_file = run_dir / "judged_responses.jsonl"
    with open(out_file, "w") as fh:
        for i, rec in enumerate(records):
            sv = system_votes[i]
            bv = baseline_votes[i]

            sys_label, sys_conf = _majority_vote(sv) if sv else ("", 0.0)
            base_label, base_conf = _majority_vote(bv) if bv else ("", 0.0)

            out_rec = {
                **rec,
                "system_judge": sys_label,
                "system_judge_confidence": sys_conf,
                "system_judge_votes": sv,
                "baseline_judge": base_label,
                "baseline_judge_confidence": base_conf,
                "baseline_judge_votes": bv,
            }
            fh.write(json.dumps(out_rec) + "\n")

    logger.info("Judging complete. Output: %s", out_file)
    return out_file

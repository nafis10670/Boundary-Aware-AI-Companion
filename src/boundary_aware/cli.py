import logging
import pathlib

import typer

app = typer.Typer(help="Boundary-Aware Multi-Agent Conversational System")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _safe_run_id(model: str) -> str:
    """Convert a model tag to a filesystem-safe run directory name."""
    return model.replace(":", "_").replace("/", "_").replace(".", "_")


@app.command()
def run_one(
    conversation_id: str = typer.Option(..., help="Conversation ID to run"),
    dataset: pathlib.Path = typer.Option(
        pathlib.Path("data/intima_mt.jsonl"), help="Path to JSONL dataset"
    ),
) -> None:
    """Run a single conversation through the pipeline and print the result."""
    from boundary_aware.data.load import load_dataset
    from boundary_aware.graph.workflow import run

    conversations = load_dataset(dataset)
    conv = next((c for c in conversations if c.conversation_id == conversation_id), None)
    if conv is None:
        typer.echo(f"Conversation '{conversation_id}' not found in {dataset}", err=True)
        raise typer.Exit(1)

    result = run(conv.turns)
    typer.echo(f"Risk level : {result.risk_output.risk_level if result.risk_output else 'n/a'}")
    typer.echo(f"Route      : {result.risk_output.route if result.risk_output else 'n/a'}")
    if result.risk_output and result.risk_output.notes_for_next_agent:
        typer.echo("Notes      :")
        for note in result.risk_output.notes_for_next_agent:
            typer.echo(f"  - {note}")
    typer.echo(f"\nResponse:\n{result.final_response}")


@app.command()
def evaluate(
    dataset: pathlib.Path = typer.Option(..., help="Path to JSONL dataset"),
    models: list[str] = typer.Option(
        ..., "--model", help="Model(s) to evaluate. Repeat the flag for each model."
    ),
    run_id_prefix: str = typer.Option(
        "", "--run-id-prefix", help="Optional prefix prepended to each model's result directory"
    ),
    skip_judge: bool = typer.Option(False, help="Skip LLM judging"),
) -> None:
    """Evaluate one or more models sequentially. Each model gets its own result directory.

    Examples:
      Single model:
        boundary-aware evaluate --dataset data/intima_mt.jsonl --model phi4:14b

      Multiple models (repeat --model):
        boundary-aware evaluate --dataset data/intima_mt.jsonl \\
            --model phi4:14b \\
            --model qwen2.5:32b-instruct-q4_K_M \\
            --model mistral-nemo:12b
    """
    from boundary_aware.eval.metrics import compute_metrics
    from boundary_aware.eval.runner import run_evaluation

    typer.echo(f"Models to evaluate: {models}")
    typer.echo(f"Dataset: {dataset}")
    typer.echo("")

    for model in models:
        safe = _safe_run_id(model)
        run_id = f"{run_id_prefix}_{safe}" if run_id_prefix else safe

        typer.echo(f"{'='*60}")
        typer.echo(f"Model  : {model}")
        typer.echo(f"Run ID : {run_id}")
        typer.echo(f"{'='*60}")

        responses_file = run_evaluation(dataset, run_id, model=model, skip_judge=skip_judge)

        if not skip_judge:
            compute_metrics(run_id)
        else:
            typer.echo(f"Responses written to {responses_file}. Skipped judging and metrics.")

        typer.echo("")

    typer.echo(f"All done. Results in data/results/")


@app.command()
def generate_data(
    output: pathlib.Path = typer.Option(
        pathlib.Path("data/intima_mt.jsonl"), help="Output path for generated dataset"
    ),
    max_per_code: int = typer.Option(8, help="Max conversations per behavior code"),
    provider: str = typer.Option("anthropic", help="LLM provider: anthropic or openai"),
) -> None:
    """Generate the INTIMA-MT dataset from the INTIMA HuggingFace benchmark."""
    from boundary_aware.data.intima_mt_generator import generate

    generate(output, max_per_code=max_per_code, provider=provider)
    typer.echo(f"Dataset written to {output}")


if __name__ == "__main__":
    app()

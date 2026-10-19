import logging
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import typer
from langgraph.types import Command

from research_agent.config import get_settings
from research_agent.runs import Graph, get_status, stream_run
from research_agent.runtime import open_graph, run_async

log = logging.getLogger(__name__)

app = typer.Typer(help="Research a question on the web and get a short cited report.")

AutoApprove = Annotated[bool, typer.Option("--yes", "-y", help="Approve the drafted plan as is.")]
OutFile = Annotated[Path | None, typer.Option("--out", "-o", help="Also save the report here.")]


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def ask(question: str, yes: AutoApprove = False, out: OutFile = None) -> None:
    """Start a new run: draft a plan, review it, research, write the report."""
    thread_id = uuid4().hex
    say(f"run {thread_id}")
    run_async(_run(thread_id, {"question": question}, yes, out))


@app.command()
def resume(thread_id: str, yes: AutoApprove = False, out: OutFile = None) -> None:
    """Pick up a paused or interrupted run (needs DATABASE_URL)."""
    run_async(_resume(thread_id, yes, out))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8102) -> None:
    """Run the HTTP API."""
    import uvicorn

    from research_agent.api import create_app

    server = uvicorn.Server(uvicorn.Config(create_app(), host=host, port=port))
    run_async(server.serve())


async def _resume(thread_id: str, yes: bool, out: Path | None) -> None:
    async with open_graph(get_settings()) as graph:
        status = await get_status(graph, thread_id)
        if status is None:
            say(f"no run with id {thread_id}")
            raise typer.Exit(1)
        if status.status == "done":
            finish(status.report, status.stop_reason, out)
            return
        payload: Command | None = None
        if status.status == "awaiting_approval":
            payload = Command(resume=review_plan(status.question, status.plan, yes, thread_id))
        await _drive(graph, thread_id, payload, yes, out)


async def _run(thread_id: str, payload: dict[str, Any], yes: bool, out: Path | None) -> None:
    async with open_graph(get_settings()) as graph:
        await _drive(graph, thread_id, payload, yes, out)


async def _drive(
    graph: Graph,
    thread_id: str,
    payload: dict[str, Any] | Command | None,
    yes: bool,
    out: Path | None,
) -> None:
    while True:
        review: dict[str, Any] | None = None
        try:
            async for event in stream_run(graph, payload, thread_id):
                if event.type == "plan":
                    review = event.data
                elif event.type == "done":
                    finish(event.data["report"], event.data["stop_reason"], out)
                else:
                    say(f"- {event.message}")
        except Exception as exc:
            # Top of the CLI: show one readable line, keep the traceback for -v.
            log.debug("run failed", exc_info=True)
            say(f"run failed: {type(exc).__name__}: {exc}")
            if get_settings().database_url:
                say(f"progress is saved. retry with: research-agent resume {thread_id}")
            raise typer.Exit(1) from None
        if review is None:
            return
        payload = Command(resume=review_plan(review["question"], review["plan"], yes, thread_id))


def review_plan(question: str, plan: list[str], yes: bool, thread_id: str) -> dict[str, Any]:
    say(f"\nPlan for: {question}")
    for number, sub_question in enumerate(plan, 1):
        say(f"  {number}. {sub_question}")
    if yes:
        return {"approved": True}
    try:
        choice = typer.prompt("[a]pprove, [e]dit or [r]eject?", default="a", err=True)
        match choice.strip().lower()[:1]:
            case "e":
                return {"approved": True, "plan": read_plan()}
            case "r":
                return {"approved": False}
            case _:
                return {"approved": True}
    except typer.Abort:
        say(f"\nleft the plan waiting. pick it up with: research-agent resume {thread_id}")
        raise typer.Exit(130) from None


def read_plan() -> list[str]:
    say("One sub-question per line. Empty line when you're done.")
    lines: list[str] = []
    while line := typer.prompt(f"{len(lines) + 1}", default="", show_default=False, err=True):
        lines.append(line)
    return lines


def finish(report: str | None, stop_reason: str | None, out: Path | None) -> None:
    if not report:
        say(f"no report ({stop_reason})")
        return
    say(f"done ({stop_reason})\n")
    typer.echo(report)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        say(f"saved to {out}")


def say(message: str) -> None:
    typer.echo(message, err=True)

"""Turns raw graph updates into progress events that the CLI and the API both use."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel

from research_agent.state import ResearchState

type Graph = CompiledStateGraph[ResearchState]
type RunState = Literal["awaiting_approval", "incomplete", "done"]
# A new question, a reply to the plan interrupt, or None to continue from the checkpoint.
type RunInput = ResearchState | Command | None


@dataclass(frozen=True)
class Event:
    type: Literal["progress", "plan", "done", "paused"]
    data: dict[str, Any] = field(default_factory=dict)
    message: str = ""


class RunStatus(BaseModel):
    thread_id: str
    status: RunState
    question: str
    plan: list[str] = []
    rounds: int = 0
    pages_read: int = 0
    tokens_used: int = 0
    stop_reason: str | None = None
    report: str | None = None


def thread_config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


async def stream_run(
    graph: Graph, payload: RunInput, thread_id: str
) -> AsyncGenerator[Event, None]:
    """Run until the graph pauses for a plan review or finishes."""
    config = thread_config(thread_id)
    # "sync" saves each step before starting the next, so a request that gets killed or cut
    # off can only lose the step it was on.
    async for chunk in graph.astream(payload, config, stream_mode="updates", durability="sync"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                yield Event("plan", dict(update[0].value) | {"thread_id": thread_id})
            else:
                yield progress_event(node, update or {})

    status = await get_status(graph, thread_id)
    if status is not None and status.status == "done":
        yield Event("done", status.model_dump())


async def stream_slice(
    graph: Graph, payload: RunInput, thread_id: str, seconds: float
) -> AsyncIterator[Event]:
    """Like `stream_run`, but stops after `seconds` and leaves the run ready to continue.

    A serverless request has a hard time limit and a run can take longer than that. Steps
    that finished are already saved and whatever was in flight is cancelled, so passing
    None in a later request picks up from there. It ends with a `paused` event.
    """
    deadline = asyncio.get_running_loop().time() + seconds
    async with aclosing(stream_run(graph, payload, thread_id)) as events:
        while True:
            limit = asyncio.timeout_at(deadline)
            try:
                async with limit:
                    event = await anext(events)
            except StopAsyncIteration:
                return
            except TimeoutError:
                # A node can time out too, and that one is a real failure.
                if not limit.expired():
                    raise
                yield Event("paused", {"thread_id": thread_id}, "ran out of time, continuing")
                return
            yield event


async def get_status(graph: Graph, thread_id: str) -> RunStatus | None:
    snapshot = await graph.aget_state(thread_config(thread_id))
    values = snapshot.values
    if not values:
        return None
    if snapshot.interrupts:
        status = "awaiting_approval"
    elif snapshot.next:
        status = "incomplete"
    else:
        status = "done"
    return RunStatus(
        thread_id=thread_id,
        status=status,
        question=values["question"],
        plan=values.get("plan", []),
        rounds=values.get("rounds", 0),
        pages_read=len(values.get("read_urls", [])),
        tokens_used=values.get("tokens_used", 0),
        stop_reason=values.get("stop_reason"),
        report=values.get("report"),
    )


def progress_event(node: str, update: dict[str, Any]) -> Event:
    data: dict[str, Any] = {"node": node}
    match node:
        case "plan":
            data["plan"] = update.get("plan", [])
            message = f"drafted a plan with {plural(len(data['plan']), 'sub-question')}"
        case "review_plan" if update.get("approved"):
            data["plan"] = update.get("plan", [])
            message = f"plan approved ({plural(len(data['plan']), 'sub-question')})"
        case "review_plan":
            data["stop_reason"] = update.get("stop_reason")
            message = f"stopped: {data['stop_reason']}"
        case "research":
            [question] = update.get("searched", ["?"])
            data |= {
                "sub_question": question,
                "pages_read": len(update.get("read_urls", [])),
                "notes": len(update.get("notes", [])),
            }
            message = (
                f'searched "{question}": read {plural(data["pages_read"], "page")}, '
                f"kept {plural(data['notes'], 'note')}"
            )
        case "reflect":
            data |= {
                "round": update.get("rounds"),
                "follow_up": update.get("pending", []),
                "stop_reason": update.get("stop_reason"),
            }
            if data["follow_up"]:
                message = f"round {data['round']} done, following up on: " + "; ".join(
                    data["follow_up"]
                )
            else:
                message = f"round {data['round']} done, stopping: {data['stop_reason']}"
        case "write":
            message = "wrote the report"
        case _:
            message = node
    return Event("progress", data, message)


def plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"

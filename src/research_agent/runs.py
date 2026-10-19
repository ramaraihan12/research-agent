"""Turns raw graph updates into progress events that the CLI and the API both use."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel

from research_agent.state import ResearchState

type Graph = CompiledStateGraph[ResearchState]


@dataclass(frozen=True)
class Event:
    type: Literal["progress", "plan", "done"]
    data: dict[str, Any] = field(default_factory=dict)
    message: str = ""


class RunStatus(BaseModel):
    thread_id: str
    status: Literal["awaiting_approval", "incomplete", "done"]
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
    graph: Graph, payload: dict[str, Any] | Command | None, thread_id: str
) -> AsyncIterator[Event]:
    """Run until the graph pauses for a plan review or finishes."""
    config = thread_config(thread_id)
    async for chunk in graph.astream(payload, config, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                yield Event("plan", dict(update[0].value) | {"thread_id": thread_id})
            else:
                yield progress_event(node, update or {})

    status = await get_status(graph, thread_id)
    if status is not None and status.status == "done":
        yield Event("done", status.model_dump())


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
            message = f"drafted a plan with {len(data['plan'])} sub-questions"
        case "review_plan" if update.get("approved"):
            data["plan"] = update.get("plan", [])
            message = f"plan approved ({len(data['plan'])} sub-questions)"
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
                f'searched "{question}": read {data["pages_read"]} pages, '
                f"kept {data['notes']} notes"
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

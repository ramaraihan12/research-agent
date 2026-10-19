import logging
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langgraph.types import Command
from pydantic import BaseModel, Field

from research_agent.config import get_settings
from research_agent.runs import Graph, RunStatus, get_status, stream_run
from research_agent.runtime import open_graph
from research_agent.state import PlanReview

log = logging.getLogger(__name__)


class StartRun(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def create_app(graph: Graph | None = None) -> FastAPI:
    """Build the API. Tests pass a ready graph; otherwise one is built from settings."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if graph is not None:
            app.state.graph = graph
            yield
            return
        async with open_graph(get_settings()) as built:
            app.state.graph = built
            yield

    app = FastAPI(title="research-agent", lifespan=lifespan)

    @app.post("/runs", response_class=EventSourceResponse)
    async def start_run(
        body: StartRun, graph: Annotated[Graph, Depends(get_graph)]
    ) -> AsyncIterable[ServerSentEvent]:
        """Draft a plan for a new question. The stream ends with a `plan` event."""
        thread_id = uuid4().hex
        yield ServerSentEvent(event="run", data={"thread_id": thread_id})
        async for event in sse_events(graph, {"question": body.question}, thread_id):
            yield event

    @app.post("/runs/{thread_id}/plan", response_class=EventSourceResponse)
    async def review_plan(
        review: PlanReview,
        thread_id: Annotated[str, Depends(paused_run)],
        graph: Annotated[Graph, Depends(get_graph)],
    ) -> AsyncIterable[ServerSentEvent]:
        """Approve, edit or reject the plan. Streams research progress, then `done`."""
        async for event in sse_events(graph, Command(resume=review.model_dump()), thread_id):
            yield event

    @app.get("/runs/{thread_id}")
    async def get_run(thread_id: str, graph: Annotated[Graph, Depends(get_graph)]) -> RunStatus:
        status = await get_status(graph, thread_id)
        if status is None:
            raise HTTPException(404, "no run with that id")
        return status

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def get_graph(request: Request) -> Graph:
    return request.app.state.graph


async def paused_run(thread_id: str, graph: Annotated[Graph, Depends(get_graph)]) -> str:
    # Checked in a dependency so a bad id gets a real 404/409 before the stream starts.
    status = await get_status(graph, thread_id)
    if status is None:
        raise HTTPException(404, "no run with that id")
    if status.status != "awaiting_approval":
        raise HTTPException(409, f"run is {status.status}, not waiting for a plan review")
    return thread_id


async def sse_events(
    graph: Graph, payload: dict[str, Any] | Command, thread_id: str
) -> AsyncIterator[ServerSentEvent]:
    try:
        async for event in stream_run(graph, payload, thread_id):
            data = event.data | ({"message": event.message} if event.message else {})
            yield ServerSentEvent(event=event.type, data=data)
    except Exception as exc:
        # Headers are already sent, so the only way to report this is in the stream.
        log.exception("run %s failed", thread_id)
        yield ServerSentEvent(event="error", data={"message": f"{type(exc).__name__}: {exc}"})
